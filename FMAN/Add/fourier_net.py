import torch
from torch import nn
import math
import gin
from .network import OFENet, Projection, Projection2
import learn2learn as l2l


@gin.configurable
class FourierNet(nn.Module):
    def __init__(self, dim_state, dim_action, dim_output, dim_discretize,
                 total_units, num_layers, batchnorm,
                 fourier_type, discount, projection_dim=256,  # 投影层的维度在256维
                 cosine_similarity=True, normalizer="batch",
                 activation=nn.ReLU(), block="densenet",
                 trainable=False, gpu=0,
                 skip_action_branch=False):
        super(FourierNet, self).__init__()
        # 初始化off net网络  用于从输入状态和动作中提取特征
        self.ofe_net = OFENet(dim_state, dim_action, dim_output, dim_discretize,
                              total_units, num_layers, batchnorm, normalizer,
                              activation, block, trainable, gpu, skip_action_branch)
        self.dim_state_features = self.ofe_net.dim_state_features

        # 用l2l中的maml算法包裹offnet使他支持ml
        self.ofe_net_original = l2l.algorithms.MAML(self.ofe_net, lr=3e-4, first_order=False)
        self.ofe_net_active = self.ofe_net_original

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_discretize = dim_discretize
        self.dim_output = dim_output
        self.end = int(dim_discretize * 0.5 + 1)

        self.projection = Projection(self.end, self.dim_state, output_dim=projection_dim, normalizer=normalizer)
        self.projection2 = Projection2(output_dim=projection_dim, normalizer=normalizer)

        # 把两个projection包裹在l2l算法内部
        self.projection_original = l2l.algorithms.MAML(self.projection, lr=3e-4, first_order=False)
        self.projection2_original = l2l.algorithms.MAML(self.projection2, lr=3e-4, first_order=False)

        self.projection_active = self.projection_original
        self.projection2_active = self.projection2_original

        self.fourier_type = fourier_type
        self.projection_dim = projection_dim
        self.cosine_similarity = cosine_similarity

        self.device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")

        # 傅里叶变换的部分
        # dim_discretize 代表了信号的离散化粒度
        # ratio表示每个离散化的步骤对应的角度增量（弧度）
        ratio = 2 * math.pi / dim_discretize
        # con 是一个张量，包含了所有离散化点对应的角度（弧度）
        con = torch.tensor([k * ratio for k in range(self.end)], device=self.device)  # something is wrong about GPU
        self.Gamma_re = discount * torch.diag(torch.cos(con))
        self.Gamma_im = -discount * torch.diag(torch.sin(con))

        ratio2 = 1.0 / (2 * math.pi * dim_discretize)
        con_re = torch.tensor([1] + [2 * math.cos(k * ratio) for k in range(1, self.end - 1)] + [-1],
                              device=self.device)
        self.con_re = ratio2 * con_re.unsqueeze(0)
        con_im = torch.tensor([0] + [-2 * math.sin(k * ratio) for k in range(1, self.end - 1)] + [0],
                              device=self.device)
        self.con_im = ratio2 * con_im.unsqueeze(0)

        self.mode = "Validation"

    def clone_for_adaptation(self):
        #mm = self.ofe_net_active  # 保存当前的 ofe_net_active
        self.ofe_net_active = self.ofe_net_original.clone()
        self.projection_active = self.projection_original.clone()
        self.projection2_active = self.projection2_original.clone()

        # # 比较 mm 和 self.ofe_net_active 的参数形状
        # for (param_mm, param_active) in zip(mm.parameters(), self.ofe_net_active.parameters()):
        #     assert param_mm.shape == param_active.shape, \
        #         f"Shape mismatch: param_mm.shape = {param_mm.shape}, param_active.shape = {param_active.shape}"
        # print("这部分参数是没问题的")
        # If no mismatch, continue with the adaptation mode
        self.mode = "Adaptation"

    # 将活动网络重置为原始网络
    def reset_cloned_networks(self):
        self.ofe_net_active = self.ofe_net_original
        self.projection_active = self.projection_original
        self.projection2_active = self.projection2_original

        self.mode = "Validation"

    # 从offnet中提取状态特征和 状态-动作特征
    def features_from_states(self, states):
        return self.ofe_net.features_from_states(states)

    def features_from_states_actions(self, states, actions):
        return self.ofe_net.features_from_states_actions(states, actions)

    def loss(self, y_target, y, target_model=None):
        trun = 15

        if target_model is not None:
            y_target2 = target_model.projection_active(y_target[:, trun:self.end - trun, :])
            _ = target_model.projection2_active(y_target2)
            y2 = self.projection_active(y[:, trun:self.end - trun, :])
            y2 = self.projection2_active(y2)

        if self.cosine_similarity:
            loss_fun = nn.CosineSimilarity(dim=-1)
            loss1 = torch.mean(loss_fun(y_target[:, :trun, :], y[:, :trun, :]))
            loss2 = torch.mean(loss_fun(y_target2, y2))
            loss3 = torch.mean(loss_fun(y_target[:, self.end - trun:self.end, :], y[:, self.end - trun:self.end, :]))

            loss = loss1 + loss2 + loss3
        else:
            loss = nn.functional.mse_loss(y_target2, y2)

        return loss

    def forward(self, states, actions=None):

        predictor_re, predictor_im = self.ofe_net(states, actions)

        return predictor_re, predictor_im

    def compute_loss(self, target_model, states, actions, next_states, next_actions, dones, Hth_states=None):
        dones = dones.unsqueeze(-1).repeat(1, self.end, self.dim_state).float()
        O = next_states[:, :self.dim_state].unsqueeze(1).repeat(1, self.end, 1)

        predicted_re, predicted_im = self(states, actions)
        next_predicted_re, next_predicted_im = target_model(next_states, next_actions)

        if self.fourier_type == 'dtft':
            with torch.no_grad():
                y_target_re = O + (torch.matmul(self.Gamma_re, next_predicted_re) - torch.matmul(self.Gamma_im,
                                                                                                 next_predicted_im)) * (
                                          1 - dones)
            y_re = predicted_re
            pred_re_loss = self.loss(y_target_re.detach(), y_re, target_model)

            with torch.no_grad():
                y_target_im = (torch.matmul(self.Gamma_im, next_predicted_re) + torch.matmul(self.Gamma_re,
                                                                                             next_predicted_im)) * (
                                          1 - dones)
            y_im = predicted_im
            pred_im_loss = self.loss(y_target_im.detach(), y_im, target_model)

        elif self.fourier_type == 'dft':
            y_target_re = O - self.coef * torch.matmul(self.con_re.unsqueeze(-1), Hth_states.unsqueeze(1)) + \
                          (torch.matmul(self.Gamma_re, next_predicted_re) - torch.matmul(self.Gamma_im,
                                                                                         next_predicted_im)) * (
                                      1 - dones)
            y_re = predicted_re
            pred_re_loss = self.loss(y_target_re.detach(), y_re, target_model)

            y_target_im = self.coef * torch.matmul(self.con_re.unsqueeze(-1), Hth_states.unsqueeze(1)) + \
                          (torch.matmul(self.Gamma_im, next_predicted_re) + torch.matmul(self.Gamma_re,
                                                                                         next_predicted_im)) * (
                                      1 - dones)
            y_im = predicted_im
            pred_im_loss = self.loss(y_target_im.detach(), y_im, target_model)

        else:
            raise ValueError(f"Invalid fourier_type: {self.fourier_type}")

        pred_loss = pred_re_loss + pred_im_loss
        return pred_loss

    def adapt(self, loss):
        #这里似乎有些区别
        # self.ofe_net_active.adapt(loss)
        # self.projection_active.adapt(loss)
        # self.projection2_active.adapt(loss)

        self.ofe_net_active.adapt(loss, allow_nograd=True, allow_unused=True)
        self.projection_active.adapt(loss, allow_nograd=True, allow_unused=True)
        self.projection2_active.adapt(loss, allow_nograd=True, allow_unused=True)