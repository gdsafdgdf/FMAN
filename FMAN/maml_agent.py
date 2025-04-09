import os
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import numpy as np
import time
import learn2learn as l2l
from policy.PPO import GaussianActor, CriticV, PPO, RolloutBuffer  # PPO components
from blocks.maml_config import Metaworld

from Add.fourier_net import FourierNet
class MAML_agent(nn.Module):
    def __init__(self, env, agent_config, maml_config):
        super(MAML_agent, self).__init__()
        self.env = env
        self.eval_env = env

        self.args = agent_config

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.replay_buffer = RolloutBuffer(obs_dim=state_dim, act_dim=action_dim, capacity=agent_config.steps_per_epoch, gamma=agent_config.discount, lam=agent_config.lam)

        # 设置特征提取器的参数
        self.extractor_kwargs = {
            "dim_state": state_dim,
            "dim_action": action_dim,
            "dim_output": 11,
            "dim_discretize": agent_config.dim_discretize,
            "total_units": 240,
            "num_layers": 6,
            "batchnorm": True,
            "fourier_type": agent_config.fourier_type,
            "discount": agent_config.discount,
            "projection_dim": agent_config.projection_dim,
            "cosine_similarity": agent_config.cosine_similarity,
            "normalizer": agent_config.normalizer
        }#add
        # Initialize the extractor and target extractor network  相当于搞了两个相同的特征提取器
        self.extractor = FourierNet(**self.extractor_kwargs, skip_action_branch=False)  # 源网络
        self.target_extractor = FourierNet(**self.extractor_kwargs, skip_action_branch=False)  ##目标网络
        self.soft_update(self.target_extractor, self.extractor, tau=1)#add

        #add
        #从这起调整actor和critic的输入维度
        self.actor = GaussianActor(
            self.extractor.dim_state_features, action_dim, max_action, layer_units=(64, 64))
        self.critic = CriticV(
            self.extractor.dim_state_features, units=(64, 64))
        
        self.critic = l2l.algorithms.MAML(self.critic, lr = 1e-3)

        self.policy = PPO(
            feature_extractor=self.extractor, #add
            actor=self.actor,
            critic=self.critic,
            pi_lr=3e-4,
            vf_lr=1e-3,
            clip_ratio=0.2,
            batch_size=64,
            discount=0.99,
            n_epoch=10,
            horizon=2048,
            gpu=0)
        
    def soft_update(self, target, source, tau=1):
        for target_param, source_param in zip(target.state_dict().values(), source.state_dict().values()):
            if isinstance(target_param, torch.Tensor) and isinstance(source_param, torch.Tensor):
                target_param.copy_((1 - tau) * target_param + tau * source_param)

    def get_action(self, state, action=None, return_distribution=False):
        # 利用特征提取器获得当前状态的特征
        state_features = self.extractor.features_from_states(state)

        # Get the action mean and logstd from the PPO actor
        action_mean, action_logstd = self.actor(state_features)  # 得到了动作的均值和对数标准差

        # 检查并调整 action_logstd 形状，使其与 action_mean 形状一致
        if action_mean.shape != action_logstd.shape:
            # 扩展 action_logstd 的形状以匹配 action_mean
            action_logstd = action_logstd.expand_as(
                action_mean)  # 或者 action_logstd = action_logstd.repeat(1, action_mean.size(1))
        # Calculate the standard deviation from the log standard deviation
        action_std = torch.exp(action_logstd)

        # Create the normal distribution
        distribution = Normal(action_mean, action_std)

        # Sample action or return log probability
        if action is None:
            action = distribution.sample()
            logprob = distribution.log_prob(action).sum(1)
        else:
            logprob = distribution.log_prob(action).sum(1)

        # Return action, log probability, entropy, and optionally the distribution
        if not return_distribution:
            return action, logprob, distribution.entropy().sum(1)
        else:
            return action, logprob, distribution.entropy().sum(1), distribution

    def get_action1(self, state, action=None, return_distribution=False):
        # state 是一个大小为 (4000, 39) 的张量
        batch_size, num_features = state.size()  # batch_size = 4000, num_features = 39

        # 用一个列表来存储每个样本的特征
        all_state_features = []

        # 遍历每一行（每个样本）并调用 features_from_states 提取每一维的特征
        for i in range(batch_size):
            single_state = state[i:i + 1, :]  # 取出 state 的第 i 个样本，形状是 (1, 39)
            state_feature = self.extractor.features_from_states(single_state)  # 提取特征
            all_state_features.append(state_feature)

        # 将提取出的所有特征合并回 4000 批次
        state_features = torch.cat(all_state_features, dim=0)  # 合并为 (4000, feature_dim)

        # Get the action mean and logstd from the PPO actor
        action_mean, action_logstd = self.actor(state_features)  # 得到了动作的均值和对数标准差

        # 检查并调整 action_logstd 形状，使其与 action_mean 形状一致
        if action_mean.shape != action_logstd.shape:
            # 扩展 action_logstd 的形状以匹配 action_mean
            action_logstd = action_logstd.expand_as(
                action_mean)  # 或者 action_logstd = action_logstd.repeat(1, action_mean.size(1))
        # Calculate the standard deviation from the log standard deviation
        action_std = torch.exp(action_logstd)

        # Create the normal distribution
        distribution = Normal(action_mean, action_std)

        # Sample action or return log probability
        if action is None:
            action = distribution.sample()
            logprob = distribution.log_prob(action).sum(1)
        else:
            logprob = distribution.log_prob(action).sum(1)

        # Return action, log probability, entropy, and optionally the distribution
        if not return_distribution:
            return action, logprob, distribution.entropy().sum(1)
        else:
            return action, logprob, distribution.entropy().sum(1), distribution

    def adapt(self, num_steps, information, config:Metaworld, lifetime_buffer, mean_reward_for_baseline, device):

        max_steps = config.num_env_steps_per_adaptation_update
        epochs = max_steps // self.args.steps_per_epoch + 1 

        total_steps = self.args.steps_per_epoch*epochs
        steps_per_epoch = self.args.steps_per_epoch

        #重置环境，初始化回合的总奖励和时间步数
        state, _ = self.env.reset()
        episode_return = 0
        episode_timesteps = 0

        #add
        round = 0
        # 采样若干条轨迹
        for i in range(self.args.random_collect):
            #print("Collect Round: ", round)
            round += 1
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_timesteps += 1

            done_flag = done
            if episode_timesteps == config.max_episode_steps:
                done_flag = False

            self.replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=0, logp=0)
            state = next_state

            if done:
                state, _ = self.env.reset()
                episode_timesteps = 0
                episode_return = 0

        # No need to pretrain the extractor!
        self.policy.clone_for_adaptation()
        self.policy.set_optimizers()
        
        #利用随机采样的数据预训练提取器 add
        #print("Pretrain: I am pretraining the extractor!")
        pretrain_step = 0
        for i in range(self.args.pre_train_step):
            #print("Pretrain Step: ", pretrain_step)
            pretrain_step += 1
            sample_states, sample_actions, sample_next_states, _, sample_dones = self.replay_buffer.sample(
                batch_size=16)
            sample_next_actions, _ = self.policy.get_action(sample_next_states)
            # 适应提取器
            pred_loss = self.policy.extractor_loss_for_adaptation(self.target_extractor,
                                                                  sample_states, sample_actions,
                                                                  sample_next_states, sample_next_actions,
                                                                  sample_dones)

            self.policy.extractor_adapt(pred_loss)

        state = np.array(state, dtype=np.float32)  # ??????

        self.replay_buffer.get() #这一步干嘛的，可能是为了清空replay_buffer中的内容吧 add

        #生命周期缓冲区初始化
        done_lifetime = information['prev_done']
        episodes_returns_lifetime=[]
        episodes_successes_lifetime=[]
        episode_return_lifetime = information['current_episode_return']
        succeeded_in_episode_lifetime=information['current_episode_success']
        current_lifetime_step = information['current_lifetime_step']
        episode_step_num=information['current_episode_step_num']  # Adding current_episode_step_num
        hidden_state=information['hidden_state']  # Adding hidden_state
        current_state = information['current_state']  # Adding current_state

        step = 0

        #初始化actor混合精度训练所需的gradscaler
        actor_scaler = torch.amp.GradScaler()

        for cur_steps in range(total_steps):
            prev_done_lifetime = done_lifetime
            # prev_current_state = current_state

            step += 1
            action, logp, v_t = self.policy.get_action_and_val(state) #从当前策略获得动作  动作概率  状态值估计

            # Step the env
            next_state, reward, terminated, truncated, info = self.env.step(action) #在环境中执行这个动作
            done = terminated or truncated
            #组合这里torch,Tensor不支持布尔型变量和float组合
            done_lifetime = torch.tensor(done, dtype=torch.float32).to(device)
            current_state = torch.tensor(next_state, dtype=torch.float32).to(device)
            episode_timesteps += 1
            episode_return += reward

            done_flag = done

            self.replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=v_t, logp=logp)

            lifetime_buffer.store_step_data(
                global_step=current_lifetime_step,
                obs=state.clone().detach().to(device).float() if isinstance(state, torch.Tensor) else torch.tensor(
                    state, dtype=torch.float32).to(device),
                act=action.clone().detach().to(device).float() if isinstance(action, torch.Tensor) else torch.tensor(
                    action, dtype=torch.float32).to(device),
                reward=reward.clone().detach().to(device).float() if isinstance(reward, torch.Tensor) else torch.tensor(
                    reward, dtype=torch.float32).to(device),
                logp=logp.clone().detach().to(device).float() if isinstance(logp, torch.Tensor) else torch.tensor(logp,
                                                                                                                  dtype=torch.float32).to(
                    device),
                prev_done=prev_done_lifetime.clone().detach().to(device).float() if isinstance(prev_done_lifetime,
                                                                                               torch.Tensor) else torch.tensor(
                    prev_done_lifetime, dtype=torch.float32).to(device)
            )

            state = next_state
            current_lifetime_step += 1
            episode_step_num += 1 #完全是为了符合information的格式搞的
            episode_return_lifetime += torch.as_tensor(reward, dtype=torch.float32).to(device)
            if info['success'] == 1.0:
                succeeded_in_episode_lifetime = True

            if done or (episode_timesteps == config.max_episode_steps) or (cur_steps + 1) % steps_per_epoch == 0:
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if done else self.policy.get_val(state)
                self.replay_buffer.finish_path(last_val)
                state, _ = self.env.reset()
                episode_timesteps = 0
                episode_return = 0

                episodes_returns_lifetime.append(episode_return_lifetime)
                episode_return_lifetime = 0
                if succeeded_in_episode_lifetime == True:
                    episodes_successes_lifetime.append(1.0)
                else:
                    episodes_successes_lifetime.append(0.0)
                done_lifetime = torch.ones(1).to(device) #add
                current_state = torch.tensor(self.env.reset()[0],dtype=torch.float32).to(device) #add
                succeeded_in_episode_lifetime = False #add

            if (cur_steps + 1) % steps_per_epoch == 0: #Train the policy every steps_per_epoch steps
                self.policy.train(self.replay_buffer, actor_scaler)


        ##将本次任务适配过程中收集的奖励、成功率数据存入lifetime_buffer
        lifetime_buffer.episodes_returns=lifetime_buffer.episodes_returns+episodes_returns_lifetime
        lifetime_buffer.episodes_successes =lifetime_buffer.episodes_successes+ episodes_successes_lifetime

        #需要把evaluation_error反向传播以后才能reset

        information = {
            'prev_done': done_lifetime,
            'current_lifetime_step': current_lifetime_step,
            'current_episode_return': episode_return_lifetime,
            'current_episode_success': succeeded_in_episode_lifetime,
            'current_state': current_state,  # Add current_state
            'current_episode_step_num': episode_step_num,  # Add current_episode_step_num
            'hidden_state': hidden_state  # Add hidden_state
        }
        
        return information
    
    def evaluate_critic(self, config:Metaworld):
        total_eval_steps = config.num_env_steps_for_estimating_maml_loss
        batch_size = self.args.batch_size

        state, _ = self.eval_env.reset() 
        step = 0
        episode_timesteps = 0
        episode_return = 0


        for cur_steps in range(total_eval_steps):
            step += 1
            #使用当前状态从策略网络获得该执行的动作和对数概率和状态价值
            action, logp, v_t = self.policy.get_action_and_val(state)

            # Step the env
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_timesteps += 1 #当前回合的时间步加1
            episode_return += reward #当前回合的总奖励加1

            done_flag = done

            #将这一回合的信息放在回放缓存区，更新状态
            self.replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=v_t, logp=logp)
            state = next_state

            if done or (episode_timesteps == config.max_episode_steps):
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if done else self.policy.get_val(state)
                self.replay_buffer.finish_path(last_val)
                state, _ = self.env.reset()
                episode_timesteps = 0
                episode_return = 0

        #copy_buffer = self.replay_buffer
        # add 开始更新extractor
        evaluate_every = 1
        extractor_losses=0
        for _ in range(evaluate_every):
            sample_states, sample_actions, sample_next_states, _, sample_dones = self.replay_buffer.sample(
                batch_size=batch_size)
            sample_next_actions, _ = self.policy.get_action(sample_next_states)

            # backward propagation to the loss

            ###  这似乎也缺一个函数
            pred_loss = self.policy.extractor_loss_for_adaptation(self.target_extractor, sample_states, sample_actions,
                                                                  sample_next_states, sample_next_actions, sample_dones)
            extractor_losses=extractor_losses+pred_loss


        # raw_states, actions, advantages, returns, logp_olds = copy_buffer.get()
        raw_states, actions, advantages, returns, logp_olds = self.replay_buffer.get()
        # add
        state_feature=self.policy.get_feature(raw_states)
        critic_loss = self.policy.critic_loss_for_adaptation(state_feature, returns)

        # critic_loss = self.policy.critic_loss_for_adaptation(raw_states, returns)



        print("世界和平！")

        #重置克隆网络
        self.policy.reset_cloned_networks()
        return critic_loss,extractor_losses
    