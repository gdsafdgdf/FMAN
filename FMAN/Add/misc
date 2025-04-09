import torch
import torch.nn.functional as F
import numpy as np


# 激活函数：Swish 和 Tanh
class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Tanh(torch.nn.Module):
    def forward(self, x):
        return torch.tanh(x)


# 控制 GPU 内存增长
def set_gpu_device_growth():
    if torch.cuda.is_available():
        # 设置 GPU 内存按需增长
        torch.cuda.set_per_process_memory_fraction(1.0, 0)  # 让 PyTorch 在每个 CUDA 设备上动态增长内存
        print('Available GPU:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Physical GPU {i}: {torch.cuda.get_device_name(i)}")
        print("Memory Growth set for GPUs.")
    else:
        print("CUDA is not available. Using CPU.")


# 获取目标维度
def get_target_dim(env_name):
    TARGET_DIM_DICT = {
        "Ant-v2": 27,
        "HalfCheetah-v2": 17,
        "Walker2d-v2": 17,
        "Hopper-v2": 11,
        "Reacher-v2": 11,
        "Humanoid-v2": 292,
        "Swimmer-v2": 8,
        "InvertedDoublePendulum-v2": 11
    }

    return TARGET_DIM_DICT.get(env_name, 0)  # 默认返回0


# 获取默认训练步骤
def get_default_steps(env_name):
    if env_name.startswith('HalfCheetah'):
        return 3000000
    elif env_name.startswith('Hopper'):
        return 1000000
    elif env_name.startswith('Walker2d'):
        return 5000000
    elif env_name.startswith('Ant'):
        return 5000000
    elif env_name.startswith('Swimmer'):
        return 3000000
    elif env_name.startswith('Humanoid'):
        return 3000000
    elif env_name.startswith('InvertedDoublePendulum'):
        return 1000000
    else:
        return 1000000  # 默认值


# 生成特征提取器名称
def make_ofe_name(ofe_layer, ofe_unit, ofe_act, ofe_block):
    exp_name = f"L{ofe_layer}_U{ofe_unit}_{ofe_act}_{ofe_block}"
    return exp_name


# 经验归一化类
class EmpiricalNormalization:
    def __init__(self, shape, batch_axis=0, eps=1e-2, dtype=torch.float32,
                 until=None, clip_threshold=None):
        self.batch_axis = batch_axis
        self.eps = dtype(eps)
        self.until = until
        self.clip_threshold = clip_threshold
        self._mean = torch.zeros(shape, dtype=dtype)
        self._var = torch.ones(shape, dtype=dtype)
        self.count = 0
        self._cached_std_inverse = None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return torch.sqrt(self._var)

    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = 1.0 / torch.sqrt(self._var + self.eps)
        return self._cached_std_inverse

    def experience(self, x):
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        self.count += count_x
        rate = x.dtype.type(count_x / self.count)

        mean_x = x.mean(dim=self.batch_axis, keepdim=True)
        var_x = x.var(dim=self.batch_axis, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))

        # clear cache
        self._cached_std_inverse = None

    def __call__(self, x, update=True):
        mean = self._mean.expand(x.shape)
        std_inv = self._std_inverse.expand(x.shape)

        if update:
            self.experience(x)

        normalized = (x - mean) * std_inv
        if self.clip_threshold is not None:
            normalized = torch.clamp(normalized, -self.clip_threshold, self.clip_threshold)
        return normalized

    def inverse(self, y):
        mean = self._mean.expand(y.shape)
        std = torch.sqrt(self._var + self.eps).expand(y.shape)
        return y * std + mean
