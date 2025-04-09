import gymnasium as gym
import numpy as np
import numpy.typing
import torch
import torch.optim as optim
import torchopt
import metaworld
import time
import random
import wandb
import ray
import os


from blocks.maml_config import get_config, Metaworld
from maml_agent import MAML_agent
from blocks.data_collection import collect_data_from_env
from blocks.TRPO_and_adapt_loss import maml_actor_trpo_update
from blocks.data_buffers import Lifetime_buffer
from blocks.general_utils import Logger, Statistics_tracker, Sampler, Tasks_batch_sampler
from arguments import parse_args
from Add.misc import set_gpu_device_growth

set_gpu_device_growth()  #可以让gpu逐步增加内存，避免从一开始就占用了大量的内存
torch.autograd.set_detect_anomaly(True)

config_setting = 'metaworld'
config = get_config(config_setting)
args = parse_args()

benchmark_name = config.benchmark_name
env_name = config.env_name

steps_per_episode = 500  # number of steps in a metaworld episode - just used for logging purposes

############ SETUP #############

# Construct the benchmark and construct an iterator that returns batches of tasks. (and sample a random env for setting up some configurations)
if benchmark_name == 'ML1':
    benchmark = metaworld.ML1(f'{env_name}', seed=config.seed)  # 设置基准环境 设置环境的随机种子，确保实验的可重复性
    exp_name = f'{benchmark_name}_{env_name}'
    task_sampler = Sampler(benchmark.train_tasks, config.num_inner_loops_per_update)  # 任务采样,确定内循环的任务数量
    example_env = benchmark.train_classes[f'{env_name}']()  # 实例化环境

    def create_env(task):
        env = benchmark.train_classes[f'{task.env_name}']()
        env.set_task(task)  # 将任务传到环境中
        env = gym.wrappers.ClipAction(env)  # 用于确保动作空间的输出值不会超出定义的动作范围
        if config.seeding == True:  # 设置环境的随机种子，可以确保实验的随机性（如动作选择，环境状态变化等）
            env.action_space.seed(config.seed)
            env.observation_space.seed(config.seed)
        return env

elif benchmark_name == 'ML10':
    benchmark = metaworld.ML10(seed=config.seed)
    exp_name = f'{benchmark_name}'
    task_sampler = Tasks_batch_sampler(benchmark, config.num_inner_loops_per_update)
    example_env = next(iter(benchmark.train_classes.items()))[1]()
elif benchmark_name == 'ML45':
    benchmark = metaworld.ML45(seed=config.seed)
    task_sampler = Tasks_batch_sampler(benchmark, config.num_inner_loops_per_update)
    exp_name = f'{benchmark_name}'
    example_env = next(iter(benchmark.train_classes.items()))[1]()

if benchmark_name == 'ML10' or benchmark_name == 'ML45':
    def create_env(task):
        benchmark_envs = benchmark.train_classes.copy()
        benchmark_envs.update(benchmark.test_classes)
        env = benchmark_envs[f'{task.env_name}']()
        env.set_task(task)
        env = gym.wrappers.ClipAction(env)
        if config.seeding == True:
            env.action_space.seed(config.seed)
            env.observation_space.seed(config.seed)
        return env

# wandb connection
model_id = int(time.time())
run_name = f"MAML_{exp_name}__{model_id}"
wandb.init(project='project_name',
           name=run_name,
           config=vars(config))

if config.seeding == True:
    random.seed(config.seed)  # 设置 Python 标准库 random 的随机种子，确保 Python 中的随机操作（如随机选择元素）可重复。
    np.random.seed(config.seed)  # 设置 NumPy 中的随机种子，确保 NumPy 生成的随机数序列在每次运行时保持一致。
    torch.manual_seed(config.seed)  # 设置 PyTorch 的随机种子，确保 PyTorch 中生成的随机数（如初始化网络权重、打乱数据等）也可以复现。
if config.device == 'auto':  # 设置计算设备是cpu或者gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = config.device

print(f'MAML: Device is set to :{device}')

print("the name is : ", benchmark_name)

print("the name of training task is : ", example_env)
maml_agent = MAML_agent(example_env, agent_config=args, maml_config=config).to(device)  # 初始化agent

# 初始化优化器
maml_actor_optimizer = optim.Adam(maml_agent.actor.parameters(), lr=config.maml_agent_lr, eps=config.maml_agent_epsilon)
maml_critic_optimizer = optim.Adam(maml_agent.policy.critic_original.parameters(), lr=config.maml_critic_lr)

maml_extractor_optimizer = optim.Adam(maml_agent.extractor.parameters(), lr=config.maml_extractor_lr)#add

# 初始化数据统计跟踪器和日志记录器
data_statistics = Statistics_tracker()
logger = Logger(num_epsiodes_of_validation=config.num_epsiodes_of_validation)

# 定义模型的保存路径和性能跟踪
best_maml_agent_model_path = f"../maml_models/{run_name}__best_model.pth"
best_model_performance = 0

# for determining best model version
def validation_performance(logger):
    # 计算模型在指定回合内获得平均成功率
    performance = np.array(logger.validation_episodes_success_percentage[-config.num_lifetimes_for_validation:]).mean()

    # 处理 validation_episodes_return，确保它是一个 Tensor 类型的列表，并转换为 Numpy 数组
    validation_returns = logger.validation_episodes_return[-config.num_lifetimes_for_validation:]
    # 如果 validation_episodes_return 列表中的元素是 Tensor 类型，转换为 Numpy 数组
    if isinstance(validation_returns[0], torch.Tensor):
        validation_returns = [tensor.cpu().numpy() for tensor in validation_returns]
    else:
        # 如果它们已经是 Numpy 数组或其他类型，直接使用
        validation_returns = np.array(validation_returns)

    # 计算回报的均值并加上 1e-6
    performance = performance + 1e-6 * np.mean(validation_returns)+config.reward

    # for resolving ties
    return performance


############ DEFINE BEHAVIOUR OF INNER LOOPS #############

def inner_loop(base_policy_state_dict, task, config: Metaworld, data_statistics):
    import sys
    sys.path.append('../')

    env = create_env(task)

    adaptation_mean_reward_for_baseline = data_statistics.rewards_mean

    # 初始化agent
    maml_agent = MAML_agent(env, agent_config=args, maml_config=config).to(device)
    # 将原始策略加载到agent中
    torchopt.recover_state_dict(maml_agent.actor, base_policy_state_dict)
    ##这里可能要补点东西

    # 初始化生命周期缓冲区，用于存储从环境中收集到的数据（例如，状态、动作、奖励等）
    lifetime_buffer = Lifetime_buffer(config.num_lifetime_steps, config.num_env_steps_per_adaptation_update, env, device, env_name=f'{task.env_name}')

    # ADAPTATION - K gradient steps
    # for each adaptation step : collect data with current policy (starting with the base policy) for doing the inner loop adaptation. Use this data to
    # Compute gradient of the tasks's policy objective function  with respect to the current policy parameters.
    # finally use this gradient to update the parameters. Repeat the process config.num_adaptation_steps times .


    information = {'current_state': torch.tensor(env.reset()[0], dtype=torch.float32).to(device),
                   'prev_done': torch.ones(1).to(device),
                   'current_episode_step_num': 0, 'current_episode_success': False, 'current_episode_return': 0,
                   'current_lifetime_step': 0, 'hidden_state': None}

    information = maml_agent.adapt(num_steps=config.num_env_steps_per_adaptation_update, information=information,
                                   config=config, lifetime_buffer=lifetime_buffer, mean_reward_for_baseline=adaptation_mean_reward_for_baseline, device=device)

    # META LOSS CALCULATION
    # collect data with adapted policy for the outer loop update (for computing the loss that is used for the maml meta update - updating the base policy) .

    # information2 = {'current_state': torch.tensor(env.reset()[0], dtype=torch.float32).to(device),
    #                'prev_done': torch.ones(1).to(device),
    #                'current_episode_step_num': 0, 'current_episode_success': False, 'current_episode_return': 0,
    #                'current_lifetime_step': 0, 'hidden_state': None}
    critic_losses,extractor_losses=maml_agent.evaluate_critic(config=config)  # back propagate pred_loss with respect to the extractor
    maml_buffer, _ = collect_data_from_env(agent=maml_agent, env=env, num_steps=config.num_env_steps_for_estimating_maml_loss, information=information,
                                           config=config, lifetime_buffer=lifetime_buffer, env_name=f'{task.env_name}',
                                           for_maml_loss=True, device=device)

    # 从适应后的 maml_agent 中提取更新后的策略参数，以便在外部更新时使用。
    adapted_policy_state_dict = torchopt.extract_state_dict(maml_agent.actor)

    return lifetime_buffer, maml_buffer, adapted_policy_state_dict,critic_losses,extractor_losses

############ OUTER LOOP #############
if ray.is_initialized:
    ray.shutdown()
ray.init()#确保可以分布式执行任务

remote_inner_loop=ray.remote(inner_loop)#inner_loop 函数包装成一个可以远程执行的 Ray 任务
config_ref=ray.put(config)

for i in range(config.num_outer_loop_updates):

    # 提取当前策略的状态字典
    policy_state_dict = torchopt.extract_state_dict(maml_agent.actor)

    ##--------------COLLECTING DATA --------------
    # 使用 Ray 的并行能力
    # 将当前智能体的参数和数据统计对象存入 Ray 的共享内存
    policy_state_dict_ref = ray.put(policy_state_dict)
    data_statistics_ref = ray.put(data_statistics)

    # 为每个任务创建输入数据列表并并行化执行
    inputs = [[policy_state_dict_ref, task, config_ref, data_statistics_ref] for task in next(task_sampler)]
    # 使用 Ray 并行执行多个任务，并收集返回的数据
    lifetime_buffers, maml_buffers, adapted_policy_state_dicts,critic_losses,extractor_losses = zip(*ray.get([remote_inner_loop.options(num_gpus=1).remote(*i) for i in inputs]))

    # take maml_agent back to its original state
    torchopt.recover_state_dict(maml_agent.actor, policy_state_dict)

    ##--------------PROCESSING STAGE --------------
    # 处理收集的数据并计算统计信息
    for lifetime_data in lifetime_buffers:
        data_statistics.update_statistics(lifetime_data)
        logger.collect_per_lifetime_metrics(lifetime_data,
                                            episodes_till_first_adaptation_update=config.num_env_steps_per_adaptation_update // steps_per_episode,
                                            episodes_after_adaptation=config.num_env_steps_for_estimating_maml_loss // steps_per_episode)

    # 预处理数据并计算回报与优势
    for maml_buffer in maml_buffers:
        maml_buffer.preprocess_data(data_stats=data_statistics, objective_mean=config.rewards_target_mean_for_maml_agent)
        mean_reward_for_control_variate = config.rewards_target_mean_for_maml_agent

        maml_buffer.calculate_returns_and_advantages(mean_reward=config.rewards_target_mean_for_maml_agent, gamma=config.maml_agent_gamma)

    ##critic进行元更新
    total_loss1=0
    for loss1 in critic_losses:
        total_loss1=total_loss1+loss1
    total_loss1=total_loss1/config.num_inner_loops_per_update
    maml_critic_optimizer.zero_grad()
    total_loss1.backward()
    maml_critic_optimizer.step()

    ##extractor进行元更新
    total_loss2 = 0
    for loss1 in extractor_losses:
        total_loss2 = total_loss2 + loss1
    total_loss2 = total_loss2 / config.num_inner_loops_per_update
    maml_extractor_optimizer.zero_grad()
    total_loss2.backward()
    maml_extractor_optimizer.step()

    # ---log and print metrics
    logger.log_per_update_metrics(num_inner_loops_per_update=config.num_inner_loops_per_update)
    print(f'completed meta update {i} , base policy return and success percentage={np.array(logger.base_maml_agent_return[-config.num_inner_loops_per_update:]).mean()} ,{np.array(logger.base_maml_agent_success_percentage[-config.num_inner_loops_per_update:]).mean()} , adapted policy return and success percentage={np.array(logger.adapted_maml_agent_return[-config.num_inner_loops_per_update:]).mean()} ,{np.array(logger.adapted_maml_agent_success_percentage[-config.num_inner_loops_per_update:]).mean()}')

    # -------Save model if required----
    model_performance = validation_performance(logger)
    if model_performance > best_model_performance:
        best_model_performance = model_performance
        print(f'best model performance= {best_model_performance}')
        statistics = {'rewards_mean': data_statistics.rewards_mean}

        data_to_save = {
            'maml_model_state_dict': maml_agent.state_dict(),
            'statistics': statistics
        }

        # Define the path where the model will be saved
        best_maml_agent_model_path = "path/to/save/best_model.pth"  # 替换为你实际的保存路径

        # Check if the directory exists, if not create it
        save_dir = os.path.dirname(best_maml_agent_model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create the directory if it doesn't exist

        # Save the model
        torch.save(data_to_save, best_maml_agent_model_path)

    # --------------  UPDATE MODEL ------------------------
    ###maml_actor_trpo_update(maml_agent=maml_agent, data_buffers=maml_buffers, adapted_policies_states=adapted_policy_state_dicts, config=config, logging=True)
wandb.finish()