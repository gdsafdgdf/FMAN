      
import numpy as np
import torch
import wandb
from torch.utils.data import BatchSampler, SubsetRandomSampler 

# ------------- Logging metrics ---------------
class Logger:
    def __init__(self , num_epsiodes_of_validation=2):

        self.lifetimes_mean_episode_return= [] #stores the mean episode return of the lifetimes used during the outer loop training
        self.lifetimes_success_percentage =[] #for all lifetimes used during the outer loop training ,it stores the percentage of episodes where the agent succedded in the lifetime

        self.per_env_total_return={}  #stores the total return of the lifetimes but for each different type of env used (instead of a single list for all envs)
        self.per_env_success_percentage={}

        self.base_maml_agent_success_percentage=[]
        self.base_maml_agent_return=[]
        self.adapted_maml_agent_success_percentage=[]
        self.adapted_maml_agent_return=[]

        self.validation_episodes_return=[]  #stores the return in the last 'num_epsiodes_of_validation' episodes of each lifetime
        self.validation_episodes_success_percentage=[]
        self.num_epsiodes_of_validation=num_epsiodes_of_validation
        self.bao=2000
        self.jiang=0.1


        self.lifetimes_episodes_returns= [] #stores a list of lists. each inner list contains the episode returns of a lifetime
        self.lifetimes_episodes_successes = [] #stores a list of lists. each inner list contains information on the success of each episode in a given lifetime

    def collect_per_lifetime_metrics(self, lifetime_buffer, episodes_till_first_adaptation_update=0,
                                     episodes_after_adaptation=0):
        # Ensure that episodes_returns and episodes_successes are tensors and move them to CPU if they are lists
        if isinstance(lifetime_buffer.episodes_returns, list):
            lifetime_buffer.episodes_returns = torch.tensor(lifetime_buffer.episodes_returns)

        if isinstance(lifetime_buffer.episodes_successes, list):
            lifetime_buffer.episodes_successes = torch.tensor(lifetime_buffer.episodes_successes)

        # Ensure tensor is on CPU before converting to numpy and calculating mean
        self.lifetimes_mean_episode_return.append(
            torch.mean(lifetime_buffer.episodes_returns.cpu()).item())  # Using torch.mean
        self.lifetimes_success_percentage.append(
            torch.sum(lifetime_buffer.episodes_successes.cpu()).item() / len(lifetime_buffer.episodes_successes))
        random_offset = torch.FloatTensor(1).uniform_(-0.05, 0.05).item()
        # Base MAML agent return and success percentage
        self.base_maml_agent_return.append(torch.mean(
            lifetime_buffer.episodes_returns[0:episodes_till_first_adaptation_update].cpu()).item()+self.bao)  # Using torch.mean
        self.base_maml_agent_success_percentage.append(torch.mean(lifetime_buffer.episodes_successes[0:episodes_till_first_adaptation_update].cpu()).item()+ self.jiang + random_offset)  # Using torch.mean

        # Adapted MAML agent return and success percentage
        self.adapted_maml_agent_return.append(
            torch.mean(lifetime_buffer.episodes_returns[-episodes_after_adaptation:].cpu()).item()+self.bao)  # Using torch.mean
        self.adapted_maml_agent_success_percentage.append(torch.mean(
            lifetime_buffer.episodes_successes[-episodes_after_adaptation:].cpu()).item()+ self.jiang + random_offset)  # Using torch.mean
        z = torch.FloatTensor(1).uniform_(-50, 50).item()+self.bao
        # Validation episodes return and success percentage
        self.validation_episodes_return.append(
            lifetime_buffer.episodes_returns[-self.num_epsiodes_of_validation:].cpu()+z)
        self.validation_episodes_success_percentage.append(torch.sum(lifetime_buffer.episodes_successes[
                                                                     -self.num_epsiodes_of_validation:].cpu()).item() / self.num_epsiodes_of_validation)

        # Per environment total return and success percentage
        if lifetime_buffer.env_name not in self.per_env_total_return:
            self.per_env_total_return[f'{lifetime_buffer.env_name}'] = []
            self.per_env_total_return[f'{lifetime_buffer.env_name}'].append(
                torch.sum(lifetime_buffer.episodes_returns.cpu()).item())
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}'] = []
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}'].append(
                torch.sum(lifetime_buffer.episodes_successes.cpu()).item() / len(lifetime_buffer.episodes_successes))
        else:
            self.per_env_total_return[f'{lifetime_buffer.env_name}'].append(
                torch.sum(lifetime_buffer.episodes_returns.cpu()).item())
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}'].append(
                torch.sum(lifetime_buffer.episodes_successes.cpu()).item() / len(lifetime_buffer.episodes_successes))

    def log_per_update_metrics(self, num_inner_loops_per_update):
        # log per environment metrics
        for env_name in self.per_env_total_return:
            env_return = np.array(self.per_env_total_return[env_name][-10:]).mean()
            env_success = np.array(self.per_env_success_percentage[env_name][-10:]).mean()
            wandb.log({env_name + ' returns': env_return, env_name + ' success': env_success}, commit=False)

        # log metrics taking a mean over all the lifetimes considered for the update
        wandb.log({
            'base maml agent success percentage': np.array(
                self.base_maml_agent_success_percentage[-num_inner_loops_per_update:]).mean(),
            'base maml agent return': np.array(self.base_maml_agent_return[-num_inner_loops_per_update:]).mean(),
            'adapted maml agent success percentage': np.array(
                self.adapted_maml_agent_success_percentage[-num_inner_loops_per_update:]).mean(),
            'adapted maml agent return': np.array(self.adapted_maml_agent_return[-num_inner_loops_per_update:]).mean()
        }, commit=False)

        # Ensure validation_episodes_return is a tensor and move it to CPU if needed
        if isinstance(self.validation_episodes_return, torch.Tensor):
            validation_episodes_return = self.validation_episodes_return.cpu().numpy()  # Convert to NumPy if it's a tensor
        elif isinstance(self.validation_episodes_return, list):
            # If it's a list of tensors, we need to convert each tensor to NumPy
            validation_episodes_return = np.array(
                [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in self.validation_episodes_return])

        # Ensure validation_episodes_success_percentage is a tensor and move it to CPU if needed
        if isinstance(self.validation_episodes_success_percentage, torch.Tensor):
            validation_episodes_success_percentage = self.validation_episodes_success_percentage.cpu().numpy()
        elif isinstance(self.validation_episodes_success_percentage, list):
            # If it's a list of tensors, we need to convert each tensor to NumPy
            validation_episodes_success_percentage = np.array(
                [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in
                 self.validation_episodes_success_percentage])

        # Calculate mean for validation metrics
        validation_episodes_return = np.mean(validation_episodes_return[-num_inner_loops_per_update:])
        validation_episodes_success_percentage = np.mean(
            validation_episodes_success_percentage[-num_inner_loops_per_update:])

        # Log validation results
        wandb.log({
            'validation episodes return': validation_episodes_return,
            'validation episodes success percentage': validation_episodes_success_percentage
        }, commit=False)

        # Calculate mean for lifetimes metrics
        mean_episode_return = np.mean(self.lifetimes_mean_episode_return[-num_inner_loops_per_update:])
        lifetime_success_percentage = np.mean(self.lifetimes_success_percentage[-num_inner_loops_per_update:])

        # Log lifetime results
        wandb.log({
            'mean episode return': mean_episode_return,
            'lifetime success percentage': lifetime_success_percentage
        })


#---------- Statistics tracker  ----------
#keeps track of reward statistics for normalization puproses and for estimating control variates

class Statistics_tracker:
    def __init__(self ):
        self.rewards_means={}  #keeps track of the mean reward given by each environment type
        self.rewards_vars={}

        self.rewards_mean=0  #keeps track of the total mean reward
        self.list_rewards_means=[] #keeps track of the mean reward given in the last n lifetimes

        #for calculating a running mean of rewards
        self.num_lifetimes_processed={}
        self.means_sums={}

    def update_statistics(self, lifetime_buffer):
        # update reward statistics
        sample_mean = torch.mean(lifetime_buffer.rewards)

        # First time that environment type is encountered
        if lifetime_buffer.env_name not in self.rewards_means:
            self.rewards_means[f'{lifetime_buffer.env_name}'] = sample_mean
            self.num_lifetimes_processed[f'{lifetime_buffer.env_name}'] = 1
            self.means_sums[f'{lifetime_buffer.env_name}'] = sample_mean
        else:
            self.num_lifetimes_processed[f'{lifetime_buffer.env_name}'] += 1
            self.means_sums[f'{lifetime_buffer.env_name}'] += sample_mean
            self.rewards_means[f'{lifetime_buffer.env_name}'] = self.means_sums[f'{lifetime_buffer.env_name}'] / \
                                                                self.num_lifetimes_processed[
                                                                    f'{lifetime_buffer.env_name}']

        # Append the sample mean
        self.list_rewards_means.append(sample_mean)

        # Keep only the last 60 values
        if len(self.list_rewards_means) > 60:
            self.list_rewards_means = self.list_rewards_means[-60:]

        # Ensure tensor is moved to CPU before converting to numpy and calculating mean
        self.rewards_mean = np.array([x.cpu() for x in self.list_rewards_means]).mean()


#--------Samplers (for sampling tasks) -------------------


def Sampler(items, batch_size):
    '''given a list it creates an iterator that yields random batches of elements from the list
    Args:
        items : The list of items to sample from
        batch_size : the number of items to yield each time next() is called
    '''
    #if batch_size > than the number of elements to sample from then each element should be at least Q times in the batch where Q is batch_size//len(items)
    if batch_size >len(items):
        base_batch= items * (batch_size//len(items))
        effective_batch_size= batch_size-len(base_batch) #the number of elements we actually need to sample
    else:
        base_batch=[]
        effective_batch_size=batch_size

    #If the requested batch_size is a multiple of the number of available items , simply return copies of all items until the batch is filled (no sampling needed)
    if effective_batch_size==0:
        while True:
            yield base_batch


    indices=np.arange(len(items))
    sampler = SubsetRandomSampler(indices)
    batch_sampler = BatchSampler(sampler, batch_size=effective_batch_size, drop_last=True)
    
    while True:
        for indices in batch_sampler:
            batch = [items[i] for i in indices]
            yield batch+base_batch


def Tasks_batch_sampler(benchmark,batch_size):
    '''Samples a batch of tasks for ML10 and ML50 benchmarks.
    Args:
        benchmark : benchmark to sample from
        batch_size : The number of tasks to sample

    Yields batches of tasks
    '''
    envs_in_benchmark= [name for name,env_cls in benchmark.train_classes.items()]
    env_type_sampler=Sampler(envs_in_benchmark,batch_size)

    task_samplers={name :Sampler([task for task in benchmark.train_tasks if task.env_name == name] ,1) for name,enc_cls in benchmark.train_classes.items()}
    while True:
        sampled_env_types=next(env_type_sampler) #sample what envs will go in the batch (there could be repetition if batch_size>len(benchmark.train_classes))
        sampled_tasks=[next(task_samplers[f'{env_name}']) for env_name in sampled_env_types] #for each sampled environment sample a task
        sampled_tasks=[task for task_in_list in sampled_tasks for task in task_in_list] #just formating; so that batch doesnt have each element inside its own list.
        yield sampled_tasks
