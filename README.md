# Project Outline

# Results
![Hopper agent trained with PPO](Hopper.gif)

# Useful links:

- [https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/] - blog post walking through PPO optimizations
- [https://www.youtube.com/watch?v=MEt6rrxH8W4] - Youtube video walking through the process of implementing PPO
- [https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml] RL Baselines (useful for tuned hyperparameters)
- [https://arxiv.org/abs/1707.06347] - Original PPO paper
- [https://arxiv.org/abs/1506.02438] - Paper on Generalized Advantage Estimation (GAE)

# Goal:

Create a highly optimized version of PPO that's able to train on gymnasium environments.

# Discrete PPO Notes:

- Vectorized architecture handles long horizon episodes by learning from fixed-length trajectory segments (look at iclr blog for psuedocode)
  - N is the number of parallel environments and M is the length of trajectory learning
- Break training data of size N x M into mini-batches to compute the gradient and update the policy
  - The size of mini batches is the batch size / the number of mini batches
  - Don't use the whole batch for the update
  - Don't randomly fetch training data as this does not guarantee that all training data points are fetched
- Normalize the advantages by subtracting their mean and dividing them by their standard deviation. This happens at the minibatch level instead of the whole batch level
- For Adam optimizer, epsilon is used to avoid division by 0 errors when updating parameters
- Store experience data in tensors
  - The observation and action data are dependent on the environment, and all of them are dependent on the number of environments/number of steps
- torch.view(-1) takes the elements of the original tensor and flattens it to a single row
- torch.zeros_like() gives a tensor of zeros with the same shape and data type as a given input tensor
- torch.reshape(1, -1) changes the tensor from its current state to a single row with the same elements. The -1 means Pytorch infers the number of columns based on the number of elements
- gym.wrappers.RecordEpisodeStatistics(env) returns episodic returns in the info variable when the episode finishes
- gym.wrappers.RecordVideo will record a video of the training at every certain number of episodes
- Batch size is the product of number of envs and the updates per batch
- Learning rate annealing by linearly decreasing the learning rate with each update
  - optimizer.param_groups[0]["lr"] updates the learning rate
- Use torch.no_grad() wherever possible to be more efficient
- Bootstrapping refers to the process of estimating the value of a state or action based on the estimates of the values of subsequent states or actions
- Penalizing the KL coefficient ensures that the updated policy is not too far from the original policy
  - The KL divergence is a measure of how far one probability distribution (like the action probability distribution from a policy) to another probability distribution
  - However, it empirically performs worse than the clipped surrogate objective according to the PPO paper

# Continuous PPO Notes:

- Normalizing the observation by subtracting its running mean and dividing by its variance greatly boosts model performance
  - Do this in gym with the VecNormalize wrapper
- On top of observation normalization, observations are also clipped
  - The VecNormalize wrapper also does this in gym
- Reward scaling can signifcantly affect the performance of the algorithm
  - The VecNormalize wrapper does this by applying a discount-based scaling scheme where the rewards are divided by the standard deviation of a rolling discounted sum of the rewards (without subtracting and re-adding the mean)
- Reward clipping is also implemented by VecNormalize, but it's not clear whether this affects performance

# PPO Pseudocode

1. Take in arguments from command line
2. Initialize seeding (random, np.random, and torch.manual_seed)
3. Initialize synchronous environments using gym.vector.SyncVectorEnv() - takes a function as the argument
4. Initialize actor and critic MLP networks (can be a single Agent class) and their optimizers
   1. For PPO, use orthogonal initialization for weights and constant initialization for biases (sqrt(2) and 0 respectively)
5. Initialize storage tensors (states, actions, log_probs, rewards, dones, values)
6. For update in number of updates (total timesteps / batch size)
   1. Collect trajectory data from the actors for updates per batch number of timesteps
   2. Compute advantage estimates
   3. Optimize the surrogate L for K epochs with minibatch size M (number of actors \* number of updates per batch)

# GAE Pseudocode

1. Compute delta_t at all timesteps using the value function
   1. The formula is as follows: -(value of current state) + reward from current state + discount factor \* value of next state
2. Compute the advantage at all timesteps using the advantage equation
   1. The formula is as follows: (gamma \* lamda)^(step number) \_ delta\_(step # + 1)
