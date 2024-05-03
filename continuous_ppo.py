import argparse
import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id",
        type=str,
        default="HalfCheetah-v4",
        help="the name of the gym environment being used",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="the seed used for the experiment"
    )
    parser.add_argument(
        "--cuda", type=bool, default=True, help="flag whether or not to use cuda"
    )
    parser.add_argument(
        "--record_video",
        type=bool,
        default=False,
        help="flag whether or not to record vidoes of training",
    )

    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor for rewards"
    )
    parser.add_argument(
        "--gae_lambda", type=float, default=0.95, help="the lambda value for GAE"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.2,
        help="the clipping coefficient described in the original PPO paper",
    )
    parser.add_argument(
        "--entropy_coeff",
        type=float,
        default=0.0,
        help="coefficient of the entropy in the loss equation",
    )
    parser.add_argument(
        "--value_coeff",
        type=float,
        default=0.5,
        help="coefficient of the value loss in the loss equation",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="the number of synchronous environments used by the agent for training",
    )
    parser.add_argument(
        "--updates_per_epoch",
        type=int,
        default=10,
        help="the number of policy updates to do per epoch (K in the original paper)",
    )
    parser.add_argument(
        "--normalize_advantage",
        type=bool,
        default=True,
        help="flag whether or not to normalize advantages",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=2000000,
        help="the total number of timesteps that the agent should train for",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="the learning rate for the optimizer",
    )
    parser.add_argument(
        "--steps_per_batch",
        type=int,
        default=2048,
        help="the number of timesteps for each sub-trajectory",
    )
    parser.add_argument(
        "--num_minibatches", type=int, default=32, help="number of minibatches"
    )

    return parser.parse_args()


# From the PPO optimization blog post github
def init_layer(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)

    return layer


class Agent(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super(Agent, self).__init__()

        self.critic = nn.Sequential(
            init_layer(nn.Linear(envs.single_observation_space.shape[0], 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 1)),
        )

        self.actor = nn.Sequential(
            init_layer(nn.Linear(envs.single_observation_space.shape[0], 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, envs.single_action_space.n)),
        )

    def get_action(self, state, action=None):
        logits = self.actor(state)
        probs = Categorical(logits=logits)

        if action == None:
            action = probs.sample()

        log_prob = probs.log_prob(action)

        return action, log_prob, probs.entropy()

    def get_value(self, state):
        return self.critic(state)


def create_env(env_id, seed, idx, record_video, run_name):
    def callback():
        env = gym.make(env_id, render_mode="rgb_array")
        if record_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, f"videos/{run_name}", episode_trigger=lambda x: x % 20 == 0
                )

        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda obs: np.clip(obs, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return callback


if __name__ == "__main__":
    args = get_args()
    date = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    run_name = f"{args.env_id}_{args.seed}_{date}"

    # Initialize seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initalize vectorized gym environments
    envs = gym.vector.SyncVectorEnv(
        [
            create_env(args.env_id, args.seed, i, args.record_video, run_name)
            for i in range(args.num_envs)
        ]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda is True else "cpu"
    )

    agent = Agent(envs).to(device)
    agent_optimizer = torch.optim.Adam(
        agent.parameters(), lr=args.learning_rate, eps=1e-5
    )

    mse_loss = nn.MSELoss()

    # Initialize storage tensors - for states and actions, each tensor has a dimension for each step.
    # The matrix in this dimension is number of envs x observation/action space
    # This means that each row in the printed tensor is the observation for one environment in one step
    # It's effectively creating a 3D matrix with the following dimensions: number of steps x number of envs x space of observation/action space
    states = torch.zeros(
        (args.steps_per_batch, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.steps_per_batch, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    log_probs = torch.zeros(args.steps_per_batch, args.num_envs).to(device)
    rewards = torch.zeros(args.steps_per_batch, args.num_envs).to(device)
    dones = torch.zeros(args.steps_per_batch, args.num_envs).to(device)
    values = torch.zeros(args.steps_per_batch, args.num_envs).to(device)

    # Initialize next_states and next_dones tensors for the initial environment step
    initial_states, _ = envs.reset()
    next_state = torch.Tensor(initial_states).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Calculate the batch size, minibatch size, and number of epochs based on the steps per batch, number of environments, and total timesteps
    batch_size = int(args.steps_per_batch * args.num_envs)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_updates = int(args.total_timesteps / batch_size)
    total_t = 0
    start_time = time.time()
    for update in range(num_updates):
        # Anneal the learning rate
        frac = 1.0 - (update - 1.0) / num_updates
        agent_optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for t in range(args.steps_per_batch):
            total_t += 1 * args.num_envs
            states[t] = next_state
            dones[t] = next_done

            with torch.no_grad():
                action, log_prob, _ = agent.get_action(next_state)
                value = agent.get_value(next_state)
                values[t] = value.flatten()

            # Store the action and its log probability
            actions[t] = action
            log_probs[t] = log_prob

            # Take a step in the environments based on the action returned from the actor network
            # The tensor has to be transferred back to the CPU since it's carrying out the environment interaction.
            # The tensor then has to be converted to a numpy array because that's what the gym environment takes
            state, reward, done, *info = envs.step(action.cpu().numpy())

            # Add the new reward to the rewards storage tensor
            rewards[t] = torch.tensor(reward).to(device)

            # Create the next_state and next_done tensor that'll be appended to their respective storage tensors during the next iteration
            next_state = torch.Tensor(state).to(device)
            next_done = torch.Tensor(done).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_state).reshape(1, -1)

            # Creates a tensor of zeros with the same shape as the rewards tensor
            advantages = torch.zeros_like(rewards).to(device)
            prev_advantage = 0

            for t in reversed(range(args.steps_per_batch)):
                # Deals with the first iteration when trying to calculate the mask
                if t == args.steps_per_batch - 1:
                    mask = 1 - next_done
                else:
                    mask = 1 - dones[t + 1]
                    next_value = values[t + 1]

                # Equation for temporal difference (TD) - ignores the next value term if it's a terminal state
                delta_t = -values[t] + rewards[t] + args.gamma * next_value * mask

                # Equation for GAE
                prev_advantage = delta_t + args.gamma * args.gae_lambda * prev_advantage
                advantages[t] = prev_advantage

                # Calculate returns (used for value function loss)
                returns = advantages + values

        # Flatten storage tensors so that they can be sliced into minibatches
        batch_states = states.reshape((-1,) + envs.single_observation_space.shape)
        batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        batch_log_probs = log_probs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_values = values.reshape(-1)
        batch_returns = returns.reshape(-1)

        # Create list of indexes that are randomized for minibatch slicing
        batch_idxs = np.arange(batch_size)

        for epoch in range(args.updates_per_epoch):
            # Randomize the batch into minibatch updates
            np.random.shuffle(batch_idxs)

            for start_idx in range(0, batch_size, minibatch_size):
                end_idx = start_idx + minibatch_size

                minibatch_idxs = batch_idxs[start_idx:end_idx]

                _, new_log_probs, entropy = agent.get_action(
                    batch_states[minibatch_idxs], batch_actions[minibatch_idxs]
                )
                new_value = agent.get_value(batch_states[minibatch_idxs]).view(-1)

                minibatch_advantages = batch_advantages[minibatch_idxs].to(device)

                if args.normalize_advantage:
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std() + 1e-8)

                ratio = torch.exp(new_log_probs - batch_log_probs[minibatch_idxs])
                surrogate_obj_1 = ratio * minibatch_advantages
                surrogate_obj_2 = (
                    torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                    * minibatch_advantages
                )

                policy_loss = torch.min(surrogate_obj_1, surrogate_obj_2).mean()
                value_loss = mse_loss(new_value, batch_returns[minibatch_idxs])
                entropy_loss = entropy.mean()

                # Calculate the loss value. The paper maximizes the objective function, but Adam uses gradient descent,
                # so need to take the negative of the objective function in the paper.
                loss = (
                    -policy_loss
                    + args.value_coeff * value_loss
                    - args.entropy_coeff * entropy_loss
                )

                agent_optimizer.zero_grad()
                loss.backward()
                agent_optimizer.step()

        print(f"Total timesteps: {total_t}")
        print(f"Loss: {loss}")
        print(f"Total training time: {round(time.time() - start_time, 2)}")
        print("")

    envs.close()
