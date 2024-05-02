import argparse
import random

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
        default="CartPole-v1",
        help="the name of the gym environment being used",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="the seed used for the experiment"
    )
    parser.add_argument(
        "--cuda", type=bool, default=True, help="flag whether or not to use cuda"
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="the number of synchronous environments used by the agent for training",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100000,
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
        default=128,
        help="the number of timesteps for each sub-trajectory",
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

    def get_action(self, state):
        logits = self.actor(state)
        probs = Categorical(logits=logits)

        action = probs.sample()
        log_prob = probs.log_prob(action)

        return action, log_prob

    def get_value(self, state):
        return self.critic(state)


def create_env(env_id, seed):
    def callback():
        env = gym.make(env_id)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return callback


if __name__ == "__main__":
    args = get_args()

    # Initialize seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initalize vectorized gym environments
    envs = gym.vector.SyncVectorEnv(
        [create_env(args.env_id, args.seed) for i in range(args.num_envs)]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda is True else "cpu"
    )

    agent = Agent(envs).to(device)
    agent_optimizer = torch.optim.Adam(
        agent.parameters(), lr=args.learning_rate, eps=1e-5
    )

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

    # Calculate the batch size and number of epochs based on the steps per batch, number of environments, and total timesteps
    batch_size = int(args.steps_per_batch * args.num_envs)
    num_epochs = int(args.total_timesteps / batch_size)
    for update in range(num_epochs):

        for t in range(args.steps_per_batch):
            states[t] = next_state
            dones[t] = next_done

            with torch.no_grad():
                action, log_prob = agent.get_action(next_state)
                value = agent.get_value(next_state)
                values[t] = value.flatten()

            # Take a step in the environments based on the action returned from the actor network
            # The tensor has to be transferred back to the CPU since it's carrying out the environment interaction.
            # The tensor then has to be converted to a numpy array because that's what the gym environment takes
            state, reward, done, *info = envs.step(action.cpu().numpy())

            # Add the new reward to the rewards storage tensor
            rewards[t] = torch.tensor(reward).to(device)

            # Create the next_state and next_done tensor that'll be appended to their respective storage tensors during the next iteration
            next_state = torch.Tensor(state).to(device)
            next_done = torch.Tensor(done).to(device)
