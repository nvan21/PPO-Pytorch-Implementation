import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


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
        "--num_epochs",
        type=int,
        default=50,
        help="number of epochs the training should go through (the number of batches to collect)",
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

    return parser.parse_args()


# From the PPO optimization blog post github
def init_layer(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)


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
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    log_probs = torch.zeros(args.num_steps, args.num_envs).to(device)
    rewards = torch.zeros(args.num_steps, args.num_envs).to(device)
    dones = torch.zeros(args.num_steps, args.num_envs).to(device)
    values = torch.zeros(args.num_steps, args.num_envs).to(device)

    for update in range(args.num_epochs):
        pass
