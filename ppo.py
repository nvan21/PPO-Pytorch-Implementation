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
