import torch
import gymnasium as gym

env = gym.make("Pendulum-v1")

print(env.observation_space.shape)
obs = torch.zeros((4, 3))
print(obs)

obs = torch.zeros((4, 3) + (5,))
print(obs)
