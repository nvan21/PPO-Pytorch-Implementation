import torch
import gymnasium as gym

env = gym.make("Pendulum-v1")

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(x.reshape(1, -1))
print(x.view())
