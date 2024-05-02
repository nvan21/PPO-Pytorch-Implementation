print(env.observation_space.shape)
obs = torch.zeros((4, 3))
print(obs)

obs = torch.zeros((4, 3) + (3,))
print(obs)
