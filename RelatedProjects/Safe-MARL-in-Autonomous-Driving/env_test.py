import gym
import time
import numpy as np
import highway_env
highway_env.register_highway_envs()

env = gym.make("u-turn-v0")

for episode in range(10):  # watch 10 episodes
    obs = env.reset()
    done = False
    print(f"Episode {episode + 1}")
    while not done:
        act = env.action_space.sample()
        result = env.step(act)
        obs, reward, done = result[0], result[1], result[2]
        env.render()
        time.sleep(0.05)

env.close()
