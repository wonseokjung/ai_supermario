#import retro

import gym_super_mario_bros
import logging
import gym
import gym_super_mario_bros
import numpy as np
import sys
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import random
from stable_baselines.ppo2.ppo2 import PPO2

from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv

movements = [
    ['NOP'],
    ['A'],
    ['B'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
#    ['down'],
#    ['up']
]


_env = gym_super_mario_bros.make('SuperMarioBros-v0')
#_env = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=1, rom_mode='rectangle')
env = BinarySpaceToDiscreteSpaceEnv(_env, movements)
env = DummyVecEnv([lambda: env])
model = PPO2(policy=CnnPolicy, env=env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()

while True:
    action, _info = model.predict(obs)

    obs, rewards, dones, info = env.step(action)
    print("학습끝")
    print(rewards)
    env.render()
