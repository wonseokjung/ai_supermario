#import retro

import gym_super_mario_bros
import logging
import gym
import gym_super_mario_bros
import numpy as np
import sys
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import random

from stable_baselines.deepq.dqn import DQN
#policy
from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines.deepq.policies import CnnPolicy


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

'''
    action_omega = random.randint(1,21)

        print(action_omega)
        for i in range(action_omega):

            state, reward, done, info = env.step(action)
            print("step",step,"iteration",i, "action", action)
'''

_env = gym_super_mario_bros.make('SuperMarioBros-v0')
#_env = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=1, rom_mode='rectangle')
env = BinarySpaceToDiscreteSpaceEnv(_env, movements)
env = DummyVecEnv([lambda: env])
model = DQN(policy=CnnPolicy, env=env, learning_rate=1e-3, buffer_size = 5000,exploration_fraction=0.1, exploration_final_eps=0.1,param_noise=False)
model.learn(total_timesteps=10000)

obs = env.reset()

while True:
    action, _info = model.predict(obs)

    obs, rewards, dones, info = env.step(action)
    print("학습끝")
    print(rewards)
    env.render()
