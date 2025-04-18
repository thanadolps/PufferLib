from pdb import set_trace as T

import gymnasium
import functools
from gymnasium.spaces import Box
import shimmy
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import pufferlib.emulation
import pufferlib.postprocess
import pufferlib.wrappers

def env_creator(name='SuperMarioBros-1-1-v2'):
    return functools.partial(make, name)

def make(name, buf=None, seed=None, render_mode='rgb_array'):
    '''Super Mario Bros'''
    env = gym_super_mario_bros.make(name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # env = pufferlib.wrappers.GymToGymnasium(env)
    env = shimmy.GymV21CompatibilityV0(env=env, render_mode=render_mode)

    # env = SkipWrapper(env, 4)
    env = gymnasium.wrappers.GrayScaleObservation(env)

    
    env = pufferlib.postprocess.ResizeObservation(env)
    env = ExpandDimObservation(env)

    # env = RenderObservation(env) # for debugging
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


class ExpandDimObservation(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(0, 255, (1, env.observation_space.shape[0], env.observation_space.shape[1]), env.observation_space.dtype)

    def observation(self, observation):
        return observation[None, :, :]

class RenderObservation(gymnasium.ObservationWrapper):
    """
    Override render to return the direct observation of the environment, useful for debugging
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_obs = None

    def observation(self, observation):
        self.last_obs = observation
        return observation
    
    def render(self):
        s = self.last_obs.shape
        if len(s) == 2 or len(s) == 3 and s[0] == 3:
            return self.last_obs
        elif len(s) == 3 and s[0] == 1:
            import cv2
            return cv2.cvtColor(self.last_obs[0], cv2.COLOR_GRAY2BGR)
        else:
            raise NotImplementedError()
        
# class SkipWrapper(gymnasium.Wrapper):
#     """
#         Generic common frame skipping wrapper
#         Will perform action for `x` additional steps
#     """
#     def __init__(self, env, repeat_count):
#         super(SkipWrapper, self).__init__(env)
#         self.repeat_count = repeat_count
#         self.stepcount = 0

#     def step(self, action):
#         done = False
#         total_reward = 0
#         current_step = 0
#         while current_step < (self.repeat_count + 1) and not done:
#             self.stepcount += 1
#             obs, reward, done, info = self.env.step(action)
#             total_reward += reward
#             current_step += 1

#         return obs, total_reward, done, info

#     def reset(self, seed=None, options=None):
#         self.stepcount = 0
#         return self.env.reset(seed=seed, options=options)

