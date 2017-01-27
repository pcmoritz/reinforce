import gym
import numpy as np

# TODO(pcm): Make this a class and provide a method that transforms
# observation_space in the environment.
def atari_preprocessor(observation):
  "Convert images from (210, 160, 3) to (3, 80, 80) by downsampling."
  return np.transpose(observation[25:-25:2,::2,:], (2, 0, 1))[None]

# TODO(pcm): Make this a Ray actor.
class BatchedEnv(object):
  "A BatchedEnv holds multiple gym enviroments and performs steps on all of them."

  def __init__(self, name, batchsize, preprocessor=None):
    self.envs = [gym.make(name) for _ in range(batchsize)]
    self.observation_space = self.envs[0].observation_space
    self.action_space = self.envs[0].action_space
    self.batchsize = batchsize
    self.preprocessor = preprocessor if preprocessor else lambda obs: obs[None]

  def reset(self):
    observations = [self.preprocessor(env.reset()) for env in self.envs]
    return np.vstack(observations)

  def step(self, actions):
    observations = []
    rewards = []
    dones = []
    for i, action in enumerate(actions):
      observation, reward, done, info = self.envs[i].step(action)
      observations.append(self.preprocessor(observation))
      rewards.append(reward)
      dones.append(done)
    return np.vstack(observations), np.array(rewards), np.array(dones)
