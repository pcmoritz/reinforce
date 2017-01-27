import numpy as np

def rollout(policy, env, max_timesteps, stochastic=True):
  observation = env.reset()
  done = np.array(env.batchsize * [False])
  t = 0

  observations = []
  rewards = []
  actions = []
  dones = []

  while not done.any() and t < max_timesteps:
    action = policy.compute_actions(observation)
    observations.append(observation)
    actions.append(action)
    observation, reward, done = env.step(action)
    rewards.append(reward)
    t += 1

  return {"observations": np.vstack(observations),
          "rewards": np.vstack(rewards),
          "actions": np.vstack(actions)}
