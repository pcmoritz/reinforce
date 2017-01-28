import numpy as np

def rollout(policy, env, max_timesteps, stochastic=True):
  observation = env.reset()
  done = np.array(env.batchsize * [False])
  t = 0

  observations = []
  rewards = []
  actions = []
  logprobs = []
  dones = []

  while not done.any() and t < max_timesteps:
    action, logprob = policy.compute_actions(observation)
    observations.append(observation)
    actions.append(action)
    logprobs.append(logprob)
    observation, reward, done = env.step(action)
    rewards.append(reward)
    t += 1

  return {"observations": np.vstack(observations),
          "rewards": np.vstack(rewards),
          "actions": np.vstack(actions),
          "logprobs": np.vstack(logprobs)}

def add_advantage_values(trajectory, gamma, lam):
  rewards = trajectory["rewards"]
  advantages = np.zeros_like(rewards)
  last_advantage = np.zeros(rewards.shape[1], dtype="float32")

  for t in reversed(range(len(rewards))):
    delta = rewards[t,:]
    last_advantage = delta + gamma * lam * last_advantage
    advantages[t,:] = last_advantage

  trajectory["advantages"] = advantages
