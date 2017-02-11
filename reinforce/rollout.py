import numpy as np

def rollout(policy, env, max_timesteps, observation_filter=lambda obs: obs, reward_filter=lambda rew: rew, stochastic=True):
  observation = env.reset()
  done = np.array(env.batchsize * [False])
  t = 0

  observations = []
  rewards = []
  unfiltered_rewards = []
  actions = []
  logprobs = []
  dones = []

  while not done.any() and t < max_timesteps:
    action, logprob = policy.compute_actions(observation)
    observations.append(observation[None])
    actions.append(action[None])
    logprobs.append(logprob[None])
    observation, reward, done = env.step(action)
    observation = observation_filter(observation)
    unfiltered_rewards.append(reward[None])
    rewards.append(reward_filter(reward)[None])
    t += 1

  return {"observations": np.vstack(observations),
          "unfiltered_rewards": np.vstack(unfiltered_rewards),
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
