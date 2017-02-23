import numpy as np
import ray

from reinforce.filter import NoFilter
from reinforce.utils import flatten, concatenate

def rollouts(policy, env, horizon, observation_filter=NoFilter(), reward_filter=NoFilter()):
  """Perform a batch of rollouts of a policy in an environment.

  Args:
    policy: The policy that will be rollout out. Can be an arbitrary object
      that supports a compute_actions(observation) function.
    env: The environment the rollout is computed in. Needs to support the
      OpenAI gym API and needs to support batches of data.
    horizon: Upper bound for the number of timesteps for each rollout in the
      batch.
    observation_filter: Function that is applied to each of the observations.
    reward_filter: Function that is applied to each of the rewards.

  Returns:
    A trajectory, which is a dictionary with keys "observations", "rewards",
    "orig_rewards", "actions", "logprobs", "dones". Each value is an array of
    shape (num_timesteps, env.batchsize, shape).
  """

  observation = observation_filter(env.reset())
  done = np.array(env.batchsize * [False])
  t = 0
  observations = []
  raw_rewards = [] # Empirical rewards
  actions = []
  logprobs = []
  dones = []

  while not done.all() and t < horizon:
    action, logprob = policy.compute_actions(observation)
    observations.append(observation[None])
    actions.append(action[None])
    logprobs.append(logprob[None])
    observation, raw_reward, done = env.step(action)
    observation = observation_filter(observation)
    raw_rewards.append(raw_reward[None])
    # if reward_filter.rs._n > 500:
    #   filt_rewards.append(reward_filter(raw_reward, update=False)[None])
    # else:
    #   print("raw_reward is ", raw_reward)
    #   filt_rewards.append(raw_reward - raw_reward.mean(axis=0))
    dones.append(done[None])
    t += 1

  return {"observations": np.vstack(observations),
          "raw_rewards": np.vstack(raw_rewards),
          "actions": np.vstack(actions),
          "logprobs": np.vstack(logprobs),
          "dones": np.vstack(dones)}

def add_advantage_values(trajectory, gamma, lam, reward_filter):
  # Compute filtered rewards
  raw_rewards = trajectory["raw_rewards"]
  # filt_rewards = trajectory["filt_rewards"] = np.zeros_like(raw_rewards)
  # if reward_filter.rs._n < 200:
  #   # We have collected an insufficient number of observations so far,
  #   # so we cannot use the reward filter
  #   rew_mean = raw_rewards.mean(axis=0)
  #   rew_std = raw_rewards.std(axis=0)
  #   for i in range(raw_rewards.shape[0]):
  #     filt_rewards[i] = (raw_rewards[i] - rew_mean) / rew_std
  # else:
  #   for i in range(raw_rewards.shape[0]):
  #     filt_rewards[i] = reward_filter(raw_rewards[i])
  trajectory["filt_rewards"] = raw_rewards

  filt_rewards = trajectory["filt_rewards"]
  # print("filt_rewards", filt_rewards)
  dones = trajectory["dones"]
  advantages = np.zeros_like(filt_rewards)
  last_advantage = np.zeros(filt_rewards.shape[1], dtype="float32")

  for t in reversed(range(len(filt_rewards))):
    delta = filt_rewards[t,:] * (1 - dones[t,:])
    last_advantage = delta + gamma * lam * last_advantage
    # print("last_advantage is ", last_advantage)
    advantages[t,:] = last_advantage
    # Update the reward filter
    # print("advantages", advantages[t,:])
    reward_filter(advantages[t,:])

  trajectory["advantages"] = advantages

@ray.remote
def compute_trajectory(policy, env, gamma, lam, horizon, observation_filter, reward_filter):
  trajectory = rollouts(policy, env, horizon, observation_filter, reward_filter)
  add_advantage_values(trajectory, gamma, lam, reward_filter)
  return trajectory

def collect_samples(agents, num_timesteps, gamma, lam, horizon, observation_filter=NoFilter(), reward_filter=NoFilter()):
  num_timesteps_so_far = 0
  trajectories = []
  total_rewards = []
  traj_len_means = []
  while num_timesteps_so_far < num_timesteps:
    trajectory_batch = ray.get([agent.compute_trajectory(gamma, lam, horizon) for agent in agents])
    # try:
    trajectory = concatenate(trajectory_batch)
    # except:
    #   import IPython
    #   IPython.embed()
    total_rewards.append(trajectory["raw_rewards"].sum(axis=0).mean() / len(agents))
    # print("mean", reward_filter.rs.mean)
    # print("std", reward_filter.rs.std)
    # print("raw_rewards", trajectory["raw_rewards"])
    # print("filt_rewards", trajectory["filt_rewards"])
    # print("")
    # import IPython
    # IPython.embed()
    trajectory = flatten(trajectory)
    not_done = np.logical_not(trajectory["dones"])
    traj_len_means.append(not_done.sum(axis=0).mean() / len(agents))
    trajectory = {key: val[not_done] for key, val in trajectory.items()}
    num_timesteps_so_far += len(trajectory["dones"])
    trajectories.append(trajectory)
  return concatenate(trajectories), np.mean(total_rewards), np.mean(traj_len_means)
