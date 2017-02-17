import tensorflow as tf
from reinforce.env import BatchedEnv, atari_preprocessor, ram_preprocessor
from reinforce.policy import ProximalPolicyLoss
from reinforce.rollout import rollouts, add_advantage_values, collect_samples
from reinforce.filter import MeanStdFilter
from reinforce.utils import iterate

config = {"kl_coeff": 0.2,
          "num_sgd_iter": 30,
          "sgd_stepsize": 5e-5,
          "sgd_batchsize": 128,
          "entropy_coeff": 0.0,
          "clip_param": 0.3,
          "kl_target": 0.02,
          "timesteps_per_batch": 10000}

env = BatchedEnv("Hopper-v1", 1, preprocessor=None)
sess = tf.Session()
ppo = ProximalPolicyLoss(env.observation_space, env.action_space, config, sess)

optimizer = tf.train.AdamOptimizer(config["sgd_stepsize"])

train_op = optimizer.minimize(ppo.loss)

sess.run(tf.global_variables_initializer())

kl_coeff = config["kl_coeff"]

observation_filter = MeanStdFilter(env.observation_space.shape)
reward_filter = MeanStdFilter((), clip=5.0)

for j in range(1000):
  print("iteration = ", j)
  trajectory, total_reward, traj_len_mean = collect_samples(config["timesteps_per_batch"], 0.995, 0.95, ppo, env, 1000, observation_filter, reward_filter)
  print("total reward is ", total_reward)
  print("trajectory length mean is ", traj_len_mean)
  print("timesteps: ", trajectory["dones"].shape[0])
  print("mean state: ", trajectory["observations"].mean(axis=0))
  print("filter mean: ", observation_filter.rs.mean)
  trajectory["advantages"] = (trajectory["advantages"] - trajectory["advantages"].mean()) / trajectory["advantages"].std()
  print("Computing policy (optimizer='" + optimizer.get_name() + "', iterations=" + str(config["num_sgd_iter"]) + ", stepsize=" + str(config["sgd_stepsize"]) + "):")
  names = ["iter", "loss", "kl", "entropy"]
  print(("{:>15}" * len(names)).format(*names))
  # import IPython
  # IPython.embed()
  for i in range(config["num_sgd_iter"]):
    # Test on current set of rollouts
    loss, kl, entropy = sess.run([ppo.loss, ppo.mean_kl, ppo.mean_entropy],
                          feed_dict={ppo.observations: trajectory["observations"],
                                     ppo.advantages: trajectory["advantages"],
                                     ppo.actions: trajectory["actions"].squeeze(),
                                     ppo.prev_logits: trajectory["logprobs"],
                                     ppo.kl_coeff: kl_coeff})
    print("{:>15}{:15.5e}{:15.5e}{:15.5e}".format(i, loss, kl, entropy))
    # Run SGD for training on current set of rollouts
    for batch in iterate(trajectory, config["sgd_batchsize"]):
      sess.run([train_op],
                                  feed_dict={ppo.observations: batch["observations"],
                                             ppo.advantages: batch["advantages"],
                                             ppo.actions: batch["actions"].squeeze(),
                                             ppo.prev_logits: batch["logprobs"],
                                             ppo.kl_coeff: kl_coeff})
  if kl > 2.0 * config["kl_target"]:
    kl_coeff *= 1.5
  elif kl < 0.5 * config["kl_target"]:
    kl_coeff *= 0.5
  print("kl div = ", kl)
  print("kl coeff = ", kl_coeff)
