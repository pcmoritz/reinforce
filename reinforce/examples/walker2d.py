import tensorflow as tf
from reinforce.env import BatchedEnv, atari_preprocessor, ram_preprocessor
from reinforce.policy import ProximalPolicyLoss
from reinforce.rollout import rollout, add_advantage_values
from reinforce.filter import MeanStdFilter
from reinforce.utils import flatten, iterate

config = {"kl_coeff": 1.0,
          "num_sgd_iter": 20,
          "sgd_stepsize": 5e-5,
          "sgd_batchsize": 128,
          "entropy_coeff": 0.0,
          "clip_param": 1.0,
          "kl_target": 0.01}

env = BatchedEnv("Hopper-v1", 512, preprocessor=None)
sess = tf.Session()
ppo = ProximalPolicyLoss(env.observation_space, env.action_space, config, sess)

optimizer = tf.train.AdamOptimizer(config["sgd_stepsize"])

train_op = optimizer.minimize(ppo.loss)

sess.run(tf.global_variables_initializer())

kl_coeff = config["kl_coeff"]

observation_filter = MeanStdFilter(env.observation_space.shape)
reward_filter = MeanStdFilter(())

for j in range(1000):
  print("iteration = ", j)
  trajectory = rollout(ppo, env, 5000, observation_filter, reward_filter)
  total_reward = trajectory["unfiltered_rewards"].sum(axis=0).mean()
  print("total reward is ", total_reward)
  add_advantage_values(trajectory, 0.995, 0.95)
  trajectory = flatten(trajectory)
  print("timesteps: ", trajectory["rewards"].shape[0])
  # print("mean state: ", trajectory["observations"].mean(axis=0))
  print("filter mean: ", observation_filter.rs.mean)
  trajectory["advantages"] = (trajectory["advantages"] - trajectory["advantages"].mean()) / trajectory["advantages"].std()
  print("Computing policy (optimizer='" + optimizer.get_name() + "', iterations=" + str(config["num_sgd_iter"]) + ", stepsize=" + str(config["sgd_stepsize"]) + "):")
  names = ["iter", "loss", "kl", "entropy"]
  print(("{:>15}" * len(names)).format(*names))
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
