import tensorflow as tf
from reinforce.env import BatchedEnv, atari_preprocessor, ram_preprocessor
from reinforce.policy import VisionPolicy
from reinforce.rollout import rollout, add_advantage_values
from reinforce.utils import flatten, iterate

config = {"kl_coeff": 0.1,
          "num_sgd_iter": 10,
          "sgd_stepsize": 1e-5,
          "sgd_batchsize": 128,
          "entropy_coeff": 0.0,
          "clip_param": 1.0,
          "kl_target": 0.01}

env = BatchedEnv("Pong-ramDeterministic-v3", 128, preprocessor=ram_preprocessor)
sess = tf.Session()
policy = VisionPolicy(env.observation_space, env.action_space, config, sess)

optimizer = tf.train.AdamOptimizer(config["sgd_stepsize"])

train_op = optimizer.minimize(policy.loss)

sess.run(tf.global_variables_initializer())

kl_coeff = config["kl_coeff"]

for j in range(1000):
  print("iteration = ", j)
  trajectory = rollout(policy, env, 2000)
  total_reward = trajectory["rewards"].sum(axis=0).mean()
  print("total reward is ", total_reward)
  add_advantage_values(trajectory, 0.99, 0.95)
  trajectory = flatten(trajectory)
  trajectory["advantages"] = (trajectory["advantages"] - trajectory["advantages"].mean()) / trajectory["advantages"].std()
  print("Computing policy (optimizer='" + optimizer.get_name() + "', iterations=" + str(config["num_sgd_iter"]) + ", stepsize=" + str(config["sgd_stepsize"]) + "):")
  names = ["iter", "loss", "kl", "entropy"]
  print(("{:>15}" * len(names)).format(*names))
  for i in range(config["num_sgd_iter"]):
    total_loss = 0.0
    for batch in iterate(trajectory, config["sgd_batchsize"]):
      loss, kl, entropy, _ = sess.run([policy.loss, policy.mean_kl, policy.mean_entropy, train_op],
                               feed_dict={policy.observations: batch["observations"],
                                          policy.advantages: batch["advantages"],
                                          policy.actions: batch["actions"].squeeze(),
                                          policy.prev_logits: batch["logprobs"],
                                          policy.kl_coeff: kl_coeff})
      total_loss += loss
    print("{:>15}{:15.5e}{:15.5e}{:15.5e}".format(i, total_loss, kl, entropy))
  if kl > 2.0 * config["kl_target"]:
    kl_coeff *= 1.5
  elif kl < 0.5 * config["kl_target"]:
    kl_coeff *= 0.5
  print("kl div = ", kl)
  print("kl coeff = ", kl_coeff)
