import tensorflow as tf
from reinforce.env import BatchedEnv, atari_preprocessor, ram_preprocessor
from reinforce.policy import VisionPolicy
from reinforce.rollout import rollout, add_advantage_values
from reinforce.utils import flatten

config = {"kl_coeff": 0.001,
          "num_sgd_iter": 20,
          "sgd_stepsize": 5e-4,
          "entropy_coeff": 0.0,
          "clip_param": 0.2,
          "kl_target": 0.02}

env = BatchedEnv("Pong-ramDeterministic-v3", 64, preprocessor=ram_preprocessor)
sess = tf.Session()
policy = VisionPolicy(env.observation_space, env.action_space, config, sess)

optimizer = tf.train.AdamOptimizer(config["sgd_stepsize"])

train_op = optimizer.minimize(policy.loss)

sess.run(tf.global_variables_initializer())

kl_coeff = config["kl_coeff"]

for j in range(1000):
  print("iteration = ", j)
  trajectory = rollout(policy, env, 2000)
  add_advantage_values(trajectory, 0.99, 0.95)
  trajectory = flatten(trajectory)
  print("reward mean = ", trajectory["rewards"].mean())
  trajectory["advantages"] = (trajectory["advantages"] - trajectory["advantages"].mean()) / trajectory["advantages"].std()
  print("Computing policy (optimizer='" + optimizer.get_name() + "', iterations=" + str(config["num_sgd_iter"]) + ", stepsize=" + str(config["sgd_stepsize"]) + "):")
  names = ["iter", "loss", "kl", "entropy"]
  print(("{:>15}" * len(names)).format(*names))
  for i in range(config["num_sgd_iter"]):
    loss, kl, entropy, _ = sess.run([policy.loss, policy.mean_kl, policy.mean_entropy, train_op],
                             feed_dict={policy.observations: trajectory["observations"],
                                        policy.advantages: trajectory["advantages"],
                                        policy.actions: trajectory["actions"].squeeze(),
                                        policy.prev_logits: trajectory["logprobs"],
                                        policy.kl_coeff: kl_coeff})
    print("{:>15}{:15.5e}{:15.5e}{:15.5e}".format(i,loss, kl, entropy))
  if kl > 2.0 * config["kl_target"]:
    kl_coeff *= 1.5
  elif kl < 0.5 * config["kl_target"]:
    kl_coeff *= 0.5
  print("kl div = ", kl)
  print("kl coeff = ", kl_coeff)
