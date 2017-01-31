import tensorflow as tf
from reinforce.env import BatchedEnv, atari_preprocessor
from reinforce.policy import VisionPolicy
from reinforce.rollout import rollout, add_advantage_values
from reinforce.utils import flatten

config = {"kl_coeff": 0.2, "num_sgd_iter": 10, "sgd_stepsize": 5e-4}

env = BatchedEnv("Pong-v3", 32, preprocessor=atari_preprocessor)
sess = tf.Session()
policy = VisionPolicy(env.observation_space, env.action_space, config, sess)

optimizer = tf.train.AdamOptimizer(config["sgd_stepsize"])

train_op = optimizer.minimize(policy.loss)

sess.run(tf.global_variables_initializer())

for j in range(1000):
  print("iteration = ", j)
  trajectory = rollout(policy, env, 1000)
  add_advantage_values(trajectory, 0.99, 0.95)
  trajectory = flatten(trajectory)
  print("reward mean = ", trajectory["rewards"].mean())
  trajectory["advantages"] = (trajectory["advantages"] - trajectory["advantages"].mean()) / trajectory["advantages"].std()
  print("Computing policy (optimizer='" + optimizer.get_name() + "', iterations=" + str(config["num_sgd_iter"]) + ", stepsize=" + str(config["sgd_stepsize"]) + "):")
  names = ["iter", "loss", "kl"]
  print(("{:>15}" * len(names)).format(*names))
  for i in range(config["num_sgd_iter"]):
    loss, kl, _ = sess.run([policy.loss, policy.mean_kl, train_op], feed_dict={policy.observations: trajectory["observations"],
                                                                               policy.advantages: trajectory["advantages"],
                                                                               policy.actions: trajectory["actions"].squeeze(),
                                                                               policy.prev_logits: trajectory["logprobs"]})
    print("{:>15}{:15.5e}{:15.5e}".format(i,loss, kl))
  print("kl diff = ", policy.compute_kl(trajectory["observations"], trajectory["logprobs"]))
