import tensorflow as tf
from reinforce.env import BatchedEnv, atari_preprocessor
from reinforce.policy import VisionPolicy
from reinforce.rollout import rollout, add_advantage_values
from reinforce.utils import flatten

env = BatchedEnv("Pong-v3", 16, preprocessor=atari_preprocessor)
sess = tf.Session()
policy = VisionPolicy(env.observation_space, env.action_space, sess)

optimizer = tf.train.AdamOptimizer(1e-4)

train_op = optimizer.minimize(policy.loss)

sess.run(tf.global_variables_initializer())

for j in range(20):
  print("iteration = ", j)
  trajectory = rollout(policy, env, 1000)
  add_advantage_values(trajectory, 0.99, 0.95)
  trajectory = flatten(trajectory)
  print("reward mean = ", trajectory["rewards"].mean())
  trajectory["advantages"] = (trajectory["advantages"] - trajectory["advantages"].mean()) / trajectory["advantages"].std()
  for i in range(10):
    loss, _ = sess.run([policy.loss, train_op], feed_dict={policy.observations: trajectory["observations"],
                                                           policy.advantages: trajectory["advantages"],
                                                           policy.actions: trajectory["actions"].squeeze(),
                                                           policy.prev_logits: trajectory["logprobs"]})
    print("loss = ", loss)
  print "kl diff = ", policy.compute_kl(trajectory["logprobs"])
