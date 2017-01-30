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

writer = tf.summary.FileWriter("/tmp/training/", sess.graph)

sess.run(tf.global_variables_initializer())

# trajectory = rollout(policy, env, 100)
# add_advantage_values(trajectory, 0.99, 0.95)

# policy.compute_loss(trajectory["observations"][0,:], trajectory["advantages"][0,:], trajectory["actions"][0].squeeze(), trajectory["logprobs"][0,:])

# trajectory = flatten(trajectory)

# standardize advantages

merged = tf.summary.merge_all()

for j in range(10):
  print("iteration = ", j)
  trajectory = rollout(policy, env, 1000)
  add_advantage_values(trajectory, 0.99, 0.95)
  trajectory = flatten(trajectory)
  print("reward mean = ", trajectory["rewards"].mean())
  trajectory["advantages"] = (trajectory["advantages"] - trajectory["advantages"].mean()) / trajectory["advantages"].std()
  for i in range(2):
    summary, _ = sess.run([merged, train_op], feed_dict={policy.observations: trajectory["observations"],
                                                         policy.advantages: trajectory["advantages"],
                                                         policy.actions: trajectory["actions"].squeeze(),
                                                         policy.prev_logits: trajectory["logprobs"]})
    writer.add_summary(summary, i)
    writer.flush()
