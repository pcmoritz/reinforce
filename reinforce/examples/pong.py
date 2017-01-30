import tensorflow as tf
from reinforce.env import BatchedEnv, atari_preprocessor
from reinforce.policy import VisionPolicy
from reinforce.rollout import rollout, add_advantage_values

env = BatchedEnv("Pong-v3", 16, preprocessor=atari_preprocessor)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
policy = VisionPolicy(env.observation_space, env.action_space, sess)

trajectory = rollout(policy, env, 100)
add_advantage_values(trajectory, 0.99, 0.95)

policy.compute_loss(trajectory["observations"][0,:], trajectory["logprobs"][0,:], trajectory["advantages"][0,:], trajectory["actions"][0].squeeze())
