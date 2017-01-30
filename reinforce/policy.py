import gym.spaces
import tensorflow as tf
from reinforce.models.visionnet import vision_net
from reinforce.distributions import Categorical

class VisionPolicy(object):

  def __init__(self, observation_space, action_space, sess):
    assert isinstance(action_space, gym.spaces.Discrete)
    self.observations = tf.placeholder(tf.float32, shape=(None, 80, 80, 3))
    self.advantages = tf.placeholder(tf.float32, shape=(None,))
    self.actions = tf.placeholder(tf.int64, shape=(None,))
    self.prev_logits = tf.placeholder(tf.float32, shape=(None, action_space.n))
    self.prev_dist = Categorical(self.prev_logits)
    self.curr_logits = vision_net(self.observations, num_classes=action_space.n)
    self.curr_dist = Categorical(self.curr_logits)
    self.sampler = self.curr_dist.sample()
    # Make loss functions.
    self.ratio = tf.exp(self.curr_dist.logp(self.actions) - self.prev_dist.logp(self.actions))
    self.loss = self.ratio * self.advantages
    self.sess = sess

  def compute_actions(self, observations):
    return self.sess.run(self.sampler, feed_dict={self.observations: observations})

  def loss(self):
    return self.loss

  def compute_loss(self, observations, advantages, actions, prev_logits):
    return self.sess.run(self.loss, feed_dict={self.observations: observations,
                                               self.advantages: advantages,
                                               self.actions: actions,
                                               self.prev_logits: prev_logits})
