import gym.spaces
import tensorflow as tf
from reinforce.models.visionnet import vision_net
from reinforce.models.fcnet import fc_net
from reinforce.distributions import Categorical

class VisionPolicy(object):

  def __init__(self, observation_space, action_space, config, sess):
    assert isinstance(action_space, gym.spaces.Discrete)
    # adapting the kl divergence
    self.kl_coeff = tf.placeholder(name='newkl', shape=(), dtype=tf.float32)
    # self.observations = tf.placeholder(tf.float32, shape=(None, 80, 80, 3))
    self.observations = tf.placeholder(tf.float32, shape=(None, 128))
    self.advantages = tf.placeholder(tf.float32, shape=(None,))
    self.actions = tf.placeholder(tf.int64, shape=(None,))
    self.prev_logits = tf.placeholder(tf.float32, shape=(None, action_space.n))
    self.prev_dist = Categorical(self.prev_logits)
    # self.curr_logits = vision_net(self.observations, num_classes=action_space.n)
    self.curr_logits = fc_net(self.observations, num_classes=action_space.n)
    self.curr_dist = Categorical(self.curr_logits)
    self.sampler = self.curr_dist.sample()
    self.entropy = self.curr_dist.entropy()
    tf.summary.scalar("policy entropy", tf.reduce_mean(self.entropy))
    # Make loss functions.
    self.ratio = tf.exp(self.curr_dist.logp(self.actions) - self.prev_dist.logp(self.actions))
    self.kl = self.prev_dist.kl(self.curr_dist)
    self.mean_kl = tf.reduce_mean(self.kl)
    self.mean_entropy = tf.reduce_mean(self.entropy)
    # XXX
    self.surr1 = self.ratio * self.advantages
    self.surr2 = tf.clip_by_value(self.ratio, 1 - config["clip_param"], 1 + config["clip_param"]) * self.advantages
    self.surr = tf.minimum(self.surr1, self.surr2)
    self.loss = tf.reduce_mean(-self.surr + self.kl_coeff * self.kl - config["entropy_coeff"] * self.entropy)
    tf.summary.scalar("loss", self.loss)
    self.sess = sess

  def compute_actions(self, observations):
    return self.sess.run([self.sampler, self.curr_logits], feed_dict={self.observations: observations})

  def loss(self):
    return self.loss

#  def compute_loss(self, observations, advantages, actions, prev_logits):
#    return self.sess.run(self.loss, feed_dict={self.observations: observations,
#                                               self.advantages: advantages,
#                                               self.actions: actions,
#                                               self.prev_logits: prev_logits})

  def compute_kl(self, observations, prev_logits):
    return self.sess.run(tf.reduce_mean(self.kl), feed_dict={self.observations: observations,
                                                             self.prev_logits: prev_logits})
