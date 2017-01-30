import tensorflow as tf

class Categorical(object):

  def __init__(self, logits):
    self.logits = logits

  def logp(self, x):
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, x)

  def entropy(self):
    a0 = self.logits - tf.reduce_max(self.logits, reduction_indices=[1], keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, reduction_indices=[1], keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), reduction_indices=[1])

  def kl(self, other):
    a0 = self.logits - tf.reduce_max(self.logits, reduction_indices=[1], keep_dims=True)
    a1 = other.logits - tf.reduce_max(other.logits, reduction_indices=[1], keep_dims=True)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = tf.reduce_sum(ea0, reduction_indices=[1], keep_dims=True)
    z1 = tf.reduce_sum(ea1, reduction_indices=[1], keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), reduction_indices=[1])

  def sample(self):
    return tf.multinomial(self.logits, 1)
