import tensorflow as tf

class Categorical(object):

  def __init__(self, logits):
    self.logits = logits

  def logp(self, x):
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, x)

  # def compute_logp(self, logits, x):
    # z = np.exp(logits - np.max(logits))
    # p = z / z.sum(axis=1)[:,None]
    # return p[x]

  def sample(self):
    return tf.multinomial(self.logits, 1)
