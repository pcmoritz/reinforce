import numpy as np

class NoFilter(object):

  def __init__(self):
    pass

  def __call__(self, x, update=True):
    return np.asarray(x)

# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):

  def __init__(self, shape=None):
    self._n = 0
    self._M = np.zeros(shape)
    self._S = np.zeros(shape)

  def push(self, x):
    x = np.asarray(x)
    # Unvectorized update of the running statistics.
    assert x.shape == self._M.shape, "x.shape = {}, self.shape = {}".format(x.shape, self._M.shape)
    self._n += 1
    if self._n == 1:
      self._M[...] = x
    else:
      old_M = self._M.copy()
      self._M[...] = old_M + (x - old_M)/self._n
      self._S[...] = self._S + (x - old_M)*(x - self._M)

  @property
  def n(self):
    return self._n

  @property
  def mean(self):
    return self._M

  @property
  def var(self):
    return self._S/(self._n - 1) if self._n > 1 else np.square(self._M)

  @property
  def std(self):
    return np.sqrt(self.var)

  @property
  def shape(self):
    return self._M.shape

class MeanStdFilter(object):
  """
  y = (x-mean)/std
  using running estimates of mean,std
  """

  def __init__(self, shape, demean=True, destd=True, clip=10.0):
    self.demean = demean
    self.destd = destd
    self.clip = clip

    self.rs = RunningStat(shape)

  def __call__(self, x, update=True):
    x = np.asarray(x)
    if update:
      if len(x.shape) == len(self.rs.shape) + 1:
        # The vectorized case.
        for i in range(x.shape[0]):
          self.rs.push(x[i])
      else:
        # The unvectorized case.
        self.rs.push(x)
    if self.demean:
      x = x - self.rs.mean
    if self.destd:
      x = x / (self.rs.std+1e-8)
    if self.clip:
      x = np.clip(x, -self.clip, self.clip)
    return x


def test_running_stat():
  for shp in ((), (3,), (3,4)):
    li = []
    rs = RunningStat(shp)
    for _ in range(5):
      val = np.random.randn(*shp)
      rs.push(val)
      li.append(val)
      m = np.mean(li, axis=0)
      assert np.allclose(rs.mean, m)
      v = np.square(m) if (len(li) == 1) else np.var(li, ddof=1, axis=0)
      assert np.allclose(rs.var, v)
