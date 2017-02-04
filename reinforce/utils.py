import numpy as np

def flatten(weights, start=0, stop=2):
  for key, val in weights.items():
    dims = val.shape[0:start] + (-1,) + val.shape[stop:]
    weights[key] = val.reshape(dims)
  return weights

def shuffle(trajectory):
  permutation = np.random.permutation(trajectory["rewards"].shape[0])
  for key, val in trajectory.items():
    trajectory[key] = val[permutation][permutation]
  return trajectory

def iterate(trajectory, batchsize):
  trajectory = shuffle(trajectory)
  curr_index = 0
  # XXX consume the whole batch
  while curr_index + batchsize < trajectory["rewards"].shape[0]:
    batch = dict()
    for key in trajectory:
      batch[key] = trajectory[key][curr_index:curr_index+batchsize]
    curr_index += batchsize
    yield batch
