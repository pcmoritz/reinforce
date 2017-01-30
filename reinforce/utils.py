def flatten(weights, start=0, stop=2):
  for key, val in weights.items():
    dims = val.shape[0:start] + (-1,) + val.shape[stop:]
    weights[key] = val.reshape(dims)
  return weights
