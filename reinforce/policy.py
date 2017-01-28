import gym.spaces
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from reinforce.models.visionnet import VisionNet

class VisionPolicy(object):

  def __init__(self, observation_space, action_space, gpu_index=0):
    assert isinstance(action_space, gym.spaces.Discrete)
    self.gpu_index = gpu_index
    self.net = VisionNet(num_classes=action_space.n)
    self.net = self.net.float().cuda(self.gpu_index)

  def compute_logprob(self, observations):
    input = torch.from_numpy(observations).type(torch.FloatTensor).cuda(self.gpu_index)
    return self.net.forward(autograd.Variable(input))

  def compute_actions(self, observations):
    logprob = self.compute_logprob(observations)
    return F.softmax(logprob).multinomial().cpu().data.numpy(), logprob.cpu().data.numpy()

  def compute_loss(self, trajectory):
    logprobs = self.compute_logprob(trajectory["observations"])
    return torch.exp(logprobs - autograd.Variable(torch.from_numpy(trajectory["logprobs"]).cuda())) * torch.from_numpy(trajectory["advantages"]).cuda()
