
import abc

import numpy as np
from scipy import stats
import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
import torch.nn.functional as F
# from pytorch_generative.models import base

class SoftTargetDistillLoss(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTargetDistillLoss, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss



class Kernel(abc.ABC, nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bandwidth=1.0):
        """Initializes a new Kernel.
        Args:
            bandwidth: The kernel's (band)width.
        """
        super().__init__()
        self.bandwidth = bandwidth

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    @abc.abstractmethod
    def forward(self, test_Xs, train_Xs):
        """Computes log p(x) for each x in test_Xs given train_Xs."""

    @abc.abstractmethod
    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class ParzenWindowKernel(Kernel):
    """Implementation of the Parzen window kernel."""

    def forward(self, test_Xs, train_Xs):
        abs_diffs = torch.abs(self._diffs(test_Xs, train_Xs))
        dims = tuple(range(len(abs_diffs.shape))[2:])
        dim = np.prod(abs_diffs.shape[2:])
        inside = torch.sum(abs_diffs / self.bandwidth <= 0.5, dim=dims) == dim
        coef = 1 / self.bandwidth**dim
        return torch.log((coef * inside).mean(dim=1))

    @torch.no_grad()
    def sample(self, train_Xs):
        device = train_Xs.device
        noise = (torch.rand(train_Xs.shape, device=device) - 0.5) * self.bandwidth
        return train_Xs + noise


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        n, d = train_Xs.shape
        n, h = torch.tensor(n, dtype=torch.float32), torch.tensor(self.bandwidth)
        pi = torch.tensor(np.pi)

        Z = 0.5 * d * torch.log(2 * pi) + d * torch.log(h) + torch.log(n)
        diffs = self._diffs(test_Xs, train_Xs) / h
        log_exp = -0.5 * torch.norm(diffs, p=2, dim=-1) ** 2

        return torch.logsumexp(log_exp - Z, dim=-1)

    @torch.no_grad()
    def sample(self, train_Xs):
        device = train_Xs.device
        noise = torch.randn(train_Xs.shape, device=device) * self.bandwidth
        return train_Xs + noise


class KernelDensityEstimator(nn.Module):
    """The KernelDensityEstimator model."""
    def __init__(self, train_Xs, kernel=None):
        """Initializes a new KernelDensityEstimator.
        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super(KernelDensityEstimator, self).__init__()
        self.kernel = kernel or GaussianKernel()
        self.train_Xs = train_Xs
        assert len(self.train_Xs.shape) == 2, "Input cannot have more than two axes."

    @property
    def device(self):
        return self.train_Xs.device

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an
    # iterative version instead.
    @torch.no_grad()
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    @torch.no_grad()
    def sample(self, n_samples):
        idxs = np.random.choice(range(len(self.train_Xs)), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])


def kl_divergence(dist1, dist2):
    num_samples_dist1 = dist1.shape[0]
    num_samples_dist2 = dist2.shape[0]
    num_samples_to_select = min(dist1, dist2)

    if num_samples_to_select < dist1.shape:
        indices = torch.randperm(len(dist1))[:num_samples_to_select]
        dist1 = dist1[indices]

    if num_samples_to_select < dist2.shape:
        indices = torch.randperm(len(dist2))[:num_samples_to_select]
        dist2 = dist2[indices]
    
    kl_div = torch.nn.functional.kl_div(dist1, dist2)

    print(dist1.shape)
    print(dist2.shape)
    return kl_div

def calculate_kl_div_from_data(data1,data2):
    data1 = torch.flatten(data1, start_dim = 1)
    data2 = torch.flatten(data2,  start_dim = 1)
    device = data1.get_device()

    print(data1.shape, data2.shape)

    # kde_1 = KernelDensityEstimator(data1)
    # kde_2 = KernelDensityEstimator(data2)

    # dist1 = kde_1(data1)
    # dist2 = kde_2(data2)
    data1 = data1.detach().cpu().numpy()
    data2 = data2.detach().cpu().numpy()

    kernel_1 = stats.gaussian_kde(data1.T)
    kernel_2 = stats.gaussian_kde(data2.T)

    dist1 = kernel_1.evaluate(data1.T)
    dist2 = kernel_2.evaluate(data2.T)

    dist1 = torch.from_numpy(dist1).to(device)
    dist1 = torch.from_numpy(dist1).to(device)

    kl_div = kl_divergence(dist1, dist2)


    print(dist1.shape, dist2.shape)

    return kl_div

def model_weight_divergence(model1,model2):
    with torch.no_grad():
        # print(torch.norm(model1))
        # print(torch.norm(model2))
        sum_weight_div = 0
        sum_model2_norm = 0
        num_layers = 0


        model1_weights = []
        model2_weights = []

        for layer in model1.parameters():
            print(layer.weight)
            model1_weights.append(layer.weight)
            num_layers += 1

        for layer in model2.parameters():
            print(layer.weight)
            model2_weights.append(layer.weight)

        for i in range(num_layers):
            weight1 = model1_weights[i].weight
            weight2 = model2_weights[i].weight
            layer_weight_div  = torch.norm(weight1 - weight2)
            sum_weight_div += layer_weight_div
            norm_weight2  = torch.norm(weight2)
            sum_model2_norm += norm_weight2
            # print(weight1)
        
        sum_weight_div /= num_layers
        sum_model2_norm /= num_layers
        weight_div = sum_weight_div / sum_model2_norm

        print(weight_div)

        return weight_div




