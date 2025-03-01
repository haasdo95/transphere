r"""
PyTorch implementation of a convolutional neural network on graphs based on
Chebyshev polynomials of the graph Laplacian.
See https://arxiv.org/abs/1606.09375 for details.
Copyright 2018 Michaël Defferrard.
Released under the terms of the MIT license.
DO NORMALIZED
"""
import math

import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import torch


# State-less function.
def cheb_conv(laplacian, x, weight, skip):
    """
    K:
       (1) if skip=False
            order of Cheb polynomial + 1. For K=1, no Laplacian applied.
            In general, Laplacian will be applied K-1 times.
       (2) if skip=True
            order of Cheb polynomial. For K=1, Laplacian is applied once, and so on.
            In general, Laplacian will be applied K times.
    :param laplacian: M^{-1}L or just L. sparse tensor shaped (V, V)
    :param x: shaped as (B, V, Fin)
    :param weight: shaped as (Fin, K, Fout)
    :param skip: True if want to skip zeroth order polynomial.
    """
    B, V, Fin = x.shape
    Fin, K, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features

    # transform to Chebyshev basis
    x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin*B])              # V x Fin*B
    x = x0.unsqueeze(0)                   # 1 x V x Fin*B

    # considering the case where skip=True
    if skip:
        K = K + 1

    if K > 1:
        x1 = torch.sparse.mm(laplacian, x0)     # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for _ in range(2, K):
        x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # K x V x Fin*B
        x0, x1 = x1, x2

    if skip:  # actually now x is of the shape (KERNEL+1) x V x Fin*B
        x = x[1:, :]  # strip off the initial x0. not used when skipping
        K = K - 1

    x = x.view([K, V, Fin, B])              # K x V x Fin x B
    x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    x = x.view([B*V, Fin*K])                # B*V x Fin*K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin*K, Fout)
    x = x.matmul(weight)      # B*V x Fout
    x = x.view([B, V, Fout])  # B x V x Fout

    return x


# State-full class.
class ChebConv(torch.nn.Module):
    """Graph convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Number of trainable parameters per filter, which is also the size of
        the convolutional kernel.
        The order of the Chebyshev polynomials is kernel_size - 1.

        * A kernel_size of 1 won't take the neighborhood into account.
        * A kernel_size of 2 will look up to the 1-neighborhood (1 hop away).
        * A kernel_size of 3 will look up to the 2-neighborhood (2 hops away).

        A kernel_size of 0 is equivalent to not having a graph (or an empty
        adjacency matrix). All the vertices are treated independently and form
        a set. Every element of that set is given to a fully connected layer
        with a weight matrix of size (out_channels x in_channels).
    bias : bool
        Whether to add a bias term.
    conv : callable
        Function which will perform the actual convolution.
    """

    def __init__(self, in_channels, out_channels, skip, kernel_size=3, bias=True,
                 conv=cheb_conv):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        def _conv(laplacian, inputs, weight):
            return conv(laplacian, inputs, weight, skip)
        self._conv = _conv

        # shape = (kernel_size, out_channels, in_channels)
        shape = (in_channels, kernel_size, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, activation='relu', fan='in',
                         distribution='normal'):
        r"""Reset weight and bias.

        * Kaiming / He is given by `activation='relu'`, `fan='in'` or
        `fan='out'`, and `distribution='normal'`.
        * Xavier / Glorot is given by `activation='linear'`, `fan='avg'`,
          and `distribution='uniform'`.
        * LeCun is given by `activation='linear'` and `fan='in'`.

        Motivation based on inits from PyTorch, TensorFlow, Keras.

        Parameters
        ----------
        activation : {'relu', 'linear', 'sigmoid', 'tanh'}
            Select the activation function your are using.
        fan : {'in', 'out', 'avg'}
            Select `'in'` to preserve variance in the forward pass.
            Select `'out'` to preserve variance in the backward pass.
            Select `'avg'` for a balance.
        distribution : {'normal', 'uniform'}
            Whether to draw weights from a normal or random distribution.

        References
        ----------
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on
        ImageNet Classification, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian
        Sun, https://arxiv.org/abs/1502.01852

        Understanding the difficulty of training deep feedforward neural
        networks, Xavier Glorot, Yoshua Bengio,
        http://proceedings.mlr.press/v9/glorot10a.html
        """

        if fan == 'in':
            fan = self.in_channels * self.kernel_size
        elif fan == 'out':
            fan = self.out_channels * self.kernel_size
        elif fan == 'avg':
            fan = (self.in_channels + self.out_channels) / 2 * self.kernel_size
        else:
            raise ValueError('unknown fan')

        if activation == 'relu':
            scale = 2  # relu kills half the activations, from He et al.
        elif activation in ['linear', 'sigmoid', 'tanh']:
            # sigmoid and tanh are linear around 0
            scale = 1  # from Glorot et al.
        else:
            raise ValueError('unknown activation')

        if distribution == 'normal':
            std = math.sqrt(scale / fan)
            self.weight.data.normal_(0, std)
        elif distribution == 'uniform':
            limit = math.sqrt(3 * scale / fan)
            self.weight.data.uniform_(-limit, limit)
        else:
            raise ValueError('unknown distribution')

        if self.bias is not None:
            self.bias.data.fill_(0)

    def set_parameters(self, weight, bias=None):
        r"""Set weight and bias.

        Parameters
        ----------
        weight : array of shape in_channels x kernel_size x out_channels
            The coefficients of the Chebyshev polynomials.
        bias : vector of length out_channels
            The bias.
        """
        self.weight = torch.nn.Parameter(torch.as_tensor(weight))
        if bias is not None:
            self.bias = torch.nn.Parameter(torch.as_tensor(bias))

    def extra_repr(self):
        s = '{in_channels} -> {out_channels}, kernel_size={kernel_size}'
        s += ', bias=' + str(self.bias is not None)
        return s.format(**self.__dict__)

    def forward(self, laplacian, inputs):
        r"""Forward graph convolution.

        Parameters
        ----------
        laplacian : sparse matrix of shape n_vertices x n_vertices
            Encode the graph structure.
        inputs : tensor of shape n_signals x n_vertices x n_features
            Data, i.e., features on the vertices.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs


class HealpixAvgPool(torch.nn.Module):

    def __init__(self, kernel_size):
        """kernel_size should be 4, 16, 64, etc."""
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x):
        """x has shape (batch, pixels, channels) and is in nested ordering"""
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.avg_pool1d(x, self.kernel_size)
        return x.permute(0, 2, 1)


class HealpixAvgUnpool(torch.nn.Module):

    def __init__(self, kernel_size):
        """kernel_size should be 4, 16, 64, etc."""
        super().__init__()
        self.kernel_size = kernel_size

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)

    def forward(self, x):
        """x has shape (batch, pixels, channels) and is in nested ordering"""
        # return x.repeat_interleave(self.kernel_size, dim=1)
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.interpolate(x, scale_factor=self.kernel_size, mode='nearest')
        return x.permute(0, 2, 1)
