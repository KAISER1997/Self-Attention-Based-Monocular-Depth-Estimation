"""
Utilities for Unsupervised depth by

@copyright This program is the confidential and proprietary product of Valeo Driving Assistance Systems
Research (DAR). Any unauthorised use, reproduction or transfer of this program is strictly prohibited.
(Subject to limited distribution and restricted disclosure only.) All rights reserved.

@author: Varun Ravi Kumar <varun-ravi.kumar@valeo.com>

@date: Nov 2018

@version: 3.6 or higher

@requirements: Numpy, Torch
"""

import collections

import numpy as np
import torch

# from monodepth_model import Resnet18_md, Resnet50_md, ResnetModel


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def get_model(model, input_channels=3, pretrained=False):
    if model == 'resnet50_md':
        out_model = Resnet50_md(input_channels)
    elif model == 'resnet18_md':
        out_model = Resnet18_md(input_channels)
    else:
        out_model = ResnetModel(input_channels, encoder=model, pretrained=pretrained)
    return out_model


def tile(input_tensor: torch.Tensor, dim: int, n_tile: int, device: str) -> torch.Tensor:
    """Tiling function implemented in pyTorch
    :param input_tensor: tensor
    :param dim: dimension to tile along
    :param n_tile: number of tiles
    :param device: cpu or gpu
    :return torch.Tensor: the tiled tensor
    """
    init_dim = input_tensor.size(dim)
    repeat_idx = [1] * input_tensor.dim()
    repeat_idx[dim] = n_tile
    input_tensor = input_tensor.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        device)
    return torch.index_select(input_tensor, dim, order_index)


def post_process_disparity(disp):
    """Disparity post processing
    :param disp: 2 channel disparity map
    :return: disparity -> processed consistent disparity
    """
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by 2 every 10 epochs after 30 epochs"""
    if 30 <= epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_tensor(numpy_array: np.array, device: str) -> torch.Tensor:
    return torch.Tensor(numpy_array).to(device)
