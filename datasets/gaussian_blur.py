"""
Taken from https://github.com/AndrewAtanov/simclr-pytorch/blob/d147c6bea1787e0d68dd334327c93e69b56f601e/utils/datautils.py#L36
which in turn is a PyTorch implementation of Gaussain Blur from the original TF repo of SimCLR 
(https://github.com/google-research/simclr/blob/244e7128004c5fd3c7805cf3135c79baa6c3bb96/data_util.py#L311)
"""

import torch
import numpy as np

class GaussianBlur(object):
    """
    Place it after ToTensor() transform
    """
    def gaussian_blur(self, image, sigma):
        image = image.unsqueeze(0)
        radius = np.int(self.kernel_size/2)
        kernel_size = radius * 2 + 1
        x = np.arange(-radius, radius + 1)

        blur_filter = np.exp(
              -np.power(x, 2.0) / (2.0 * np.power(np.float(sigma), 2.0)))
        blur_filter /= np.sum(blur_filter)

        conv1 = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), groups=3, padding=[kernel_size//2, 0], bias=False)
        conv1.weight = torch.nn.Parameter(
            torch.Tensor(np.tile(blur_filter.reshape(kernel_size, 1, 1, 1), 3).transpose([3, 2, 0, 1])))

        conv2 = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size), groups=3, padding=[0, kernel_size//2], bias=False)
        conv2.weight = torch.nn.Parameter(
            torch.Tensor(np.tile(blur_filter.reshape(kernel_size, 1, 1, 1), 3).transpose([3, 2, 1, 0])))

        res = conv2(conv1(image))
        assert res.shape == image.shape
        return res[0]

    def __init__(self, kernel_size, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            if np.random.uniform() < self.p:
                return self.gaussian_blur(img, sigma=np.random.uniform(0.2, 2))
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, p={1})'.format(self.kernel_size, self.p)
