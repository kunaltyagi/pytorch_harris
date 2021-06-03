#! /usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path
import torch
from torch import nn
import torchvision as tv
from torch.nn import functional as F
from typing import Optional, Tuple

def get_gaussian_2d(size, mean, stddev):
    grid1d = np.linspace(-1, 1, size)
    x, y = np.meshgrid(grid1d, grid1d)
    grid2d = np.sqrt(x*x + y*y)

    gauss = np.exp(-(grid2d - mean)**2 / (2 * stddev**2))
    return gauss


class CornerDetection(nn.Module):
    """
    Converts a batch of RGB images into grayscale + blur + edges + Corner detection
    1, C, H, W
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        corner_window: int = 3,
        nms_window: int = 3,
        blur_window: Optional[int] = None,
        gradient_tensor: Optional[torch.Tensor] = None,
    ):
        """
        gradient_tensor: tensor in [2, 1, K, K] format, of type dtype
            If None, a tensor of shape [2, 1, 3, 3] is used by default

        corner_window: Size of window to consider a neighborhood for corner detection
        nms_window: Size of window to perform Non-Maximal Supression

        blur_window:
        """
        super(CornerDetection, self).__init__()

        self.dtype = dtype

        assert corner_window % 2, "Corner window size needs to be odd to preserve size"
        if blur_window:
            assert blur_window % 2, "Blur window size needs to be odd to preserve size"

        self.corner_window_size = corner_window
        self.nms_window_size = nms_window
        self.blur_window_size = blur_window

        if gradient_tensor is not None:
            assert gradient_tensor.shape[0] == 2, "Only 2-axis gradients are supported"
            assert (
                gradient_tensor.shape[1] == 1
            ), "Gradient needs to be batch independent"
            assert gradient_tensor.shape[2] % 2, "Gradient window size needs to be odd"
            width, height = gradient_tensor.shape[-2:]
            assert width == height, "Gradient window size needs to be square"

            assert (
                gradient_tensor.dtype == dtype
            ), "Everything needs to be of the same dtype"

            self.scharr = gradient_tensor
        else:
            scharr_x = torch.tensor(
                [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=self.dtype
            )
            scharr_y = scharr_x.transpose(0, 1)

            # tensor.shape = out_channels, in_channel, k, k
            self.scharr = torch.stack([scharr_x, scharr_y]).unsqueeze(1)

    def forward(self, x):
        gray = tv.transforms.Grayscale()
        image = gray(x)

        edges = F.conv2d(image, self.scharr, padding=int(self.scharr.shape[-1] // 2))

        # Shi-Tomasi
        # Ix*Ix and Iy*Iy
        edges_sqr = edges.multiply(edges)
        # Ix*Iy
        # edges_xy = edges.prod(dim=1)  # Doesn't work with OpenVINO
        edges_xy = edges[:, 0, ::] * edges[:, 1, ::]
        # Put them together
        edges_prod = torch.cat([edges_sqr.squeeze(0), edges_xy]).unsqueeze(0)
        # In total 3 layers
        conv_dims = 3
        # sum all elements in a window
        M = F.conv2d(
            edges_prod,
            padding=self.corner_window_size // 2,
            groups=conv_dims,
            weight=torch.ones(
                conv_dims, 1, self.corner_window_size, self.corner_window_size
            ),
        )
        # Now we have:
        # 0: \sum_{window} IxIx
        # 1: \sum_{window} IyIy
        # 2: \sum_{window} IxIy

        # min eigen_value = trace/2 - sqrt((trace/2)**2 - determinant)
        # trace = m[0]+m[1]
        # determinant = m[0]m[1] - m[2]**2
        # => eigen_value = (m[0]+m[1])/2 - sqrt(((m[0]-m[1])/2)**2+m[2]**2)
        IxIx = M[:, 0, ::]
        IyIy = M[:, 1, ::]
        IxIy = M[:, 2, ::]
        IxIx_IyIy = IxIx - IyIy
        trace = IxIx + IyIy

        eig_val = (
            trace - (IxIx_IyIy.multiply_(IxIx_IyIy) + 4 * IxIy.multiply_(IxIy)).sqrt()
        )
        # apply non-max suppression
        local_maxima = (
            F.max_pool2d(eig_val, self.nms_window_size, stride=1, padding=1) == eig_val
        )
        result = local_maxima * eig_val

        # needs to be (edges**2)**(0.5) because edges.abs() doesn't work with OpenVINO
        abs_edges = edges_sqr.sqrt_()

        # used as a short-cut for abs_edges.sum(dim=1).sqrt()
        edge = abs_edges.sum(dim=1) / 2
        return edge, eig_val


def export_onnx(args):
    """
    Exports the model to an ONNX file.
    """
    # Define the expected input shape (dummy input)
    shape = (1, 3, 300, 300)
    # Create the Model
    dtype = torch.float32
    model = CornerDetection(dtype=dtype)
    X = torch.ones(shape, dtype=dtype)
    output_file = args.output
    print(f"Writing to {output_file}")
    torch.onnx.export(
        model,
        X,
        f"{output_file.as_posix()}",
        opset_version=12,
        do_constant_folding=True,
    )


def export(args):
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(args)
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--output", default="out/model.onnx", help="output filename"
    )

    args, _ = parser.parse_known_args()
    args.output = Path(args.output)
    return args


if __name__ == "__main__":
    args = parse_args()
    export(args)
