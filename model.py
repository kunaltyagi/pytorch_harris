#! /usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path
import torch
from torch import nn
import torchvision as tv
from torch.nn import functional as F
from typing import Tuple


class Harris(nn.Module):
    """
    Converts a batch of RGB -> Gray Scale
    1, C, H, W
    """

    def __init__(self, shape: Tuple[int, int, int, int], dtype=torch.float32):
        super(Harris, self).__init__()
        self.shape = shape
        self.dtype = dtype
        scharr_x = torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=self.dtype
        )
        scharr_y = scharr_x.transpose(0, 1)

        # tensor.shape = out_channels, in_channel, k, k
        self.scharr = torch.stack([scharr_x, scharr_y]).unsqueeze(1)
        self.window_size = 3  # needs to be odd

    def forward(self, x):
        gray = tv.transforms.Grayscale()
        image = gray(x)
        # maintain size with kernel size 3
        edges = F.conv2d(image, self.scharr, padding=1)

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
            padding=self.window_size // 2,
            groups=conv_dims,
            weight=torch.ones(conv_dims, 1, self.window_size, self.window_size),
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
        # apply non-max suppression, doesn't work on OpenVINO
        local_maxima = F.max_pool2d(eig_val, 3, stride=1, padding=1) == eig_val
        result = local_maxima * eig_val

        abs_edges = edges_sqr.sqrt_()
        edge = abs_edges.sum(dim=1) / 2
        return edge, eig_val

        thresh = torch.where(edge > 140, 255, 0)

        return thresh


def export_onnx(args):
    """
    Exports the model to an ONNX file.
    """
    # Define the expected input shape (dummy input)
    shape = (1, 3, 300, 300)
    # Create the Model
    dtype = torch.float32
    model = Harris(shape=shape, dtype=dtype)
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
