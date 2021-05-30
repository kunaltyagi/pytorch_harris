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

    def __init__(self, shape: Tuple[int, int, int, int], dtype=torch.float32, use_abs=False):
        super(Harris, self).__init__()
        self.shape = shape
        self.dtype = dtype
        scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=self.dtype)
        scharr_y = scharr_x.transpose(0, 1)

        # tensor.shape = out_channels, in_channel, k, k
        self.scharr = torch.stack([scharr_x, scharr_y]).unsqueeze(1)
        self.use_abs = use_abs

    def forward(self, x):
        gray = tv.transforms.Grayscale()
        image = gray(x)
        # maintain size with kernel size 3
        edges = F.conv2d(image, self.scharr, padding=1)

        if self.use_abs:
            abs_edges = edges.abs()
        else:
            abs_edges = edges.pow(2).pow(0.5)

        edge = abs_edges.sum(dim=1)/2
        return edge

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
    model = Harris(shape=shape, dtype=dtype, use_abs=args.abs)
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

    parser.add_argument("--abs", action="store_true", help="Use abs instead of ((x)^2)^(1/2)")
    parser.add_argument("-o", "--output", default="out/model.onnx", help="output filename")

    args, _ = parser.parse_known_args()
    args.output = Path(args.output)
    return args


if __name__ == "__main__":
    args = parse_args()
    export(args)
