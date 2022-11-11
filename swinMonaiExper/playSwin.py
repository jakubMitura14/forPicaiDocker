

import itertools
from typing import Optional, Sequence, Tuple, Type, Union
import concurrent.futures
import functools
import glob
import importlib.util
import itertools
import json
import math
import multiprocessing
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
import sys
def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res


my_swin_unetr =loadLib("my_swin_unetr", "/workspaces/forPicaiDocker/swinMonaiExper/my_swin_unetr.py")



input_image_sizeMin=(32,256,256)


my_swin_unetr.SwinUNETR(spatial_dims=3,
        in_channels=3,
        out_channels=2,
        img_size=input_image_sizeMin)