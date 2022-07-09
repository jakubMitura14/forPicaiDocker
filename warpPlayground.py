# starting point https://github.com/NVIDIA/warp/blob/main/examples/example_dem.py



# for loop gradients 
#https://github.com/NVIDIA/warp/blob/0fc9d81a8c0cb98f08009dfe0e665930756c17df/warp/tests/test_grad.py
# good example with loss function 
# https://github.com/NVIDIA/warp/blob/0fc9d81a8c0cb98f08009dfe0e665930756c17df/examples/example_sim_grad_bounce.py

import math

import numpy as np
import warp as wp
#import warp.render

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim

wp.init()


@wp.kernel
def for_loop_grad(n: int, 
                  x: wp.array(dtype=float),
                  s: wp.array(dtype=float)):

    sum = float(0.0)

    for i in range(n):
        sum = sum + x[i]*2.0

    s[0] = sum








# first we need to load image into pytorch tensor

#now we load pytorch tensor into warp

#we define simple loss function

#we minimize the loss