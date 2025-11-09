import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    X_re = X.reshape((batch_size, in_channels, input_height * input_width))
    X_out = X_out.reshape((batch_size, out_channels, out_pool_height * out_pool_width))

    # Process the images in batches
    for b in nl.affine_range(batch_size):

        X_single = X_re[b]

        # Initialize convolution output
        conv_output = nl.zeros((out_channels, out_height * out_width), dtype=X_single.dtype, buffer=nl.sbuf)
        
        # Process convolution following the algorithm pseudocode
        for i in nl.affine_range(filter_height):
            for j in nl.affine_range(filter_width):
                
                # Create shifted input tensor in HBM
                input_shifted = nl.ndarray((in_channels, out_height * out_width), dtype=X_single.dtype, buffer=nl.hbm)

                # Populate the shifted input tensor
                # For each output position (out_h, out_w), we take input at (out_h + i, out_w + j)
                for out_h in nl.affine_range(out_height):
                    for out_w in nl.affine_range(out_width):
                        # Input position after applying shift (i, j)
                        in_h = out_h + i
                        in_w = out_w + j
                        in_idx = in_h * input_width + in_w
                        out_idx = out_h * out_width + out_w
                        
                        # Copy all channels for this spatial position
                        nisa.dma_copy(src=X_single[:, in_idx:in_idx+1], 
                                  dst=input_shifted[:, out_idx:out_idx+1])
                
                # Get the weight slice for position (i, j)
                weight_slice = W[:, :, i, j]  # Shape: (out_channels, in_channels)
                weight_slice_T = nisa.dma_transpose(weight_slice)  # Shape: (in_channels, out_channels)
                
                temp_result = nl.ndarray((out_channels, out_height * out_width), dtype=X_single.dtype, buffer=nl.sbuf)
                
                print("weight_slice_T shape:", weight_slice_T.shape)
                print("input_shifted shape:", input_shifted.shape)


                # Perform tiled matrix multiplication
                nki_matmul_tiled_(weight_slice_T, input_shifted, temp_result)
                
                # Add to accumulated convolution output
                conv_output += temp_result

        # copy over to X_out
        # nl.device_print("value of conv_output:", conv_output)
        nisa.dma_copy(src=conv_output, dst=X_out[b])
    X_out = X_out.reshape((batch_size, out_channels, out_pool_height, out_pool_width))

    return X_out

