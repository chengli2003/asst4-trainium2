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
    TILE_OUT_C = nl.tile_size.gemm_stationary_fmax  # 128 (output channels)
    TILE_IN_C = nl.tile_size.pmax  # 128 (input channels)
    TILE_OUT_H = min(nl.tile_size.gemm_moving_fmax // out_width, out_height)
    TILE_OUT_W = out_width
    TILE_IN_H = TILE_OUT_H + filter_height - 1
    TILE_IN_W = input_width

    # Calculate number of tiles
    n_tiles_out_ch = out_channels // TILE_OUT_C
    n_tiles_in_ch = in_channels // TILE_IN_C
    n_tile_h = out_height // TILE_OUT_H
    PARTITION_DIM = 128

    # copy bias to SBUF once
    bias_sbuf = nl.ndarray((TILE_OUT_C, n_tiles_out_ch), dtype=bias.dtype, buffer=nl.sbuf)
    for tile_out_ch in nl.affine_range(n_tiles_out_ch):
        out_ch_start = tile_out_ch * TILE_OUT_C
        out_ch_end = (tile_out_ch + 1) * TILE_OUT_C
        nisa.dma_copy(src=bias[out_ch_start:out_ch_end], dst=bias_sbuf[:, tile_out_ch])

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for tile_out_ch in nl.affine_range(n_tiles_out_ch):
            out_ch_start = tile_out_ch * TILE_OUT_C
            out_ch_end = (tile_out_ch + 1) * TILE_OUT_C

            output_tile_sbuf = nl.ndarray((TILE_OUT_C, out_height, out_width), dtype=X.dtype, buffer=nl.sbuf)

            # reshape to (TILE_OUT_C, 1, 1)
            bias_tile = nl.ndarray((TILE_OUT_C, 1, 1), dtype=bias.dtype, buffer=nl.sbuf)
            nisa.dma_copy(src=bias_sbuf[:, tile_out_ch], dst=bias_tile)

            for tile_h in nl.affine_range(n_tile_h):
                
                # Allocate PSUM for this output tile (accumulate across input channels)
                output_tile_psum = nl.zeros((TILE_OUT_C, TILE_OUT_H, TILE_OUT_W), nl.float32, buffer=nl.psum)
                
                for tile_in_ch in nl.affine_range(n_tiles_in_ch):

                    input_tile = nl.ndarray((TILE_IN_C, TILE_IN_H, TILE_IN_W), dtype=X.dtype, buffer=nl.sbuf)
                    ch_start = tile_in_ch * TILE_IN_C
                    ch_end = (tile_in_ch + 1) * TILE_IN_C
                    h_in_start = tile_h * TILE_OUT_H
                    h_in_end = (tile_h + 1) * TILE_OUT_H + filter_height - 1
                    nisa.dma_copy(src=X[b, ch_start:ch_end, h_in_start: h_in_end, :], dst=input_tile)

                    # load weight tiles and transpose
                    weight_tile = nl.ndarray((TILE_OUT_C, TILE_IN_C, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(src=W[out_ch_start:out_ch_end, ch_start:ch_end, :, :], dst=weight_tile)
                    weight_tile_T_sbuf = tensor_transpose_4d(weight_tile)

                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            tile_idx = tile_in_ch * n_tiles_out_ch + tile_out_ch

                            output_tile_psum += nisa.nc_matmul(weight_tile_T_sbuf[:, :, i, j], input_tile[:, i : i + TILE_OUT_H, j : j + TILE_OUT_W])

                # add bias
                output_tile_sbuf[:, tile_h * TILE_OUT_H: (tile_h + 1) * TILE_OUT_H, :] = nisa.tensor_tensor(output_tile_psum, bias_tile, op=nl.add)

            # Write back the output tile to the output tensor
            nisa.dma_copy(src=output_tile_sbuf, dst=X_out[b, out_ch_start:out_ch_end, :, :])

    return X_out



"""
This kernel implements a 4D tensor transpose that transposes dimensions to (1, 0, 2, 3).
Input tensor shape: (d0, d1, d2, d3)
Output tensor shape: (d1, d0, d2, d3)
Output is stored in SBUF buffer.
"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def tensor_transpose_4d(input_tensor):
    d0, d1, d2, d3 = input_tensor.shape
    
    # Output tensor with transposed dimensions (1, 0, 2, 3)
    out = nl.ndarray((d1, d0, d2, d3), dtype=input_tensor.dtype, buffer=nl.sbuf)
    
    # Process each slice along dimensions 2 and 3
    for i2 in nl.affine_range(d2):
        for i3 in nl.affine_range(d3):
            # Extract 2D slice from input: shape (d0, d1)
            input_slice = nisa.tensor_copy(src=input_tensor[:, :, i2, i3])
            
            # Transpose the 2D slice using the existing matrix transpose logic
            transposed_psum = nisa.nc_transpose(input_slice)

            transposed_tile = nisa.tensor_copy(src=transposed_psum)
            
            # Store the transposed slice in the output tensor
            out[:, :, i2, i3] = nisa.tensor_copy(src=transposed_tile)
    
    return out
