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

    # print(f"Fused Conv2D-MaxPool Tiling Info:"
    #       f"\nTILE_OUT_C: {TILE_OUT_C}, n_tiles_out_ch: {n_tiles_out_ch}"
    #       f"\nTILE_IN_C: {TILE_IN_C}, n_tiles_in_ch: {n_tiles_in_ch}"\
    #       f"\nTILE_OUT_H: {TILE_OUT_H}, TILE_OUT_W: {TILE_OUT_W}"
    #       f"\nTILE_IN_H: {TILE_IN_H}, TILE_IN_W: {TILE_IN_W}"
    #     )

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for tile_out_ch in nl.affine_range(n_tiles_out_ch):
            out_ch_start = tile_out_ch * TILE_OUT_C
            out_ch_end = (tile_out_ch + 1) * TILE_OUT_C
            for tile_h in nl.affine_range(n_tile_h):
                
                # Allocate PSUM for this output tile
                output_tile_psum = nl.zeros((TILE_OUT_C, TILE_OUT_H, TILE_OUT_W), nl.float32, buffer=nl.psum)
                
                for tile_in_ch in nl.affine_range(n_tiles_in_ch):

                    input_tile = nl.ndarray((TILE_IN_C, TILE_IN_H, TILE_IN_W), dtype=X.dtype, buffer=nl.sbuf)
                    ch_start = tile_in_ch * TILE_IN_C
                    ch_end = (tile_in_ch + 1) * TILE_IN_C
                    h_in_start = tile_h * TILE_OUT_H
                    h_in_end = (tile_h + 1) * TILE_OUT_H + filter_height - 1
                    nisa.dma_copy(src=X[b, ch_start:ch_end, h_in_start: h_in_end, :], dst=input_tile)

                    # Get weight tile to SBUF and transpose
                    # weight_tile = nl.ndarray((TILE_OUT_C, TILE_IN_C, filter_height, filter_width), dtype=W.dtype, buffer=nl.hbm)
                    # nisa.dma_copy(src=W[out_ch_start:out_ch_end, ch_start:ch_end, :, :], dst=weight_tile)
                    # weight_tile_T = matrix_transpose(weight_tile.reshape((TILE_OUT_C, TILE_IN_C * filter_height * filter_width))).reshape((TILE_IN_C, TILE_OUT_C, filter_height, filter_width))
                    # # copy back to sbuf
                    # weight_tile_T_sbuf = nl.ndarray((TILE_IN_C, TILE_OUT_C, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
                    # nisa.dma_copy(src=weight_tile_T, dst=weight_tile_T_sbuf)
        
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            weight_tile = nl.ndarray((TILE_OUT_C, TILE_IN_C), dtype=W.dtype, buffer=nl.sbuf)
                            nisa.dma_copy(src=W[out_ch_start:out_ch_end, ch_start:ch_end, i, j], dst=weight_tile)
                            weight_tile_T_psum = nisa.nc_transpose(weight_tile)
                            weight_tile_T = nisa.tensor_copy(src=weight_tile_T_psum)
                            output_tile_psum += nisa.nc_matmul(weight_tile_T, input_tile[:, i : i + TILE_OUT_H, j : j + TILE_OUT_W])

                # copy from PSUM to SBUF (after all input channels are accumulated)
                output_tile = nisa.tensor_copy(src=output_tile_psum)

                # add bias
                bias_tile = nl.ndarray((TILE_OUT_C, 1, 1), dtype=bias.dtype, buffer=nl.sbuf)
                nisa.dma_copy(src=bias[out_ch_start:out_ch_end], dst=bias_tile)
                output_tile = nisa.tensor_tensor(output_tile, bias_tile, op=nl.add)

                # Write back the output tile to the output tensor
                nisa.dma_copy(src=output_tile, dst=X_out[b, out_ch_start:out_ch_end, tile_h * TILE_OUT_H: (tile_h + 1) * TILE_OUT_H, :])

    return X_out



"""
This kernel implements a simple 2D matrix transpose.
It uses a tile-based approach along with NKI's built-in transpose kernel,
which only works on tiles of size <= 128x128.
"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def matrix_transpose(a_tensor):
    M, N = a_tensor.shape
    out = nl.ndarray((N, M), dtype=a_tensor.dtype, buffer=nl.hbm)
    tile_dim = nl.tile_size.pmax  # this should be 128

    assert M % tile_dim == N % tile_dim == 0, "Matrix dimensions not divisible by tile dimension!"

    for m in nl.affine_range(M // tile_dim):
        for n in nl.affine_range(N // tile_dim):
            # Allocate space for the input tile
            a_tile = nl.ndarray((tile_dim, tile_dim), dtype=a_tensor.dtype, buffer=nl.sbuf)
            
            # Load the tile from the input tensor
            nisa.dma_copy(src=a_tensor[m * tile_dim : (m + 1) * tile_dim, n * tile_dim : (n + 1) * tile_dim], dst=a_tile)

            # Transpose the tile (this outputs to PSUM)
            transposed_psum = nisa.nc_transpose(a_tile)
            
            # Copy from PSUM to SBUF
            transposed_tile = nisa.tensor_copy(src=transposed_psum)

            # Store the transposed tile into the output tensor
            nisa.dma_copy(src=transposed_tile, dst=out[n * tile_dim : (n + 1) * tile_dim, m * tile_dim : (m + 1) * tile_dim])

    return out