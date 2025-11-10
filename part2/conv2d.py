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
                
                # Create input_shifted with dimensions (in_channels, in_height * in_width)
                # This represents the relevant input positions for this filter position (i, j)
                input_shifted = nl.zeros((in_channels, input_height * input_width), dtype=X_single.dtype, buffer=nl.sbuf)
                
                # For each valid output position, extract the corresponding input position
                for out_h in nl.affine_range(out_height):
                    for out_w in nl.affine_range(out_width):
                        # The input position for this output position with filter offset (i, j)
                        in_h = out_h + i
                        in_w = out_w + j
                        in_idx = in_h * input_width + in_w
                        out_idx = out_h * out_width + out_w
                        
                        # Copy all channels for this spatial position
                        nisa.dma_copy(src=X_single[:, in_idx:in_idx+1], 
                                      dst=input_shifted[:, out_idx:out_idx+1])
                
                # Get the weight slice for position (i, j)
                weight_slice = W[:, :, i, j]  # Shape: (out_channels, in_channels)
                weight_slice_T = matrix_transpose(weight_slice)  # Shape: (in_channels, out_channels)
                
                temp_result = nl.ndarray((out_channels, input_height * input_width), dtype=X_single.dtype, buffer=nl.sbuf)
                
                # Perform tiled matrix multiplication: weight_slice_T @ input_shifted
                nki_matmul_tiled_(weight_slice_T, input_shifted, temp_result)

                # nl.device_print("value of temp_result:", temp_result)
                
                # Extract the valid output region from temp_result and add to conv_output
                for out_h in nl.affine_range(out_height):
                    for out_w in nl.affine_range(out_width):
                        out_idx = out_h * out_width + out_w
                        
                        # Add the result for this spatial position to conv_output
                        for c in nl.affine_range(out_channels):
                            conv_output[c, out_idx] += temp_result[c, out_idx]

        # copy over to X_out
        # nl.device_print("value of conv_output:", conv_output)
        nisa.dma_copy(src=conv_output, dst=X_out[b])
    X_out = X_out.reshape((batch_size, out_channels, out_pool_height, out_pool_width))

    return X_out



# # Helper function to process a single image
# """
# X_single: shape (in_channels, input_height * input_width)
# output_single: shape (out_channels, out_pool_height * out_pool_width)
# """
# @nki.compiler.skip_middle_end_transformations
# @nki.jit 
# def fused_conv2d_maxpool_single_image(X_single, W, bias, output_single, pool_size, input_height, input_width):
#     """Process a single image through convolution and maxpool"""
    
#     in_channels, _ = X_single.shape
#     out_channels, _, filter_height, filter_width = W.shape
    
#     out_height = input_height - filter_height + 1
#     out_width = input_width - filter_width + 1
    
#     out_pool_height = out_height // pool_size
#     out_pool_width = out_width // pool_size

#     # Initialize convolution output
#     conv_output = nl.ndarray(output_single.shape, dtype=X_single.dtype, buffer=nl.sbuf)
#     for c in nl.affine_range(out_channels):
#         for i in nl.affine_range(out_height * out_width):
#             conv_output[c, i] = 0.0
    
#     # Process convolution following the algorithm pseudocode
#     for i in nl.affine_range(filter_height):
#         for j in nl.affine_range(filter_width):
            
#             # Create shifted input tensor in HBM
#             input_shifted = nl.ndarray((in_channels, out_height * out_width), dtype=X_single.dtype, buffer=nl.hbm)

#             # Populate the shifted input tensor
#             # For each output position (out_h, out_w), we take input at (out_h + i, out_w + j)
#             for out_h in nl.affine_range(out_height):
#                 for out_w in nl.affine_range(out_width):
#                     # Input position after applying shift (i, j)
#                     in_h = out_h + i
#                     in_w = out_w + j
#                     in_idx = in_h * input_width + in_w
#                     out_idx = out_h * out_width + out_w
                    
#                     # Copy all channels for this spatial position
#                     nisa.dma_copy(src=X_single[:, in_idx:in_idx+1], 
#                                   dst=input_shifted[:, out_idx:out_idx+1])
            
#             # Get the weight slice for position (i, j)
#             weight_slice = W[:, :, i, j]  # Shape: (out_channels, in_channels)
#             weight_slice_T = nisa.dma_transpose(weight_slice)  # Shape: (in_channels, out_channels)
            
#             temp_result = nl.ndarray((out_channels, out_height * out_width), dtype=X_single.dtype, buffer=nl.sbuf)
            
#             # Perform tiled matrix multiplication
#             nki_matmul_tiled_(weight_slice_T, input_shifted, temp_result)
            
#             # Add to accumulated convolution output
#             # Since we're accumulating across loop indices i,j, we need to be careful with dependencies
#             # Use element-wise addition directly to HBM to avoid dependency issues
#             for c in nl.affine_range(out_channels):
#                 for spatial_idx in nl.affine_range(out_height * out_width):
#                     conv_output[c, spatial_idx] += temp_result[c, spatial_idx]

    # copy over to output_single
    # nisa.dma_copy(src=conv_output, dst=output_single)

    
    # # Add bias to convolution result
    # for c in nl.affine_range(out_channels):
    #     for spatial_idx in nl.affine_range(out_height * out_width):
    #         conv_output[c, spatial_idx] += bias[c]
    
    # # Apply maxpooling
    # if pool_size == 1:
    #     # No pooling, copy convolution output directly to final output
    #     # Convert flattened conv_output to 3D layout for output_single
    #     for c in nl.affine_range(out_channels):
    #         for h in nl.affine_range(out_height):
    #             for w in nl.affine_range(out_width):
    #                 spatial_idx = h * out_width + w
    #                 output_single[c, h, w] = conv_output[c, spatial_idx]
    # else:
    #     # Apply 2x2 max pooling
    #     # Process each output channel and pooled position
    #     for c in nl.affine_range(out_channels):
    #         for ph in nl.affine_range(out_pool_height):
    #             for pw in nl.affine_range(out_pool_width):
    #                 # Get 2x2 window starting positions
    #                 h_start = ph * 2
    #                 w_start = pw * 2
                    
    #                 # Extract the 2x2 window values from flattened conv_output
    #                 idx_00 = h_start * out_width + w_start
    #                 idx_01 = h_start * out_width + (w_start + 1)
    #                 idx_10 = (h_start + 1) * out_width + w_start
    #                 idx_11 = (h_start + 1) * out_width + (w_start + 1)
                    
    #                 val_00 = conv_output[c, idx_00]
    #                 val_01 = conv_output[c, idx_01]
    #                 val_10 = conv_output[c, idx_10]
    #                 val_11 = conv_output[c, idx_11]
                    
    #                 # Find maximum manually since tensor operations might not work for scalars
    #                 max_val = val_00
    #                 if val_01 > max_val:
    #                     max_val = val_01
    #                 if val_10 > max_val:
    #                     max_val = val_10
    #                 if val_11 > max_val:
    #                     max_val = val_11
                    
    #                 output_single[c, ph, pw] = max_val


@nki.compiler.skip_middle_end_transformations
@nki.jit
def nki_matmul_tiled_(lhsT, rhs, result):
    """NKI kernel to compute a matrix multiplication operation in a tiled manner"""

    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"

    # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

    # Maximum partition dimension of a tile
    TILE_K = nl.tile_size.pmax  # 128

    # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Use affine_range to loop over tiles
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # Allocate a tensor in PSUM
            res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                # Declare the tiles on SBUF
                lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                # Load tiles from lhsT and rhs
                nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
                nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])

                # Accumulate partial-sums into PSUM
                res_psum += nisa.nc_matmul(lhsT_tile, rhs_tile)

            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
            res_sb = nl.copy(res_psum, dtype=result.dtype)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)

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

    # TODO: Your implementation here. The only compute instruction you should use is `nisa.nc_transpose`.
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