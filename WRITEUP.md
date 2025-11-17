# Assignment 4 Implementation Write-Up

## Overview
This document describes the implementations for Assignment 4: Programming a Machine Learning Accelerator on AWS Trainium. The assignment consists of two main parts: optimizing vector operations and matrix transpose (Part 1), and implementing a fused convolution-maxpool kernel (Part 2).

## Part 1: Vector Operations and Matrix Transpose

### 1.1 Vector Addition Optimizations

#### Naive Implementation
The baseline `vector_add_naive` kernel loads entire vectors into SBUF, performs addition, and stores the result. This approach is limited to vectors of size ≤128 due to the partition size constraint.

#### Tiled Implementation (`vector_add_tiled`)
**Key Optimization**: Implemented chunking with `ROW_CHUNK = 128` to process vectors in tiles.
- **Approach**: Process vectors in chunks of 128 elements, enabling handling of arbitrarily large vectors
- **Memory Pattern**: Load tile → Compute → Store tile, repeated for each chunk
- **Benefits**: Removes size limitations while maintaining simple memory access patterns

#### Streaming Implementation (`vector_add_stream`)
**Key Optimization**: Implemented 2D tiling with `FREE_DIM = 1000` to maximize memory bandwidth utilization.
- **Approach**: Reshape vectors into (128, N/128) format and process in 2D tiles of size (128, 1000)
- **Memory Pattern**: Amortizes DMA transfer overhead by loading larger contiguous blocks
- **Benefits**: Significantly improved memory bandwidth utilization and reduced DMA overhead

### 1.2 Matrix Transpose Implementation

#### Algorithm
Implemented a tiled 2D matrix transpose using NKI's built-in `nisa.nc_transpose` instruction:

```python
def matrix_transpose(a_tensor):
    M, N = a_tensor.shape
    tile_dim = 128  # Maximum tile size supported by nc_transpose
    
    for m in range(M // tile_dim):
        for n in range(N // tile_dim):
            # Load 128x128 tile from input matrix
            # Transpose using nc_transpose (outputs to PSUM)
            # Copy from PSUM to SBUF
            # Store transposed tile to output matrix at swapped position
```

**Key Design Decisions**:
- Used maximum supported tile size (128x128) for optimal performance
- Properly handled the PSUM → SBUF copy after transpose operation
- Ensured correct indexing for transposed tile placement

## Part 2: Fused Convolution-MaxPool Implementation

### 2.1 Overall Architecture

Implemented a fused convolution kernel that combines 2D convolution with bias addition. The implementation uses a multi-level tiling strategy to efficiently manage memory hierarchy and computation.

### 2.2 Key Optimizations

#### Tiling Strategy
```python
TILE_OUT_C = 128    # Output channels (GEMM stationary dimension)
TILE_IN_C = 128     # Input channels (partition dimension)  
TILE_OUT_H = min(gemm_moving_fmax // out_width, out_height)
TILE_OUT_W = out_width
```

**Rationale**: 
- Output channel tiling matches GEMM stationary dimension (128) for optimal matrix multiplication
- Input channel tiling respects partition limits (128)
- Height tiling ensures tiles fit within GEMM moving dimension constraints

#### Bias Optimization
**Problem Identified**: Original implementation performed redundant DMA copies of bias data for each height tile.

**Solution Implemented**: 
```python
# Pre-load all bias data once at kernel start
bias_sbuf = nl.ndarray((TILE_OUT_C, n_tiles_out_ch), dtype=bias.dtype, buffer=nl.sbuf)
for tile_out_ch in nl.affine_range(n_tiles_out_ch):
    nisa.dma_copy(src=bias[...], dst=bias_sbuf[:, tile_out_ch])

# Reuse pre-loaded bias for each computation tile
bias_tile = nisa.dma_copy(src=bias_sbuf[:, tile_out_ch], dst=bias_tile_local)
```

**Benefits**: 
- Eliminated redundant bias DMA operations
- Reduced memory bandwidth pressure
- Improved overall kernel performance

### 2.3 4D Tensor Transpose Implementation

Implemented `tensor_transpose_4d` to transpose weight tensors from (out_ch, in_ch, h, w) to (in_ch, out_ch, h, w):

```python
def tensor_transpose_4d(input_tensor):
    d0, d1, d2, d3 = input_tensor.shape
    out = nl.ndarray((d1, d0, d2, d3), dtype=input_tensor.dtype, buffer=nl.sbuf)
    
    for i2 in nl.affine_range(d2):
        for i3 in nl.affine_range(d3):
            # Extract 2D slice (d0, d1)
            input_slice = input_tensor[:, :, i2, i3]
            # Transpose using nc_transpose
            transposed_psum = nisa.nc_transpose(input_slice)
            # Copy and store result
            out[:, :, i2, i3] = nisa.tensor_copy(src=transposed_psum)
```

**Design Rationale**:
- Leverages existing 2D transpose hardware (`nc_transpose`)
- Processes each spatial location (h, w) independently
- Maintains data locality within SBUF throughout the operation

### 2.4 Convolution Computation Flow

The main computation follows this optimized flow:

1. **Outer Loops**: Batch → Output Channel Tiles → Height Tiles
2. **Inner Loops**: Input Channel Tiles → Filter Height → Filter Width
3. **Core Operations**:
   - Load input activation tile with appropriate padding
   - Load and transpose weight tile using `tensor_transpose_4d`
   - Perform matrix multiplication using `nisa.nc_matmul` (accumulates in PSUM)
   - Add bias using optimized pre-loaded bias data
   - Store result tile back to HBM

## Performance Considerations

### Memory Access Patterns
- **Coalesced Access**: Ensured contiguous memory access patterns where possible
- **Temporal Locality**: Reused loaded data (especially bias) across multiple computations  
- **Spatial Locality**: Organized tiling to maximize cache hit rates

### Compute Engine Utilization
- **GEMM Engine**: Used for core convolution matrix multiplications
- **Transpose Engine**: Leveraged hardware transpose for weight reordering
- **Vector Engine**: Used for bias addition and tensor operations

### Memory Hierarchy Optimization
- **SBUF Management**: Carefully managed SBUF allocation to avoid conflicts
- **PSUM Accumulation**: Used PSUM buffer for efficient accumulation across input channels
- **DMA Optimization**: Minimized DMA operations through strategic data reuse

## Conclusion

The implementations demonstrate effective utilization of Trainium's memory hierarchy and compute engines. Key optimizations include strategic tiling for memory locality, elimination of redundant data movement, and proper utilization of specialized hardware instructions. The fused convolution kernel particularly benefits from the bias optimization and 4D transpose implementation, showing how careful attention to memory access patterns can significantly impact accelerator performance.