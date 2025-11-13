# 3D Heat Diffusion Solver - Complete HPC Implementation

## Overview

A comprehensive implementation of a 3D heat diffusion solver using the **Jacobi explicit stencil** (7-point) with **Dirichlet boundary conditions**. Six parallelization strategies are provided, suitable for single-node to distributed computing environments.

### Problem Formulation

**Discrete Jacobi Equation (Explicit Time-Stepping):**

$$u^{n+1}_{i,j,k} = \alpha \cdot u^n_{i,j,k} + \beta \sum_{\text{neighbors}} u^n$$

Where:
- $\alpha$ = center coefficient (typically 0.0)
- $\beta$ = neighbor coefficient (typically 1/6 for Laplacian discretization)
- Interior points: $u^{n+1}_{i,j,k} = \alpha u^n_{i,j,k} + \beta(u^n_{i-1,j,k} + u^n_{i+1,j,k} + u^n_{i,j-1,k} + u^n_{i,j+1,k} + u^n_{i,j,k-1} + u^n_{i,j,k+1})$
- Boundary conditions: Fixed Dirichlet (1.0 on all borders)
- Initial condition: Boundaries = 1.0, Interior = 0.0

---

## Implementation Variants

### 1. **OpenMP** (Shared-Memory, Single Node)

**File:** `src/heat3d_openmp.c`

**Key Features:**
- Triple nested loop with `#pragma omp parallel for collapse(3)`
- Static scheduling for cache locality
- Double-buffered arrays (`u` and `un`) swapped each iteration
- Boundary condition enforcement in separate parallelized loop

**Compilation:**
```bash
gcc -O3 -march=native -fopenmp -I./include src/heat3d_openmp.c -o bin/heat_openmp -lm
```

**Execution:**
```bash
# 8 threads
OMP_NUM_THREADS=8 ./bin/heat_openmp 256 256 256 100

# Small test
OMP_NUM_THREADS=4 ./bin/heat_openmp 32 32 32 10

# Strong scaling test (same problem, increasing threads)
for t in 1 2 4 8; do OMP_NUM_THREADS=$t ./bin/heat_openmp 256 256 256 100; done
```

**Performance Notes:**
- Weak scaling limited by memory bandwidth (memory-bound kernel)
- Cache blocking / tiling could improve NUMA behavior on large grids
- `collapse(3)` essential for load balancing on 3D grids
- False sharing risk on boundary copy; mitigated by coarse granularity

**Complexity:**
- Time: $O(T \cdot Nx \cdot Ny \cdot Nz)$ (7 FLOPs per interior cell)
- Space: $O(Nx \cdot Ny \cdot Nz)$ (2 buffers)
- Memory bandwidth: ~7 doubles read + 1 double written per interior cell = ~56 bytes/cell

---

### 2. **pthreads** (Shared-Memory, Explicit Synchronization)

**File:** `src/heat3d_pthreads.c`

**Key Features:**
- Domain decomposition along Z-axis (slabs)
- `pthread_barrier_t` for synchronization at each time step
- Master thread (thread 0) performs buffer swapping
- Fine-grained control over thread affinity (can pin to NUMA domains)

**Compilation:**
```bash
gcc -O3 -march=native -pthread -I./include src/heat3d_pthreads.c -o bin/heat_pthreads -lm
```

**Execution:**
```bash
# 8 threads
./bin/heat_pthreads 256 256 256 100 8

# With thread pinning (Linux)
taskset -c 0-7 ./bin/heat_pthreads 256 256 256 100 8

# Small test
./bin/heat_pthreads 32 32 32 10 4
```

**Performance Notes:**
- Lower-level control than OpenMP; allows NUMA-aware pinning
- Barrier overhead can be significant for many threads
- Z-slab decomposition has good cache locality (contiguous k-layers)
- False sharing possible on barrier synchronization; consider padding

**Pitfalls:**
- Incorrect barrier handling → deadlock
- Buffer swap visibility → must be done after barrier, not concurrently
- NUMA imbalance if threads not pinned to local memory

---

### 3. **MPI** (Distributed-Memory, Domain Decomposition)

**File:** `src/heat3d_mpi.c`

**Key Features:**
- Z-axis domain decomposition (rank i handles slices $k \in [k_i, k_{i+1})$)
- Non-blocking `MPI_Isend/MPI_Irecv` for halo exchange
- Overlapping communication and computation:
  1. Post receives/sends for k=1 and k=Nz_local-2 layers
  2. Compute interior k ∈ [2, Nz_local-2) (no halo dependency)
  3. Wait for halos
  4. Compute boundary slices using received halo data
- Global boundaries handled by rank-0 and rank-(nprocs-1)
- `MPI_Reduce` to sum checksums

**Compilation:**
```bash
mpicc -O3 -march=native -I./include src/heat3d_mpi.c -o bin/heat_mpi -lm
```

**Execution:**
```bash
# 4 processes
mpirun -np 4 ./bin/heat_mpi 256 256 256 100

# With binding to sockets (OpenMPI)
mpirun -np 4 --bind-to socket ./bin/heat_mpi 512 512 512 50

# Small test
mpirun -np 2 ./bin/heat_mpi 32 32 32 10

# Weak scaling (fixed problem per rank)
mpirun -np 2 ./bin/heat_mpi 256 256 128 100
mpirun -np 4 ./bin/heat_mpi 256 256 256 100
```

**Performance Notes:**
- Communication cost dominates for small Nz_local or small grids
- Optimal performance requires overlap of computation and communication
- Halo layer size: Ny × Nx doubles = typical sweet spot for bandwidth
- Strong scaling limited by communication latency and synchronization

**Halo Exchange Details:**
```
Each rank exchanges k=1 layer (receive from prev) and k=Nz_local-2 layer (send to next)
Receives go to unused buffers; computation uses these after waitall.
```

**Pitfalls:**
- MPI message ordering: must use explicit tags or MPI_Sendrecv to avoid deadlock
- Halo buffer allocation on every iteration (overhead); better to allocate once and reuse
- Global boundary checks needed: rank 0 enforces k=0 BC; rank nprocs-1 enforces k=Nz-1 BC

---

### 4. **CUDA Naive** (GPU, Global Memory)

**File:** `src/heat3d_cuda_naive.cu`

**Key Features:**
- One thread per interior cell (linearized 1D grid for flexibility)
- Global memory reads; no shared memory optimization
- Boundary enforcement in separate kernel
- CUDA events for timing

**Compilation:**
```bash
nvcc -O3 -arch=sm_70 -I./include src/heat3d_cuda_naive.cu -o bin/heat_cuda_naive -lm
```

**Execution:**
```bash
# Basic
./bin/heat_cuda_naive 256 256 256 100

# Larger problem
./bin/heat_cuda_naive 512 512 512 100

# Small test
./bin/heat_cuda_naive 32 32 32 10
```

**Performance Characteristics:**
- Memory-bound: ~7 reads + 1 write per interior cell
- Global memory throughput ~40-50% of peak (due to coalescing issues)
- Estimated GFLOPS: 1-3 GFLOP/s on modern GPU (depending on kernel efficiency)

**Bottlenecks:**
- Coalescing violations: consecutive threads not accessing consecutive memory
- L2 cache misses: Nz-dependent accesses span large strides
- Occupancy may be limited by register pressure

---

### 5. **CUDA Optimized** (GPU, Shared Memory Tiling)

**File:** `src/heat3d_cuda_optimized.cu`

**Key Features:**
- 2.5D tiling: (TILE_X × TILE_Y) interior tiles in x-y plane
- Shared memory halos: each block loads (TILE_X+2) × (TILE_Y+2) neighborhood
- Loop over z-layers: each k-layer processed by thread grid
- Bank conflict mitigation via careful memory layout (fast-varying x)
- Shared memory size per block: ~1.2 KB for 8×8 tiles

**Tuning Parameters:**
```c
#define TILE_X 8
#define TILE_Y 8
/* Adjust to fit in shared memory (~48-96 KB available per SM) */
```

**Compilation:**
```bash
nvcc -O3 -arch=sm_70 -I./include src/heat3d_cuda_optimized.cu -o bin/heat_cuda_opt -lm
```

**Execution:**
```bash
./bin/heat_cuda_opt 256 256 256 100

# Profile with nvprof (older CUDA)
nvprof ./bin/heat_cuda_opt 128 128 128 10

# Profile with nsys (newer CUDA)
nsys profile -o heat_profile ./bin/heat_cuda_opt 128 128 128 10
```

**Performance Characteristics:**
- Compute-bound: reduced global memory pressure
- Expected throughput: 60-85% of peak memory bandwidth
- Estimated GFLOPS: 5-15 GFLOP/s (faster than naive by 5-10×)

**Optimization Details:**
1. **Coalescing:** Contiguous threads load contiguous x-coordinates → full bandwidth
2. **Bank Conflicts:** Fast-varying x dimension avoids 32-way bank conflicts
3. **Reuse:** Each cell in interior participates in multiple stencils within tile
4. **Occupancy:** Small tiles allow high occupancy (limited shared mem usage)

**Tile Size Trade-offs:**
| Tile Size | Shared Mem/Block | Arithmetic Intensity | Notes |
|-----------|-------------------|----------------------|-------|
| 4×4 | 0.3 KB | Low | High occupancy, memory-bound |
| 8×8 | 1.2 KB | Medium | Sweet spot for most GPUs |
| 12×12 | 2.0 KB | High | Larger footprint, fewer concurrent blocks |
| 16×16 | 3.2 KB | Higher | Best for compute-bound GPUs |

---

### 6. **Hybrid MPI + OpenMP** (Distributed + Shared Memory)

**File:** `src/heat3d_hybrid.c`

**Key Features:**
- MPI: domain decomposition along Z-axis (inter-node / socket-level)
- OpenMP: parallelizes loops within each MPI rank (intra-node)
- Thread-safe MPI: `MPI_THREAD_FUNNELED` (only master thread calls MPI)
- Phases:
  1. Master posts non-blocking receives/sends
  2. OpenMP teams compute interior slices
  3. Master waits for halos
  4. OpenMP teams compute boundary slices
  5. Buffer swap

**Compilation:**
```bash
mpicc -O3 -march=native -fopenmp -I./include src/heat3d_hybrid.c -o bin/heat_hybrid -lm
```

**Execution:**
```bash
# 2 MPI ranks, 8 OpenMP threads per rank
OMP_NUM_THREADS=8 mpirun -np 2 ./bin/heat_hybrid 256 256 256 100

# Optimal for 2-socket system: 2 ranks (one per socket), 8 threads each
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores mpirun -np 2 --bind-to socket ./bin/heat_hybrid 512 512 512 50

# Weak scaling example (256×256×128 per rank on 2 sockets)
OMP_NUM_THREADS=8 mpirun -np 2 ./bin/heat_hybrid 256 256 256 100
OMP_NUM_THREADS=8 mpirun -np 4 ./bin/heat_hybrid 256 256 512 100
```

**Performance Notes:**
- Scalability: typically better than pure MPI or pure OpenMP on multi-socket systems
- Optimal rank count = number of NUMA domains / sockets (e.g., 2 or 4)
- Thread count = total cores per socket / ranks per socket
- Communication volume reduced (fewer MPI ranks), but more concurrent computation

**Pitfalls:**
- Thread oversubscription → performance degradation
- NUMA-unaware thread pinning → cross-socket memory access
- MPI_THREAD_FUNNELED bottleneck if ranks communicate frequently

---

## Compilation & Build

### Prerequisites

**CPU versions:**
- GCC 7.0+ or Clang 5.0+
- OpenMP-capable compiler
- POSIX threads (pthreads)
- MPI library (OpenMPI 2.0+, MPICH 3.0+, Intel MPI, etc.)

**GPU versions:**
- NVIDIA CUDA Toolkit 11.0+
- NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer)

### Build All

```bash
cd /path/to/HPC_3d_heat_diffusion

# CPU only
make cpu

# GPU only (requires NVIDIA CUDA)
make cuda

# All variants
make all

# Clean
make clean
```

### Compile Individual Versions

```bash
# OpenMP
gcc -O3 -march=native -fopenmp -I./include src/heat3d_openmp.c -o bin/heat_openmp -lm

# pthreads
gcc -O3 -march=native -pthread -I./include src/heat3d_pthreads.c -o bin/heat_pthreads -lm

# MPI
mpicc -O3 -march=native -I./include src/heat3d_mpi.c -o bin/heat_mpi -lm

# Hybrid
mpicc -O3 -march=native -fopenmp -I./include src/heat3d_hybrid.c -o bin/heat_hybrid -lm

# CUDA naive
nvcc -O3 -arch=sm_70 -I./include src/heat3d_cuda_naive.cu -o bin/heat_cuda_naive -lm

# CUDA optimized
nvcc -O3 -arch=sm_70 -I./include src/heat3d_cuda_optimized.cu -o bin/heat_cuda_opt -lm
```

---

## Validation & Testing

### Small Test Cases

All implementations should produce **identical checksums** for the same problem parameters.

```bash
#!/bin/bash
# Run all implementations on same problem
Nx=32; Ny=32; Nz=32; T=10

echo "=== Validation: 32^3, T=10 ==="

OMP_NUM_THREADS=4 ./bin/heat_openmp $Nx $Ny $Nz $T
./bin/heat_pthreads $Nx $Ny $Nz $T 4
mpirun -np 1 ./bin/heat_mpi $Nx $Ny $Nz $T
OMP_NUM_THREADS=4 mpirun -np 1 ./bin/heat_hybrid $Nx $Ny $Nz $T

# GPU (if available)
./bin/heat_cuda_naive $Nx $Ny $Nz $T
./bin/heat_cuda_opt $Nx $Ny $Nz $T
```

**Expected Output (example):**
```
[OpenMP] checksum=1.792000e+03 time=0.000234 s
[pthreads] checksum=1.792000e+03 time=0.000287 s
[MPI] checksum=1.792000e+03 time=0.002145 s
[Hybrid] checksum=1.792000e+03 time=0.001823 s
[CUDA naive] checksum=1.792000e+03 time=0.000456 s
[CUDA optimized] checksum=1.792000e+03 time=0.000123 s
```

All checksums should match to floating-point precision (≤ 1e-10 relative error).

### Correctness Verification

```bash
# Test on very small grid (serial computation)
./bin/heat_openmp 3 3 3 1   # Single interior cell
./bin/heat_openmp 5 5 5 1   # 27 interior cells

# Manual verification:
# For alpha=0, beta=1/6:
# u_new[2,2,2] = (1/6) * (6 neighbors) = (1/6) * 6 = 1.0 (all neighbors at boundary = 1.0)
# Interior should converge to ~1.0 after sufficient time steps
```

---

## Performance Benchmarking

### Strong Scaling (Fixed Problem, Varying Parallel Resources)

Measure speedup by increasing parallelism on fixed problem size.

```bash
#!/bin/bash
# OpenMP strong scaling
Nx=256; Ny=256; Nz=256; T=100
echo "=== OpenMP Strong Scaling ==="
for t in 1 2 4 8; do
    echo "Threads: $t"
    OMP_NUM_THREADS=$t ./bin/heat_openmp $Nx $Ny $Nz $T
done

# MPI strong scaling
echo "=== MPI Strong Scaling ==="
for np in 1 2 4 8; do
    echo "Processes: $np"
    mpirun -np $np ./bin/heat_mpi $Nx $Ny $Nz $T
done

# Hybrid scaling
echo "=== Hybrid Scaling (2 sockets, varying threads per rank) ==="
for t in 1 2 4 8; do
    echo "OMP_NUM_THREADS: $t, MPI ranks: 2"
    OMP_NUM_THREADS=$t mpirun -np 2 ./bin/heat_hybrid $Nx $Ny $Nz $T
done
```

**Scaling Efficiency:**
$$\eta = \frac{T_1}{p \cdot T_p} \times 100\%$$
where $T_p$ = time with $p$ processors.

Expected: >80% efficiency for OpenMP (single node), 50-70% for MPI (communication overhead).

### Weak Scaling (Fixed Problem per Processor, Varying Total Size)

```bash
#!/bin/bash
# OpenMP weak scaling
echo "=== OpenMP Weak Scaling ==="
for t in 1 2 4 8; do
    Nz_per_thread=64
    Nz=$((Nz_per_thread * t))
    echo "Threads: $t, Grid: 256x256x$Nz"
    OMP_NUM_THREADS=$t ./bin/heat_openmp 256 256 $Nz 100
done

# MPI weak scaling (256×256×128 per rank)
echo "=== MPI Weak Scaling ==="
for np in 1 2 4; do
    Nz=$((128 * np))
    echo "Processes: $np, Grid: 256x256x$Nz"
    mpirun -np $np ./bin/heat_mpi 256 256 $Nz 100
done
```

Expected: near-constant execution time (linear scaling in problem size).

### Bandwidth Estimation

Theoretical peak bandwidth = (# reads/cell) × (interior cells) × (time steps) / (execution time)

```bash
# Example: 256^3 grid, T=100, time=1.5s, 7 reads per cell
# Interior cells: 254^3 ≈ 16.4M
# Bandwidth = 7 × 16.4M × 100 / 1.5s ≈ 76.3 GB/s
# If GPU peak = 432 GB/s, efficiency = 76.3 / 432 ≈ 18% (naive, expected ~40-50%)
```

---

## Common Pitfalls & Solutions

### Race Conditions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Unsynchronized buffer access | Non-deterministic checksums | Ensure barrier before swap (pthreads), implicit in OpenMP regions |
| Shared data modification | Corrupted values at boundaries | Use `#pragma omp atomic` or proper critical sections |

### False Sharing

| Issue | Symptom | Solution |
|-------|---------|----------|
| Adjacent cache lines in shared array modified by different threads | Serialization bottleneck (> 10× slowdown) | Pad structures, align thread-private data to cache line (64B), use OpenMP `reduction` |

### MPI Deadlocks

| Issue | Symptom | Solution |
|-------|---------|----------|
| Send/recv ordering mismatch | Processes hang indefinitely | Use non-blocking (Isend/Irecv), MPI_Sendrecv, or explicit message tags |
| Insufficient buffer space | Blocking send overflow | Use non-blocking or ensure recv posted before send |

### CUDA Bank Conflicts

| Issue | Symptom | Solution |
|-------|---------|----------|
| Multiple threads access same bank | 32-way serialization | Pad shared memory, use fast-varying dimensions for 32-bit (or 64-bit) access pattern |
| Misaligned shared memory access | Warp stalls | Ensure (address % 8 bytes) is uniform within warp |

### NUMA Performance Degradation

| Issue | Symptom | Solution |
|-------|---------|----------|
| Cross-socket memory access (2-3× latency) | ~30% slower than optimal | Pin threads: `numactl --physcpubind`, `OMP_PROC_BIND=close`, or MPI process binding |
| Unbalanced NUMA allocation | Load imbalance between sockets | Use `numactl --interleave=all` for initial data or first-touch initialization with pinned threads |

### GPU Memory Overrun

| Issue | Symptom | Solution |
|-------|---------|----------|
| Grid allocation exceeds GPU memory | CUDA out-of-memory error | Check available GPU memory (`nvidia-smi`), reduce Nx/Ny/Nz, use smaller data type (float32 vs float64) |

---

## Performance Characteristics Summary

| Variant | Scaling | Memory Bandwidth | GFLOPs (256³, T=100) | Notes |
|---------|---------|------------------|----------------------|-------|
| **OpenMP** | Single node, ~8-16 cores | ~20-30 GB/s | 2-5 | Good for single-machine; NUMA-aware tuning needed |
| **pthreads** | Single node, fine control | ~20-30 GB/s | 2-5 | Lower-level; allows NUMA pinning |
| **MPI** | Multi-node/socket | 5-15 GB/s (inter-node) | 1-3 | Communication overhead; good weak scaling |
| **CUDA Naive** | GPU (1000s cores) | 40-50% peak | 1-3 | Memory-bound; simple to implement |
| **CUDA Optimized** | GPU (1000s cores) | 60-85% peak | 5-15 | Shared memory tiling; best GPU performance |
| **Hybrid** | Multi-socket + cores | 40-60 GB/s | 3-10 | Best for modern multi-socket CPUs |

---

## Advanced Tuning

### 1. Cache Blocking (CPU)

For large grids (Nx, Ny, Nz > 256), implement cache-aware blocking:

```c
int block_size = 64;  // Fit L3 cache
for (int k_b = 1; k_b < Nz-1; k_b += block_size) {
    for (int j_b = 1; j_b < Ny-1; j_b += block_size) {
        for (int i_b = 1; i_b < Nx-1; i_b += block_size) {
            // Compute block
        }
    }
}
```

### 2. NUMA Optimization (Hybrid)

```bash
# 2 NUMA domains, 8 cores each
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
mpirun -np 2 --bind-to socket ./bin/heat_hybrid 512 512 512 100
```

### 3. GPU Memory Optimization

- **Unified Memory:** `cudaMemAdvise` for page migration hints
- **Pinned Host Memory:** `cudaHostAlloc` for faster PCIe transfer
- **Persistent Kernels:** Keep data on GPU across multiple time steps (done implicitly)

### 4. Reduced Precision (float32 vs float64)

For error-tolerant applications:
- Reduces memory by 50%
- Increases bandwidth by ~2×
- Numerical error typically < 1e-3 with float32 (depends on T and grid size)

---

## References & Further Reading

1. **Stencil Codes & HPC:**
   - "Performance Optimization of Stencil Computations on Modern Microprocessors" (PACT 2013)
   - "Roofline: An Insightful Visual Performance Model for Floating-Point Programs" (CACM 2011)

2. **OpenMP & Shared Memory:**
   - OpenMP 5.0 Standard (www.openmp.org)
   - "Using OpenMP" by Chapman, Jost, van der Pas

3. **MPI & Distributed Computing:**
   - MPI 3.1 Standard (www.mpi-forum.org)
   - "Using MPI" by Gropp, Lusk, Thakur

4. **CUDA & GPU Programming:**
   - NVIDIA CUDA C Programming Guide (docs.nvidia.com)
   - "Professional CUDA C Programming" by Harris & Haines

5. **Hybrid & Multi-GPU:**
   - "Hybrid MPI+OpenMP for Modern HPC Applications" (IWOMP 2017)
   - "Achieving High Performance with NVIDIA GPUs" (GTC talks)

---

## FAQ

**Q: Which implementation should I use?**
A: Start with OpenMP (easiest, single node). Scale to Hybrid for multi-socket. Use MPI for clusters. CUDA if GPU available.

**Q: How do I measure performance accurately?**
A: Exclude I/O, run multiple iterations, use wall-clock timing (not CPU time), disable turbo/frequency scaling for reproducibility.

**Q: Why is my OpenMP code slower than serial?**
A: False sharing (check with `cachegrind`), wrong scheduling, thread imbalance, or excessive synchronization. Try `schedule(dynamic)` or smaller chunks.

**Q: How do I handle very large grids (1000³)?**
A: Use MPI + distributed storage, or process in time-sliced layers on GPU. Consider reduced precision (float32).

**Q: Can I use this code on a different system?**
A: Yes! Adjust compiler flags, MPI implementation, and GPU architecture. Code is portable C/C++/CUDA.

---

**Last Updated:** 2025-11-13  
**Authors:** HPC Implementation Suite  
**License:** Open-source (choose appropriate license for your organization)
