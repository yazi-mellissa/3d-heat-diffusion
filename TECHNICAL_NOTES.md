# Technical Notes & Performance Analysis

## Memory Layout & Indexing

### Row-Major (C) Layout

All implementations use **row-major (C-contiguous) layout** for 3D arrays:

```c
/* Array stored as: u[0,0,0], u[0,0,1], ..., u[0,0,Nx-1], u[0,1,0], ... */
size_t idx = k*Ny*Nx + j*Nx + i;
```

**Advantages:**
- Fast-varying dimension (i) for x-iteration → better coalescing on GPUs
- Contiguous k-layers for distributed computing (MPI)
- Cache-friendly for row-wise iteration

**Stencil indexing:**
```c
idx         = k*Ny*Nx + j*Nx + i;
idx - 1     = k*Ny*Nx + j*Nx + (i-1)      // i-1
idx + 1     = k*Ny*Nx + j*Nx + (i+1)      // i+1
idx - Nx    = k*Ny*Nx + (j-1)*Nx + i      // j-1
idx + Nx    = k*Ny*Nx + (j+1)*Nx + i      // j+1
idx - Nx*Ny = (k-1)*Ny*Nx + j*Nx + i      // k-1
idx + Nx*Ny = (k+1)*Ny*Nx + j*Nx + i      // k+1
```

---

## OpenMP Implementation Details

### Collapse Directive

```c
#pragma omp parallel for collapse(3) schedule(static)
for (int k = 1; k < Nz-1; k++) {
    for (int j = 1; j < Ny-1; j++) {
        for (int i = 1; i < Nx-1; i++) {
            // ... compute stencil
        }
    }
}
```

**Why `collapse(3)`?**
- Without collapse: OpenMP parallelizes only outer loop (k), leaving (j,i) serial
- Interior cells: $(Nx-2) \times (Ny-2) \times (Nz-2)$ → many more parallel tasks
- With collapse(3): All 3 loops collapsed into single flat iteration space
- Load balancing: better distribution across threads (especially if Nz is small)

**Schedule choice:**
- `schedule(static)`: distributes iterations contiguously to threads
  - Pros: better cache locality, no synchronization overhead
  - Cons: load imbalance if iterations unequal (rare for stencil)
- Alternative: `schedule(dynamic)` or `schedule(guided)` if load imbalance

### False Sharing in Boundary Copy

```c
#pragma omp parallel for schedule(static)
for (size_t p = 0; p < N; p++) {
    int i = p % Nx;
    int j = (p / Nx) % Ny;
    int k = p / (Nx * Ny);
    
    if (i == 0 || i == Nx - 1 || ...) {
        un[p] = u[p];  // Boundary update
    }
}
```

**False sharing risk:**
- Boundary cells sparse in memory (not contiguous)
- Thread A (copying cell [0,j,k]) and Thread B (copying cell [Nx-1,j,k]) may access adjacent memory
- Cache line (64 bytes) contains ~8 doubles → high probability of conflicts
- **Mitigation:** Coarse granularity (one thread per cache line) auto-achieved by static scheduling

---

## pthreads Implementation Details

### Barrier Synchronization

```c
for (int t = 0; t < T; t++) {
    /* Compute slices (thread-local work) */
    for (int k = z0; k < z1; k++) { ... }
    
    /* Synchronization point: all threads must reach here */
    pthread_barrier_wait(&barrier);
    
    /* Thread 0 swaps buffers (mutual exclusion implicit) */
    if (tid == 0) {
        double *tmp = u;
        u = un;
        un = tmp;
    }
    
    /* All threads must see updated pointers */
    pthread_barrier_wait(&barrier);
}
```

**Barrier Cost:**
- Small overhead per barrier: ~100-500 ns (on modern CPUs)
- Per time step: 2 barriers → 200-1000 ns total
- For 100 time steps, 8 threads: ~1-10 µs barrier overhead
- Negligible if computation time > 1 ms per timestep

### Z-slab Decomposition

```c
int z0 = (tid * Nz) / num_threads;
int z1 = ((tid + 1) * Nz) / num_threads;
```

**Advantages:**
- Contiguous memory per thread (good NUMA locality)
- No false sharing across threads (each thread owns disjoint k-range)
- Easy to pin to NUMA node: `numactl --cpubind=node:$nodeid`

**Disadvantages:**
- Load imbalance possible if Nz not divisible by num_threads
- Communication-intensive for multi-socket systems (halos at k-boundaries)

---

## MPI Implementation Details

### Domain Decomposition & Halo Exchange

**Global grid:** $Nx \times Ny \times Nz$ points  
**Per rank i:** $Nz_{\text{local},i}$ layers where:
$$Nz_{\text{local},i} = \lfloor (Nz + \text{nprocs} - 1) / \text{nprocs} \rfloor$$

**Halo layers:**
- k=1 layer (interior boundary with k=0): sent to rank i-1, received from rank i-1
- k=Nz_local-2 layer: sent to rank i+1, received from rank i+1
- Global boundaries (k=0, k=Nz-1): enforced by appropriate ranks only

**Message pattern (ring topology):**
```
Rank 0          Rank 1          Rank 2
+-------+       +-------+       +-------+
| k=0   | ←→ k=1 layer ←→ | k=1   |
| ...   |       | ...   |       | ...   |
| k=Nz1 | ←→ k=Nz1-2     ←→ | k=Nz2 |
+-------+       +-------+       +-------+
        ↓              ↓              ↓
    Interior     Interior        Interior
```

**Non-blocking protocol:**
```c
MPI_Irecv(..., prev_rank, tag_from_prev);
MPI_Irecv(..., next_rank, tag_from_next);
MPI_Isend(..., prev_rank, tag_to_prev);
MPI_Isend(..., next_rank, tag_to_next);

/* Compute interior (k in [2, Nz_local-2)) while MPI proceeds */
compute_interior();

/* Wait for halos to arrive */
MPI_Waitall(...);

/* Compute boundary slices using received halos */
compute_boundary();
```

**Overlap benefit:**
- Interior computation: ~T_comp seconds
- Halo exchange: ~T_comm seconds
- Pure MPI (sequential): T_comp + T_comm
- Overlapped MPI: max(T_comp, T_comm)
- Speedup: (T_comp + T_comm) / max(T_comp, T_comm) ≈ 1.5-2× if balanced

### Global Checksum Reduction

```c
double local_sum = compute_checksum_local(u, N_local);
double global_sum;
MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
```

**Complexity:** $O(\log \text{nprocs})$ (tree reduction)

---

## CUDA Naive Implementation Details

### Thread Mapping

**Problem:** Map 1D grid/block topology to 3D interior points

```c
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int total_interior = (Nx-2) * (Ny-2) * (Nz-2);

if (global_tid >= total_interior) return;

/* Reverse-map 1D index to (i, j, k) */
int i = (global_tid % (Nx - 2)) + 1;
int j_idx = global_tid / (Nx - 2);
int j = (j_idx % (Ny - 2)) + 1;
int k = (j_idx / (Ny - 2)) + 1;
```

**Why linearized grid?**
- 3D grid limits: blockDim.z ≤ 32, gridDim.z ≤ 65535
- Linearized: blockDim.x ≤ 1024, gridDim.x ≤ 2^31-1 (much larger)

### Memory Access Patterns

**7 global reads + 1 write per interior cell:**

```c
double center = u_in[idx];                 // Read 1
double xi_lo  = u_in[idx - 1];             // Read 2
double xi_hi  = u_in[idx + 1];             // Read 3
double yj_lo  = u_in[idx - Nx];            // Read 4
double yj_hi  = u_in[idx + Nx];            // Read 5
double zk_lo  = u_in[idx - Nx*Ny];         // Read 6
double zk_hi  = u_in[idx + Nx*Ny];         // Read 7
u_out[idx] = ... ;                         // Write 1
```

**Coalescing analysis:**
- Threads in warp (0, 1, ..., 31) compute consecutive i-indices
- Memory addresses: idx, idx+1, idx+2, ..., idx+31
- **First read (center):** coalesced into 4 L2 transactions (128B each)
- **xi_lo, xi_hi:** also coalesced
- **yj_lo:** addresses = idx ± Nx → strides of Nx (likely non-coalesced)
- **zk_lo, zk_hi:** addresses = idx ± Nx*Ny → large strides (scattered)

**Effective bandwidth:** ~30-50% of peak (due to scattered access patterns)

---

## CUDA Optimized Implementation Details

### Shared Memory Tiling

**Tile structure:**
```c
#define TILE_X 8
#define TILE_Y 8
#define TILE_SIZE_X (TILE_X + 2)  /* 10 with halos */
#define TILE_SIZE_Y (TILE_Y + 2)  /* 10 with halos */

extern __shared__ double shared_tile[];
/* Layout: shared_tile[ty][tx] where (ty,tx) in [0,TILE_SIZE_Y) x [0,TILE_SIZE_X) */
```

**Memory size per block:** 
$(8+2) \times (8+2) \times 8$ bytes = 800 bytes (tiny, fits in ~48 KB shared mem available)

**Loading halos:**
```
Thread (tx, ty) loads:
  - Interior:    shmem[ty+1][tx+1] = global[k, j_base+ty, i_base+tx]
  - Left halo:   shmem[ty+1][tx]   = global[k, j_base+ty, i_base+tx-1]  (if tx==0)
  - Right halo:  shmem[ty+1][tx+2] = global[k, j_base+ty, i_base+tx+1]  (if tx==TILE_X-1)
  - Similarly for top/bottom/corners
```

**Synchronization:**
```c
__syncthreads();      // Ensure all threads loaded their cells
/* Compute stencil using shmem + zk_lo, zk_hi from global */
__syncthreads();      // Before next k iteration
```

### Bank Conflict Mitigation

**NVIDIA GPU shared memory organization:**
- 32 banks, each 4 bytes wide (for 32-bit types) or 8 bytes (for 64-bit)
- Address modulo 32 determines bank number
- Sequential addresses map to different banks (ideal)

**Fast-varying x dimension:** ensures natural bank mapping
```c
/* Layout: shmem[ty * TILE_SIZE_X + tx] */
/* Consecutive threads (tx) access different banks (tx % 32) */
```

**Potential conflict:**
- If ty differs, but (ty * TILE_SIZE_X + tx) % 32 is same
- TILE_SIZE_X = 10 → bank mapping repeats every 32/10 ≈ 3 rows
- Typical conflict rate: ~1-2 per warp (minimal impact)

---

## Hybrid MPI+OpenMP Implementation Details

### Thread Safety

```c
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
```

**Thread level:** `MPI_THREAD_FUNNELED`
- Only the master thread (tid=0) may call MPI functions
- OpenMP threads work on computation only
- Implicit synchronization at barriers ensures visibility

### Execution Pattern

```
Time step t:
  ├─ Master thread: post MPI Irecv/Isend
  ├─ All threads: compute interior with #pragma omp parallel for collapse(3)
  ├─ (Implicit barrier at end of omp region)
  ├─ Master thread: MPI_Waitall for halos
  ├─ All threads: compute boundary slices with #pragma omp parallel for
  ├─ All threads: enforce BCs with #pragma omp parallel for
  └─ Implicit barrier; proceed to t+1
```

**Load balancing:**
- Interior computation: $(Nx-2) \times (Ny-2) \times (Nz_{\text{local}}-4)$ cells
- Boundary slices: $2 \times (Nx-2) \times (Ny-2)$ cells
- Ratio: typically 10-100× more interior → boundary usually fast

---

## Numerical Considerations

### Stability & Convergence

**Jacobi method (explicit):**
- Stable if $|\alpha| + 6 \beta \leq 1$ (CFL-like condition)
- Typical: $\alpha = 0$, $\beta = 1/6$ → center coefficient = 1 (marginal)
- For diffusion: $u^{n+1} = u^n + \Delta t \, k \nabla^2 u^n$
  - $\beta = k \, \Delta t / (\Delta x)^2$ must satisfy stability

**Dirichlet BC:**
- Boundaries fixed at u=1.0 throughout
- Interior diffusion drives u towards 1.0
- After sufficient time steps, interior u ≈ 1.0

### Floating-Point Precision

**Double precision (float64):**
- Machine epsilon ≈ 2.22e-16
- For accumulation (checksum): $N = 256^3 = 16.8M$ terms
- Relative error ≈ $\sqrt{N} \times \epsilon ≈ 1e-9$ (expected)
- All implementations should match to ~1e-10 relative

**float32 (for memory-constrained):**
- Machine epsilon ≈ 1.19e-7
- Relative error ≈ 1e-4 (degraded but acceptable for some applications)
- 50% memory reduction, 2× bandwidth benefit

---

## Profiling & Debugging

### CPU Profiling

**Linux with `perf`:**
```bash
perf record -g OMP_NUM_THREADS=8 ./bin/heat_openmp 256 256 256 100
perf report
```

**Cachegrind (memory behavior):**
```bash
valgrind --tool=cachegrind OMP_NUM_THREADS=2 ./bin/heat_openmp 32 32 32 10
cg_annotate cachegrind.out.<pid>
```

**Thread analysis (Intel VTune):**
```bash
vtune -collect hpc-performance -result-dir ./vtune_result ./bin/heat_openmp 256 256 256 100
```

### GPU Profiling

**NVIDIA Nsight Systems:**
```bash
nsys profile -o heat_profile ./bin/heat_cuda_opt 128 128 128 10
nsys-ui heat_profile.nsys-rep
```

**Nsight Compute (kernel-level):**
```bash
ncu --set full ./bin/heat_cuda_opt 128 128 128 10
```

**Legacy nvprof:**
```bash
nvprof --analysis-metrics ./bin/heat_cuda_opt 128 128 128 10
```

---

## Scaling Laws & Limits

### Amdahl's Law

Parallel speedup limited by serial fraction:
$$S = \frac{1}{f + (1-f)/p}$$

where $f$ = serial fraction, $p$ = processors.

**Example:** If 10% serial, 90% parallel:
- p=8:  S = 1 / (0.1 + 0.9/8) ≈ 6.1× (efficiency ≈ 76%)
- p=64: S = 1 / (0.1 + 0.9/64) ≈ 7.4× (efficiency ≈ 12%)

**For stencil codes:** serial fraction ≈ 2-5% (halo exchange + sync)

### Roofline Model

Maximum achievable performance bounded by:
$$\text{GFLOPs} = \min(\text{peak GFLOP/s}, \text{peak GB/s} \times \text{AI})$$

where **AI** (arithmetic intensity) = FLOPs / bytes transferred.

**Heat3D stencil:**
- FLOPs per interior cell: 7 (6 adds + 1 multiply)
- Bytes per cell: $7 \times 8 = 56$ (reads) + $8$ (write) = 64 bytes
- AI = 7 / 64 ≈ 0.11 FLOP/byte

**CPU roofline (e.g., Xeon, 50 GB/s peak, 1 TFLOP/s peak):**
- Memory-bound limit: 50 GB/s × 0.11 ≈ 5.5 GFLOP/s (far below peak)

**GPU roofline (e.g., RTX 3080, 760 GB/s peak, 30 TFLOP/s peak):**
- Memory-bound limit: 760 GB/s × 0.11 ≈ 84 GFLOP/s (still limited by memory)
- Shared memory optimization reduces effective AI → higher GFLOP/s

---

## Validation Checklist

- [ ] Checksums match across all implementations (same initial conditions)
- [ ] OpenMP shows >80% scaling efficiency (2, 4, 8 threads)
- [ ] pthreads matches OpenMP performance
- [ ] MPI scales (weak: constant time as problem grows proportionally)
- [ ] CUDA naive < CUDA optimized by 5-10×
- [ ] Hybrid MPI+OpenMP faster than pure MPI on multi-socket
- [ ] No memory leaks (valgrind, CUDA memcheck)
- [ ] Boundary conditions correctly enforced (interior → 1.0 after many steps)
- [ ] No race conditions (tsan, race detector)
- [ ] GFLOPS estimates reasonable (1-5 GFLOP/s CPU, 5-20 GFLOP/s GPU)

---

## References for Further Optimization

1. **OpenMP:** OpenMP 5.0 Specification, collapse/schedule directives
2. **MPI:** Gropp et al. "Using Advanced MPI", Chapter 4 (non-blocking communication)
3. **CUDA:** Harris et al. "Optimizing Parallel Reduction in CUDA", shared memory tiling
4. **Roofline:** Williams et al. "Roofline: An Insightful Visual Performance Model"
5. **Stencil:** Datta et al. "Stencil Computation Optimization and Auto-tuning"

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-13
