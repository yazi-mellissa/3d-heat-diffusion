/*
 * heat3d_cuda_naive.cu - 3D Heat Diffusion Solver using CUDA (naive global memory)
 *
 * Jacobi explicit stencil (7-point) with CUDA parallelization
 * Naive implementation: each thread computes one cell, reading from global memory
 *
 * Compilation:
 *   nvcc -O3 -arch=sm_70 heat3d_cuda_naive.cu -o heat_cuda_naive -I../include
 *
 * Execution:
 *   ./heat_cuda_naive 256 256 256 100
 *
 * Notes:
 * - Simple kernel: one thread per (i,j,k) cell
 * - 3D grid/block configuration (with linearization fallback for large grids)
 * - Global memory bound: lots of redundant loads/coalescing issues
 * - No shared memory optimization
 * - Boundary conditions enforced in separate kernel or explicit handling
 *
 * Performance characteristics:
 * - Memory-bound (low arithmetic intensity ~7 FLOPs per 7 double reads = ~1.0 FLOPs/load)
 * - Global memory throughput limited by coalescing violations
 * - Typical bandwidth: 30-50% of peak (vs 80%+ for optimized versions)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

/* ============================================================================
 * CUDA KERNEL: NAIVE 7-POINT STENCIL
 * ============================================================================ */

/**
 * Naive kernel: each thread computes u_new[i,j,k] from global memory reads
 * 
 * Grid configuration: linearized 1D grid for flexibility
 * - blockIdx.x iterates over thread blocks
 * - threadIdx.x is thread index within block
 * - Map global_tid to (i,j,k)
 */
__global__ void step_kernel_naive(
    const double *u_in,
    double *u_out,
    int Nx, int Ny, int Nz,
    double alpha, double beta)
{
    /* Compute global thread index */
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_interior = (Nx - 2) * (Ny - 2) * (Nz - 2);
    
    if (global_tid >= total_interior) return;
    
    /* Map 1D index to 3D coordinates (interior only: i,j,k in [1, N-2]) */
    int i = (global_tid % (Nx - 2)) + 1;
    int j_idx = global_tid / (Nx - 2);
    int j = (j_idx % (Ny - 2)) + 1;
    int k = (j_idx / (Ny - 2)) + 1;
    
    /* Linear index in 3D array */
    size_t idx = (size_t)k * Ny * Nx + j * Nx + i;
    
    /* 7-point stencil: read from global memory */
    double center = u_in[idx];
    double xi_lo  = u_in[idx - 1];              /* i-1 */
    double xi_hi  = u_in[idx + 1];              /* i+1 */
    double yj_lo  = u_in[idx - Nx];             /* j-1 */
    double yj_hi  = u_in[idx + Nx];             /* j+1 */
    double zk_lo  = u_in[idx - (size_t)Nx*Ny]; /* k-1 */
    double zk_hi  = u_in[idx + (size_t)Nx*Ny]; /* k+1 */
    
    /* Jacobi update */
    u_out[idx] = alpha * center + beta * (xi_lo + xi_hi + yj_lo + yj_hi + zk_lo + zk_hi);
}

/**
 * Kernel to enforce Dirichlet boundary conditions
 * Copy boundary cells from u_in to u_out
 */
__global__ void enforce_bc_kernel(
    const double *u_in,
    double *u_out,
    int Nx, int Ny, int Nz)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t N = (size_t)Nx * Ny * Nz;
    
    if (global_tid >= N) return;
    
    size_t idx = global_tid;
    int i = idx % Nx;
    int j = (idx / Nx) % Ny;
    int k = idx / (Nx * Ny);
    
    /* Check if on boundary */
    if (i == 0 || i == Nx - 1 ||
        j == 0 || j == Ny - 1 ||
        k == 0 || k == Nz - 1) {
        u_out[idx] = u_in[idx];
    }
}

/* ============================================================================
 * HOST UTILITIES
 * ============================================================================ */

/**
 * Check CUDA errors
 */
static void check_cuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

/**
 * Initialize grid on host
 */
static void init_grid_host(double *u, int Nx, int Ny, int Nz)
{
    size_t N = (size_t)Nx * Ny * Nz;
    for (size_t p = 0; p < N; p++) {
        int i = p % Nx;
        int j = (p / Nx) % Ny;
        int k = p / (Nx * Ny);
        
        if (i == 0 || i == Nx - 1 ||
            j == 0 || j == Ny - 1 ||
            k == 0 || k == Nz - 1) {
            u[p] = 1.0;
        } else {
            u[p] = 0.0;
        }
    }
}

/**
 * Compute checksum on host
 */
static double compute_checksum_host(const double *u, size_t N)
{
    double sum = 0.0;
    for (size_t p = 0; p < N; p++) {
        sum += u[p];
    }
    return sum;
}

/* ============================================================================
 * MAIN SOLVER
 * ============================================================================ */

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s Nx Ny Nz T [alpha] [beta]\n", argv[0]);
        fprintf(stderr, "  Nx, Ny, Nz : grid dimensions\n");
        fprintf(stderr, "  T          : number of time steps\n");
        fprintf(stderr, "  alpha      : stencil center coefficient (default 0.0)\n");
        fprintf(stderr, "  beta       : stencil neighbor coefficient (default 1/6)\n");
        return 1;
    }

    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);
    int T = atoi(argv[4]);
    double alpha = (argc > 5) ? atof(argv[5]) : 0.0;
    double beta = (argc > 6) ? atof(argv[6]) : 1.0 / 6.0;

    if (Nx < 3 || Ny < 3 || Nz < 3 || T < 1) {
        fprintf(stderr, "Error: Nx, Ny, Nz >= 3, T >= 1\n");
        return 1;
    }

    size_t N = (size_t)Nx * Ny * Nz;
    size_t memory_bytes = 2 * N * sizeof(double);

    if (memory_bytes > 4LL * 1024 * 1024 * 1024) {
        fprintf(stderr, "Warning: memory footprint %.2f GB may exceed GPU memory\n",
                (double)memory_bytes / (1024*1024*1024));
    }

    printf("[CUDA naive] Grid: %d x %d x %d, T=%d, alpha=%.2e, beta=%.2e\n",
           Nx, Ny, Nz, T, alpha, beta);
    printf("[CUDA naive] Memory: %.2f GB\n", (double)memory_bytes / (1024*1024*1024));

    /* ========================================================================
     * ALLOCATE HOST MEMORY & INITIALIZE
     * ======================================================================== */
    double *h_u = (double *)malloc(N * sizeof(double));
    if (!h_u) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    init_grid_host(h_u, Nx, Ny, Nz);

    /* ========================================================================
     * ALLOCATE DEVICE MEMORY
     * ======================================================================== */
    double *d_u = NULL, *d_un = NULL;
    check_cuda(cudaMalloc(&d_u, N * sizeof(double)), "cudaMalloc d_u");
    check_cuda(cudaMalloc(&d_un, N * sizeof(double)), "cudaMalloc d_un");

    check_cuda(cudaMemcpy(d_u, h_u, N * sizeof(double), cudaMemcpyHostToDevice),
               "cudaMemcpy h_u -> d_u");
    check_cuda(cudaMemcpy(d_un, h_u, N * sizeof(double), cudaMemcpyHostToDevice),
               "cudaMemcpy h_u -> d_un");

    /* ========================================================================
     * CONFIGURE KERNEL LAUNCH
     * ======================================================================== */
    int num_interior = (Nx - 2) * (Ny - 2) * (Nz - 2);
    int block_size = 256;
    int grid_size = (num_interior + block_size - 1) / block_size;

    printf("[CUDA naive] Interior points: %d, grid size: %d, block size: %d\n",
           num_interior, grid_size, block_size);

    /* Full grid for boundary kernel */
    int grid_size_full = (N + block_size - 1) / block_size;

    /* ========================================================================
     * MAIN TIME-STEPPING LOOP
     * ======================================================================== */
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");

    for (int t = 0; t < T; t++) {
        /* Compute interior stencil */
        step_kernel_naive<<<grid_size, block_size>>>(
            d_u, d_un, Nx, Ny, Nz, alpha, beta);
        check_cuda(cudaGetLastError(), "step_kernel_naive");

        /* Enforce boundary conditions */
        enforce_bc_kernel<<<grid_size_full, block_size>>>(d_u, d_un, Nx, Ny, Nz);
        check_cuda(cudaGetLastError(), "enforce_bc_kernel");

        /* Swap device pointers */
        double *tmp = d_u;
        d_u = d_un;
        d_un = tmp;
    }

    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

    float elapsed_ms = 0.0;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    double elapsed = elapsed_ms / 1000.0;

    /* ========================================================================
     * COPY RESULT BACK & COMPUTE CHECKSUM
     * ======================================================================== */
    check_cuda(cudaMemcpy(h_u, d_u, N * sizeof(double), cudaMemcpyDeviceToHost),
               "cudaMemcpy d_u -> h_u");

    double checksum = compute_checksum_host(h_u, N);

    printf("[CUDA naive] checksum=%.15e time=%.6f s\n", checksum, elapsed);
    printf("[CUDA naive] GFLOPS estimate: %.2f\n",
           7.0 * (Nx - 2) * (Ny - 2) * (Nz - 2) * T / 1e9 / elapsed);

    /* ========================================================================
     * CLEANUP
     * ======================================================================== */
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check_cuda(cudaFree(d_u), "cudaFree d_u");
    check_cuda(cudaFree(d_un), "cudaFree d_un");
    free(h_u);

    return 0;
}
