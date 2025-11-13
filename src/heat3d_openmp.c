/*
 * heat3d_openmp.c - 3D Heat Diffusion Solver using OpenMP
 *
 * Jacobi explicit stencil (7-point) with OpenMP parallelization
 * Decoupling: triple loop with collapse(3) directive
 * 
 * Compilation:
 *   gcc -O3 -march=native -fopenmp heat3d_openmp.c -o heat_openmp -I../include -lm
 *
 * Execution:
 *   OMP_NUM_THREADS=8 ./heat_openmp 256 256 256 100
 *
 * Notes:
 * - collapse(3) to parallelize all three nested loops
 * - Proper scheduling (static, dynamic, guided) affects cache locality
 * - Buffer swapping: u and un pointers swapped after each iteration
 * - Boundary conditions (Dirichlet) are maintained by copying borders
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../include/heat3d.h"

/**
 * Main solver: Jacobi stencil with OpenMP parallelization
 */
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

    /* Parse arguments */
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);
    int T = atoi(argv[4]);
    
    double alpha = (argc > 5) ? atof(argv[5]) : 0.0;
    double beta = (argc > 6) ? atof(argv[6]) : 1.0 / 6.0;

    /* Check grid dimensions */
    if (Nx < 3 || Ny < 3 || Nz < 3 || T < 1) {
        fprintf(stderr, "Error: Nx, Ny, Nz must be >= 3, T must be >= 1\n");
        return 1;
    }

    size_t N = (size_t)Nx * Ny * Nz;
    
    /* Check memory footprint */
    size_t memory_bytes = 2 * N * sizeof(double);
    if (memory_bytes > 16LL * 1024 * 1024 * 1024) { /* 16 GB limit */
        fprintf(stderr, "Warning: memory footprint ~%.2f GB\n", (double)memory_bytes / (1024*1024*1024));
    }

    /* Allocate buffers */
    double *u = (double *)malloc(N * sizeof(double));
    double *un = (double *)malloc(N * sizeof(double));
    
    if (!u || !un) {
        fprintf(stderr, "Error: malloc failed (tried to allocate %.2f GB)\n",
                (double)memory_bytes / (1024*1024*1024));
        free(u);
        free(un);
        return 1;
    }

    /* Initialize grid: Dirichlet BC (boundaries = 1.0, interior = 0.0) */
    init_grid(u, Nx, Ny, Nz);
    memcpy(un, u, N * sizeof(double));

    /* Print configuration */
    int num_threads = omp_get_max_threads();
    printf("[OpenMP] Grid: %d x %d x %d, T=%d, alpha=%.2e, beta=%.2e\n",
           Nx, Ny, Nz, T, alpha, beta);
    printf("[OpenMP] Threads: %d, Memory: %.2f GB\n",
           num_threads, (double)(2 * N * sizeof(double)) / (1024*1024*1024));

    /* ========================================================================
     * MAIN TIME STEPPING LOOP
     * ======================================================================== */
    double t_start = omp_get_wtime();

    for (int t = 0; t < T; t++) {
        /* Compute interior points using 7-point stencil
         * collapse(3): parallelize all three nested loops for better load balancing
         * schedule(static): cache-friendly on large grids
         */
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < Nz - 1; k++) {
            for (int j = 1; j < Ny - 1; j++) {
                for (int i = 1; i < Nx - 1; i++) {
                    size_t idx = (size_t)k * Ny * Nx + j * Nx + i;
                    
                    /* 7-point stencil: center + 6 neighbors */
                    double center = u[idx];
                    double xi_lo  = u[idx - 1];           /* i-1 */
                    double xi_hi  = u[idx + 1];           /* i+1 */
                    double yj_lo  = u[idx - Nx];          /* j-1 */
                    double yj_hi  = u[idx + Nx];          /* j+1 */
                    double zk_lo  = u[idx - Nx * Ny];     /* k-1 */
                    double zk_hi  = u[idx + Nx * Ny];     /* k+1 */
                    
                    un[idx] = alpha * center + 
                              beta * (xi_lo + xi_hi + yj_lo + yj_hi + zk_lo + zk_hi);
                }
            }
        }

        /* Enforce Dirichlet boundary conditions (keep borders fixed at 1.0)
         * This parallelizes boundary copy using a flat loop
         */
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < N; p++) {
            int i = p % Nx;
            int j = (p / Nx) % Ny;
            int k = p / (Nx * Ny);
            
            /* If on boundary, keep original value */
            if (i == 0 || i == Nx - 1 ||
                j == 0 || j == Ny - 1 ||
                k == 0 || k == Nz - 1) {
                un[p] = u[p];
            }
        }

        /* Swap buffers */
        double *tmp = u;
        u = un;
        un = tmp;
    }

    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;

    /* ========================================================================
     * COMPUTE CHECKSUM & REPORT
     * ======================================================================== */
    double checksum = compute_checksum(u, N);

    print_result("OpenMP", checksum, elapsed);
    printf("[OpenMP] GFLOPS estimate: %.2f (7*interior*T / 1e9 / time)\n",
           7.0 * (Nx-2) * (Ny-2) * (Nz-2) * T / 1e9 / elapsed);

    /* Cleanup */
    free(u);
    free(un);

    return 0;
}
