/*
 * heat3d_pthreads.c - 3D Heat Diffusion Solver using pthreads
 *
 * Jacobi explicit stencil (7-point) with pthreads parallelization
 * Domain decomposition: Z-slicing, each thread computes a slab of k-layers
 * Synchronization: pthread_barrier_t at each time step
 *
 * Compilation:
 *   gcc -O3 -march=native -pthread heat3d_pthreads.c -o heat_pthreads -I../include -lm
 *
 * Execution:
 *   ./heat_pthreads 256 256 256 100 8
 *   (last argument: number of threads)
 *
 * Notes:
 * - Domain decomposition along Z-axis: each thread processes k in [z0, z1)
 * - Shared u and un buffers accessed with atomic/lock-free swap via barrier
 * - Barrier synchronization ensures all threads finish before buffer swap
 * - Cache locality: threads work on contiguous k-slices (good NUMA behavior)
 * - False sharing risk: minimize shared writes between threads
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include "../include/heat3d.h"

/* ============================================================================
 * THREAD ARGUMENT STRUCTURE
 * ============================================================================ */
typedef struct {
    int thread_id;              /* Thread index in [0, num_threads) */
    int num_threads;            /* Total number of threads */
    
    /* Problem parameters (read-only) */
    int Nx, Ny, Nz;
    int T;
    double alpha, beta;
    
    /* Shared buffers */
    double *u;                  /* Current time step */
    double *un;                 /* Next time step */
    
    /* Synchronization */
    pthread_barrier_t *barrier;
    int *swap_flag;             /* Signals when to swap buffers */
    
} thread_arg_t;

/**
 * Get current time in seconds (using gettimeofday for better precision)
 */
static double get_wtime(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

/**
 * Worker thread function
 * - Computes stencil for z-slice assigned to this thread
 * - Synchronizes via barrier at each time step
 * - Thread 0 performs buffer swap
 */
static void *worker_thread(void *arg)
{
    thread_arg_t *a = (thread_arg_t *)arg;
    int tid = a->thread_id;
    
    /* Compute z-slab boundaries for this thread */
    int z0 = (tid * a->Nz) / a->num_threads;
    int z1 = ((tid + 1) * a->Nz) / a->num_threads;
    
    if (tid == 0) {
        printf("[pthreads] Thread 0: z in [%d, %d)\n", z0, z1);
    }

    /* Main time-stepping loop */
    for (int t = 0; t < a->T; t++) {
        
        /* Compute interior stencil for this thread's z-slab
         * Each thread works on k in [z0_interior, z1_interior)
         * where z0_interior = max(1, z0) and z1_interior = min(Nz-1, z1)
         */
        int k_start = (z0 == 0) ? 1 : z0;
        int k_end   = (z1 == a->Nz) ? a->Nz - 1 : z1;
        
        for (int k = k_start; k < k_end; k++) {
            for (int j = 1; j < a->Ny - 1; j++) {
                for (int i = 1; i < a->Nx - 1; i++) {
                    size_t idx = (size_t)k * a->Ny * a->Nx + j * a->Nx + i;
                    
                    /* 7-point stencil */
                    double center = a->u[idx];
                    double xi_lo  = a->u[idx - 1];
                    double xi_hi  = a->u[idx + 1];
                    double yj_lo  = a->u[idx - a->Nx];
                    double yj_hi  = a->u[idx + a->Nx];
                    double zk_lo  = a->u[idx - a->Nx * a->Ny];
                    double zk_hi  = a->u[idx + a->Nx * a->Ny];
                    
                    a->un[idx] = a->alpha * center +
                                 a->beta * (xi_lo + xi_hi + yj_lo + yj_hi + zk_lo + zk_hi);
                }
            }
        }

        /* Enforce Dirichlet boundary conditions for this thread's z-slab
         * Boundaries (i=0, i=Nx-1, j=0, j=Ny-1) within each k-slab are copied
         * k-boundaries (k=0, k=Nz-1) handled by appropriate threads only
         */
        for (int k = z0; k < z1; k++) {
            for (int j = 0; j < a->Ny; j++) {
                for (int i = 0; i < a->Nx; i++) {
                    size_t idx = (size_t)k * a->Ny * a->Nx + j * a->Nx + i;
                    
                    /* Copy boundaries */
                    if (i == 0 || i == a->Nx - 1 ||
                        j == 0 || j == a->Ny - 1 ||
                        k == 0 || k == a->Nz - 1) {
                        a->un[idx] = a->u[idx];
                    }
                }
            }
        }

        /* Synchronization barrier: wait for all threads to finish this time step */
        pthread_barrier_wait(a->barrier);

        /* Thread 0 swaps buffers (producer pattern) */
        if (tid == 0) {
            double *tmp = a->u;
            a->u = a->un;
            a->un = tmp;
        }

        /* All threads wait again before next iteration */
        pthread_barrier_wait(a->barrier);
    }

    return NULL;
}

/**
 * Main solver
 */
int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s Nx Ny Nz T [num_threads] [alpha] [beta]\n", argv[0]);
        fprintf(stderr, "  Nx, Ny, Nz : grid dimensions\n");
        fprintf(stderr, "  T          : number of time steps\n");
        fprintf(stderr, "  num_threads: number of pthreads (default: #CPUs)\n");
        fprintf(stderr, "  alpha      : stencil center coefficient (default 0.0)\n");
        fprintf(stderr, "  beta       : stencil neighbor coefficient (default 1/6)\n");
        return 1;
    }

    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);
    int T = atoi(argv[4]);
    int num_threads = (argc > 5) ? atoi(argv[5]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    double alpha = (argc > 6) ? atof(argv[6]) : 0.0;
    double beta = (argc > 7) ? atof(argv[7]) : 1.0 / 6.0;

    if (Nx < 3 || Ny < 3 || Nz < 3 || T < 1 || num_threads < 1) {
        fprintf(stderr, "Error: Nx, Ny, Nz >= 3, T >= 1, num_threads >= 1\n");
        return 1;
    }

    size_t N = (size_t)Nx * Ny * Nz;

    /* Allocate buffers */
    double *u = (double *)malloc(N * sizeof(double));
    double *un = (double *)malloc(N * sizeof(double));

    if (!u || !un) {
        fprintf(stderr, "Error: malloc failed\n");
        free(u);
        free(un);
        return 1;
    }

    /* Initialize */
    init_grid(u, Nx, Ny, Nz);
    memcpy(un, u, N * sizeof(double));

    printf("[pthreads] Grid: %d x %d x %d, T=%d, threads=%d\n",
           Nx, Ny, Nz, T, num_threads);

    /* ========================================================================
     * CREATE THREADS
     * ======================================================================== */
    
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    thread_arg_t *args = (thread_arg_t *)malloc(num_threads * sizeof(thread_arg_t));
    pthread_barrier_t barrier;

    if (!threads || !args) {
        fprintf(stderr, "Error: thread allocation failed\n");
        free(threads);
        free(args);
        free(u);
        free(un);
        return 1;
    }

    /* Initialize barrier (num_threads + 1 for potential synchronization) */
    if (pthread_barrier_init(&barrier, NULL, num_threads) != 0) {
        fprintf(stderr, "Error: pthread_barrier_init failed\n");
        return 1;
    }

    /* Prepare thread arguments */
    for (int t = 0; t < num_threads; t++) {
        args[t].thread_id = t;
        args[t].num_threads = num_threads;
        args[t].Nx = Nx;
        args[t].Ny = Ny;
        args[t].Nz = Nz;
        args[t].T = T;
        args[t].alpha = alpha;
        args[t].beta = beta;
        args[t].u = u;
        args[t].un = un;
        args[t].barrier = &barrier;
    }

    double t_start = get_wtime();

    /* Spawn threads */
    for (int t = 0; t < num_threads; t++) {
        if (pthread_create(&threads[t], NULL, worker_thread, &args[t]) != 0) {
            fprintf(stderr, "Error: pthread_create failed\n");
            return 1;
        }
    }

    /* Wait for all threads to finish */
    for (int t = 0; t < num_threads; t++) {
        if (pthread_join(threads[t], NULL) != 0) {
            fprintf(stderr, "Error: pthread_join failed\n");
            return 1;
        }
    }

    double t_end = get_wtime();
    double elapsed = t_end - t_start;

    /* ========================================================================
     * COMPUTE CHECKSUM & REPORT
     * ======================================================================== */
    double checksum = compute_checksum(u, N);

    print_result("pthreads", checksum, elapsed);
    printf("[pthreads] GFLOPS estimate: %.2f\n",
           7.0 * (Nx-2) * (Ny-2) * (Nz-2) * T / 1e9 / elapsed);

    /* Cleanup */
    pthread_barrier_destroy(&barrier);
    free(threads);
    free(args);
    free(u);
    free(un);

    return 0;
}
