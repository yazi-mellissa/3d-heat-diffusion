/*
 * heat3d_hybrid.c - 3D Heat Diffusion Solver using MPI + OpenMP (Hybrid)
 *
 * Jacobi explicit stencil (7-point) with hybrid parallelization
 * MPI: domain decomposition along Z-axis (inter-node)
 * OpenMP: parallelizes loops within each rank (intra-node)
 *
 * Compilation:
 *   mpicc -O3 -march=native -fopenmp heat3d_hybrid.c -o heat_hybrid -I../include -lm
 *
 * Execution:
 *   mpirun -np 2 ./heat_hybrid 256 256 256 100
 *   OMP_NUM_THREADS=8 mpirun -np 2 ./heat_hybrid 512 512 512 50
 *
 * Notes:
 * - Combines MPI domain decomposition (Z-slices, one rank per NUMA domain typically)
 * - OpenMP parallelizes interior triple loops within each rank
 * - Thread safety: MPI calls made by master thread only (implicit serialization)
 * - Halo exchange and boundary computation sequential; interior computation parallel
 * - Expected speedup: MPI scales across sockets/nodes, OpenMP within shared-memory
 *
 * Performance tuning:
 * - Ranks per node: typically = number of NUMA domains (e.g., 2 or 4)
 * - Threads per rank: cores_per_node / ranks_per_node
 * - Use OMP_PROC_BIND=close and OMP_PLACES=cores for best NUMA behavior
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/* ============================================================================
 * HYBRID MPI + OpenMP SOLVER
 * ============================================================================ */

static double compute_checksum_local(const double *u, size_t N)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t p = 0; p < N; p++) {
        sum += u[p];
    }
    return sum;
}

static void init_local_grid(double *u, int Nx, int Ny, int Nz_local,
                             int z_global_start, int rank, int nprocs,
                             int Nz_global)
{
    size_t N_local = (size_t)Nz_local * Ny * Nx;
    
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < N_local; p++) {
        int i = p % Nx;
        int j = (p / Nx) % Ny;
        int k_local = p / (Nx * Ny);
        int k_global = z_global_start + k_local;
        
        if (i == 0 || i == Nx - 1 ||
            j == 0 || j == Ny - 1 ||
            k_global == 0 || k_global == Nz_global - 1) {
            u[p] = 1.0;
        } else {
            u[p] = 0.0;
        }
    }
}

int main(int argc, char **argv)
{
    /* Initialize MPI with THREAD_FUNNELED (only master thread calls MPI) */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ========================================================================
     * PARSE ARGUMENTS (rank 0 broadcasts to all)
     * ======================================================================== */
    int Nx = 0, Ny = 0, Nz = 0, T = 0;
    double alpha = 0.0, beta = 1.0 / 6.0;

    if (rank == 0) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s Nx Ny Nz T [alpha] [beta]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nz = atoi(argv[3]);
        T = atoi(argv[4]);
        alpha = (argc > 5) ? atof(argv[5]) : 0.0;
        beta = (argc > 6) ? atof(argv[6]) : 1.0 / 6.0;

        if (Nx < 3 || Ny < 3 || Nz < 3 || T < 1) {
            fprintf(stderr, "Error: invalid dimensions\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ========================================================================
     * DOMAIN DECOMPOSITION (Z-axis)
     * ======================================================================== */
    int Nz_local = (Nz + nprocs - 1) / nprocs;
    int z_global_start = rank * Nz_local;
    int z_global_end = MIN((rank + 1) * Nz_local, Nz);
    Nz_local = z_global_end - z_global_start;

    int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next_rank = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    size_t N_local = (size_t)Nz_local * Ny * Nx;
    
    double *u = (double *)malloc(N_local * sizeof(double));
    double *un = (double *)malloc(N_local * sizeof(double));

    if (!u || !un) {
        fprintf(stderr, "[rank %d] malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    init_local_grid(u, Nx, Ny, Nz_local, z_global_start, rank, nprocs, Nz);
    memcpy(un, u, N_local * sizeof(double));

    int num_omp_threads = omp_get_max_threads();
    if (rank == 0) {
        printf("[Hybrid] Grid: %d x %d x %d, T=%d\n", Nx, Ny, Nz, T);
        printf("[Hybrid] MPI ranks: %d, OpenMP threads/rank: %d\n", nprocs, num_omp_threads);
    }

    /* ========================================================================
     * MAIN TIME-STEPPING LOOP
     * ======================================================================== */
    double t_start = MPI_Wtime();

    for (int t = 0; t < T; t++) {
        
        /* Allocate halo buffers (master thread) */
        double *halo_send_lo = (double *)malloc(Ny * Nx * sizeof(double));
        double *halo_recv_lo = (double *)malloc(Ny * Nx * sizeof(double));
        double *halo_send_hi = (double *)malloc(Ny * Nx * sizeof(double));
        double *halo_recv_hi = (double *)malloc(Ny * Nx * sizeof(double));

        if (!halo_send_lo || !halo_recv_lo || !halo_send_hi || !halo_recv_hi) {
            fprintf(stderr, "[rank %d] halo allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* PHASE 1: Halo packing (single-threaded, master thread) */
        if (Nz_local >= 3) {
            if (z_global_start > 0) {
                size_t base_1 = (size_t)1 * Ny * Nx;
                memcpy(halo_send_lo, &u[base_1], Ny * Nx * sizeof(double));
            }
            if (z_global_end < Nz) {
                size_t base_hi = (size_t)(Nz_local - 2) * Ny * Nx;
                memcpy(halo_send_hi, &u[base_hi], Ny * Nx * sizeof(double));
            }
        }

        /* PHASE 2: Post non-blocking MPI (single-threaded) */
        MPI_Request recv_req[2];
        MPI_Request send_req[2];

        if (z_global_start > 0) {
            MPI_Irecv(halo_recv_lo, Ny * Nx, MPI_DOUBLE, prev_rank,
                     0, MPI_COMM_WORLD, &recv_req[0]);
        } else {
            recv_req[0] = MPI_REQUEST_NULL;
        }

        if (z_global_end < Nz) {
            MPI_Irecv(halo_recv_hi, Ny * Nx, MPI_DOUBLE, next_rank,
                     1, MPI_COMM_WORLD, &recv_req[1]);
        } else {
            recv_req[1] = MPI_REQUEST_NULL;
        }

        if (z_global_start > 0) {
            MPI_Isend(halo_send_lo, Ny * Nx, MPI_DOUBLE, prev_rank,
                     1, MPI_COMM_WORLD, &send_req[0]);
        } else {
            send_req[0] = MPI_REQUEST_NULL;
        }

        if (z_global_end < Nz) {
            MPI_Isend(halo_send_hi, Ny * Nx, MPI_DOUBLE, next_rank,
                     0, MPI_COMM_WORLD, &send_req[1]);
        } else {
            send_req[1] = MPI_REQUEST_NULL;
        }

        /* PHASE 3: Compute interior slices with OpenMP parallelization
         * All OpenMP threads participate here (no barrier needed internally)
         */
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 2; k < Nz_local - 2; k++) {
            for (int j = 1; j < Ny - 1; j++) {
                for (int i = 1; i < Nx - 1; i++) {
                    size_t idx = (size_t)k * Ny * Nx + j * Nx + i;
                    
                    double center = u[idx];
                    double xi_lo = u[idx - 1];
                    double xi_hi = u[idx + 1];
                    double yj_lo = u[idx - Nx];
                    double yj_hi = u[idx + Nx];
                    double zk_lo = u[idx - (size_t)Nx * Ny];
                    double zk_hi = u[idx + (size_t)Nx * Ny];
                    
                    un[idx] = alpha * center +
                              beta * (xi_lo + xi_hi + yj_lo + yj_hi + zk_lo + zk_hi);
                }
            }
        }

        /* PHASE 4: Wait for halos (single-threaded, implicit barrier at end of omp region) */
        #pragma omp barrier  /* All threads synchronize */

        /* Master thread waits for MPI communication */
        #pragma omp master
        {
            MPI_Status status[4];
            MPI_Waitall(2, recv_req, status);
            MPI_Waitall(2, send_req, status);
        }

        #pragma omp barrier  /* Ensure all threads wait for halo arrival */

        /* PHASE 5: Compute boundary slices with OpenMP
         * Threads can now safely use halo data
         */
        if (Nz_local >= 3) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 1; j < Ny - 1; j++) {
                for (int i = 1; i < Nx - 1; i++) {
                    size_t idx_k1 = (size_t)1 * Ny * Nx + j * Nx + i;
                    
                    double center = u[idx_k1];
                    double xi_lo = u[idx_k1 - 1];
                    double xi_hi = u[idx_k1 + 1];
                    double yj_lo = u[idx_k1 - Nx];
                    double yj_hi = u[idx_k1 + Nx];
                    double zk_lo = (z_global_start == 0) ? u[idx_k1 - (size_t)Nx*Ny]
                                                         : halo_recv_lo[j * Nx + i];
                    double zk_hi = u[idx_k1 + (size_t)Nx * Ny];
                    
                    un[idx_k1] = alpha * center +
                                 beta * (xi_lo + xi_hi + yj_lo + yj_hi + zk_lo + zk_hi);
                }
            }

            if (Nz_local >= 4) {
                #pragma omp parallel for collapse(2) schedule(static)
                for (int j = 1; j < Ny - 1; j++) {
                    for (int i = 1; i < Nx - 1; i++) {
                        int k = Nz_local - 2;
                        size_t idx = (size_t)k * Ny * Nx + j * Nx + i;
                        
                        double center = u[idx];
                        double xi_lo = u[idx - 1];
                        double xi_hi = u[idx + 1];
                        double yj_lo = u[idx - Nx];
                        double yj_hi = u[idx + Nx];
                        double zk_lo = u[idx - (size_t)Nx * Ny];
                        double zk_hi = (z_global_end == Nz) ? u[idx + (size_t)Nx*Ny]
                                                            : halo_recv_hi[j * Nx + i];
                        
                        un[idx] = alpha * center +
                                  beta * (xi_lo + xi_hi + yj_lo + yj_hi + zk_lo + zk_hi);
                    }
                }
            }
        }

        /* PHASE 6: Enforce boundary conditions (OpenMP) */
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < Nz_local; k++) {
            int k_global = z_global_start + k;
            
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    size_t idx = (size_t)k * Ny * Nx + j * Nx + i;
                    
                    if (i == 0 || i == Nx - 1 ||
                        j == 0 || j == Ny - 1 ||
                        k_global == 0 || k_global == Nz - 1) {
                        un[idx] = u[idx];
                    }
                }
            }
        }

        /* Buffer swap */
        double *tmp = u;
        u = un;
        un = tmp;

        free(halo_send_lo);
        free(halo_recv_lo);
        free(halo_send_hi);
        free(halo_recv_hi);
    }

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    /* ========================================================================
     * CHECKSUM REDUCTION
     * ======================================================================== */
    double local_sum = compute_checksum_local(u, N_local);
    double global_sum = 0.0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[Hybrid] checksum=%.15e time=%.6f s\n", global_sum, elapsed);
        printf("[Hybrid] GFLOPS estimate: %.2f\n",
               7.0 * (Nx - 2) * (Ny - 2) * (Nz - 2) * T / 1e9 / elapsed);
    }

    free(u);
    free(un);

    MPI_Finalize();
    return 0;
}
