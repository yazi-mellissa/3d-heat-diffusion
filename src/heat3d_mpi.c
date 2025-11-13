/*
 * heat3d_mpi.c - 3D Heat Diffusion Solver using MPI
 *
 * Jacobi explicit stencil (7-point) with MPI domain decomposition
 * Domain decomposition: along Z-axis (slabs), each rank handles Nz_local layers
 * Halo exchange: non-blocking MPI_Isend/MPI_Irecv for k=1 and k=Nz_local-2 boundaries
 *
 * Compilation:
 *   mpicc -O3 -march=native heat3d_mpi.c -o heat_mpi -I../include -lm
 *
 * Execution:
 *   mpirun -np 4 ./heat_mpi 256 256 256 100
 *   mpirun -np 8 --bind-to socket ./heat_mpi 512 512 512 50
 *
 * Notes:
 * - Global grid size Nx x Ny x Nz is decomposed along Z among ranks
 * - Each rank allocates Nz_local slices plus 2 halo layers (front & back)
 * - Non-blocking halo exchange: MPI_Isend/MPI_Irecv, then compute interior,
 *   then MPI_Waitall before computing boundary slices
 * - Global boundaries (k=0 and k=Nz-1) handled only by rank 0 and rank-nprocs-1
 * - Checksum reduction: MPI_Reduce sum to rank 0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/* ============================================================================
 * MPI 3D HEAT DIFFUSION SOLVER
 * ============================================================================ */

/**
 * Compute checksum
 */
static double compute_checksum_local(const double *u, size_t N)
{
    double sum = 0.0;
    for (size_t p = 0; p < N; p++) {
        sum += u[p];
    }
    return sum;
}

/**
 * Initialize local grid with Dirichlet boundary conditions
 * 
 * @param u          local array (with halos): size Nz_local*Ny*Nx
 * @param Nx, Ny     global grid dimensions
 * @param Nz_local   local z-dimension (excluding halos)
 * @param z_global_start  global z-index where this rank's domain starts
 * @param rank, nprocs  MPI rank and number of processes
 */
static void init_local_grid(double *u, int Nx, int Ny, int Nz_local,
                             int z_global_start, int rank, int nprocs,
                             int Nz_global)
{
    size_t N_local = (size_t)Nz_local * Ny * Nx;
    
    for (size_t p = 0; p < N_local; p++) {
        int i = p % Nx;
        int j = (p / Nx) % Ny;
        int k_local = p / (Nx * Ny);
        int k_global = z_global_start + k_local;
        
        /* Dirichlet BC: boundaries fixed at 1.0 */
        if (i == 0 || i == Nx - 1 ||
            j == 0 || j == Ny - 1 ||
            k_global == 0 || k_global == Nz_global - 1) {
            u[p] = 1.0;
        } else {
            u[p] = 0.0;
        }
    }
}

/**
 * Main solver
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ========================================================================
     * PARSE ARGUMENTS (broadcast from rank 0)
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
            fprintf(stderr, "Error: Nx, Ny, Nz >= 3, T >= 1\n");
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
     * COMPUTE LOCAL DOMAIN DECOMPOSITION (along Z)
     * ======================================================================== */
    int Nz_local = (Nz + nprocs - 1) / nprocs;
    int z_global_start = rank * Nz_local;
    int z_global_end = MIN((rank + 1) * Nz_local, Nz);
    Nz_local = z_global_end - z_global_start;

    int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next_rank = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    size_t N_local = (size_t)Nz_local * Ny * Nx;
    
    /* Allocate with halos (not used here, but could be extended for future) */
    double *u = (double *)malloc(N_local * sizeof(double));
    double *un = (double *)malloc(N_local * sizeof(double));

    if (!u || !un) {
        fprintf(stderr, "[rank %d] Error: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Initialize */
    init_local_grid(u, Nx, Ny, Nz_local, z_global_start, rank, nprocs, Nz);
    memcpy(un, u, N_local * sizeof(double));

    if (rank == 0) {
        printf("[MPI] Grid: %d x %d x %d, T=%d, nprocs=%d\n", Nx, Ny, Nz, T, nprocs);
        printf("[MPI] Local Nz per rank: %d (with remainder)\n", Nz / nprocs);
    }

    /* ========================================================================
     * MAIN TIME-STEPPING LOOP
     * ======================================================================== */
    double t_start = MPI_Wtime();

    for (int t = 0; t < T; t++) {
        
        /* Allocate for halo exchange
         * For simplicity, we use contiguous 2D slices (j,i) for send/recv
         */
        double *halo_send_lo = (double *)malloc(Ny * Nx * sizeof(double));
        double *halo_recv_lo = (double *)malloc(Ny * Nx * sizeof(double));
        double *halo_send_hi = (double *)malloc(Ny * Nx * sizeof(double));
        double *halo_recv_hi = (double *)malloc(Ny * Nx * sizeof(double));

        if (!halo_send_lo || !halo_recv_lo || !halo_send_hi || !halo_recv_hi) {
            fprintf(stderr, "[rank %d] Error: halo allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* PHASE 1: Pack and post non-blocking receives/sends
         * 
         * Halo layers:
         *  - k=1 layer (second from bottom): send to prev_rank, receive from prev_rank
         *  - k=Nz_local-2 layer (second from top): send to next_rank, receive from next_rank
         */
        if (Nz_local >= 3) {
            /* Pack k=1 layer (send to prev) */
            if (z_global_start > 0) {  /* Not on global boundary */
                size_t base_1 = (size_t)1 * Ny * Nx;
                memcpy(halo_send_lo, &u[base_1], Ny * Nx * sizeof(double));
            }

            /* Pack k=Nz_local-2 layer (send to next) */
            if (z_global_end < Nz) {  /* Not on global boundary */
                size_t base_hi = (size_t)(Nz_local - 2) * Ny * Nx;
                memcpy(halo_send_hi, &u[base_hi], Ny * Nx * sizeof(double));
            }
        }

        /* Post non-blocking receives */
        MPI_Request recv_req[2];
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

        /* Post non-blocking sends */
        MPI_Request send_req[2];
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

        /* PHASE 2: Compute interior slices (k in [2, Nz_local-2)) 
         * These don't depend on halos, so can compute in parallel with communication
         */
        for (int k = 2; k < Nz_local - 2; k++) {
            for (int j = 1; j < Ny - 1; j++) {
                for (int i = 1; i < Nx - 1; i++) {
                    size_t idx = (size_t)k * Ny * Nx + j * Nx + i;
                    
                    double center = u[idx];
                    double xi_lo = u[idx - 1];
                    double xi_hi = u[idx + 1];
                    double yj_lo = u[idx - Nx];
                    double yj_hi = u[idx + Nx];
                    double zk_lo = u[idx - Nx * Ny];
                    double zk_hi = u[idx + Nx * Ny];
                    
                    un[idx] = alpha * center +
                              beta * (xi_lo + xi_hi + yj_lo + yj_hi + zk_lo + zk_hi);
                }
            }
        }

        /* PHASE 3: Wait for halo arrival */
        MPI_Status status[4];
        MPI_Waitall(2, recv_req, status);
        MPI_Waitall(2, send_req, status);

        /* PHASE 4: Compute boundary slices using received halos
         * 
         * k=1 layer needs halo from k=0 (received from prev or use boundary condition)
         * k=Nz_local-2 layer needs halo from k=Nz_local-1 (received from next or boundary)
         */
        if (Nz_local >= 3) {
            /* k=1 layer */
            for (int j = 1; j < Ny - 1; j++) {
                for (int i = 1; i < Nx - 1; i++) {
                    size_t idx = (size_t)1 * Ny * Nx + j * Nx + i;
                    
                    double center = u[idx];
                    double xi_lo = u[idx - 1];
                    double xi_hi = u[idx + 1];
                    double yj_lo = u[idx - Nx];
                    double yj_hi = u[idx + Nx];
                    double zk_lo = (z_global_start == 0) ? u[idx - Nx * Ny] : halo_recv_lo[j * Nx + i];
                    double zk_hi = u[idx + Nx * Ny];
                    
                    un[idx] = alpha * center +
                              beta * (xi_lo + xi_hi + yj_lo + yj_hi + zk_lo + zk_hi);
                }
            }

            /* k=Nz_local-2 layer */
            if (Nz_local >= 4) {
                int k = Nz_local - 2;
                for (int j = 1; j < Ny - 1; j++) {
                    for (int i = 1; i < Nx - 1; i++) {
                        size_t idx = (size_t)k * Ny * Nx + j * Nx + i;
                        
                        double center = u[idx];
                        double xi_lo = u[idx - 1];
                        double xi_hi = u[idx + 1];
                        double yj_lo = u[idx - Nx];
                        double yj_hi = u[idx + Nx];
                        double zk_lo = u[idx - Nx * Ny];
                        double zk_hi = (z_global_end == Nz) ? u[idx + Nx * Ny] : halo_recv_hi[j * Nx + i];
                        
                        un[idx] = alpha * center +
                                  beta * (xi_lo + xi_hi + yj_lo + yj_hi + zk_lo + zk_hi);
                    }
                }
            }
        }

        /* Enforce Dirichlet boundary conditions on interior (non-global-boundary) cells */
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

        /* Swap buffers */
        double *tmp = u;
        u = un;
        un = tmp;

        /* Cleanup halo buffers */
        free(halo_send_lo);
        free(halo_recv_lo);
        free(halo_send_hi);
        free(halo_recv_hi);
    }

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    /* ========================================================================
     * COMPUTE LOCAL CHECKSUM & REDUCE
     * ======================================================================== */
    double local_sum = compute_checksum_local(u, N_local);
    double global_sum = 0.0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[MPI] checksum=%.15e time=%.6f s\n", global_sum, elapsed);
        printf("[MPI] GFLOPS estimate: %.2f\n",
               7.0 * (Nx - 2) * (Ny - 2) * (Nz - 2) * T / 1e9 / elapsed);
    }

    /* Cleanup */
    free(u);
    free(un);

    MPI_Finalize();
    return 0;
}
