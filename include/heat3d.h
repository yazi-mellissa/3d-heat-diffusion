/*
 * heat3d.h - Common utilities and structures for 3D heat diffusion solver
 * 
 * Jacobi stencil (7-point) on Nx x Ny x Nz grid:
 * u_new[i,j,k] = alpha * u[i,j,k] 
 *               + beta * (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k] + u[i,j,k-1] + u[i,j,k+1])
 */

#ifndef HEAT3D_H
#define HEAT3D_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * STRUCTURE FOR PROBLEM PARAMETERS
 * ============================================================================ */
typedef struct {
    int Nx, Ny, Nz;     /* Grid dimensions */
    int T;              /* Number of time steps */
    double alpha;       /* Coefficient for center (typically 0.0) */
    double beta;        /* Coefficient for neighbors (typically 1/6) */
} ProblemParams;

/* ============================================================================
 * INITIALIZATION & UTILITIES
 * ============================================================================ */

/**
 * Initialize 3D grid with Dirichlet boundary conditions
 * Boundaries (i=0, i=Nx-1, j=0, j=Ny-1, k=0, k=Nz-1) = 1.0
 * Interior = 0.0
 * 
 * @param u       pointer to array of size Nx*Ny*Nz
 * @param Nx, Ny, Nz  grid dimensions
 */
static inline void init_grid(double *u, int Nx, int Ny, int Nz)
{
    size_t N = (size_t)Nx * Ny * Nz;
    for (size_t p = 0; p < N; p++) {
        int i = p % Nx;
        int j = (p / Nx) % Ny;
        int k = p / (Nx * Ny);
        
        /* Dirichlet BC: boundaries fixed to 1.0 */
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
 * Compute checksum (sum of all values) for validation
 * 
 * @param u    pointer to array
 * @param N    total number of elements
 * @return     sum of all elements
 */
static inline double compute_checksum(const double *u, size_t N)
{
    double sum = 0.0;
    for (size_t p = 0; p < N; p++) {
        sum += u[p];
    }
    return sum;
}

/**
 * Print grid info and result summary
 */
static inline void print_result(const char *version, double checksum, double time_elapsed)
{
    printf("[%s] checksum=%.15e time=%.6f s\n", version, checksum, time_elapsed);
}

/* ============================================================================
 * TIMING UTILITIES
 * ============================================================================ */

#ifdef _OPENMP
/* Use OpenMP timer if available */
#define GET_TIME() omp_get_wtime()
#else
/* Fallback to clock() */
static inline double get_time_cpu(void)
{
    return (double)clock() / CLOCKS_PER_SEC;
}
#define GET_TIME() get_time_cpu()
#endif

/* ============================================================================
 * STENCIL COMPUTATION INLINE
 * ============================================================================ */

/**
 * Single 7-point stencil update (central differences)
 * Used by all implementations
 */
static inline double stencil_7point(
    const double *u, int Nx, int Ny, int Nz,
    int i, int j, int k,
    double alpha, double beta)
{
    size_t idx = (size_t)k * Ny * Nx + j * Nx + i;
    
    /* Boundary check (should not happen in main loop) */
    if (i <= 0 || i >= Nx - 1 ||
        j <= 0 || j >= Ny - 1 ||
        k <= 0 || k >= Nz - 1) {
        return u[idx];
    }
    
    double center = u[idx];
    double xi_lo  = u[idx - 1];           /* i-1 */
    double xi_hi  = u[idx + 1];           /* i+1 */
    double yj_lo  = u[idx - Nx];          /* j-1 */
    double yj_hi  = u[idx + Nx];          /* j+1 */
    double zk_lo  = u[idx - Nx * Ny];     /* k-1 */
    double zk_hi  = u[idx + Nx * Ny];     /* k+1 */
    
    return alpha * center + beta * (xi_lo + xi_hi + yj_lo + yj_hi + zk_lo + zk_hi);
}

/* ============================================================================
 * VALIDATION & ERROR CHECKING
 * ============================================================================ */

/**
 * Validate two solutions and report relative difference
 */
static inline double relative_error(const double *u1, const double *u2, size_t N)
{
    double sum_sq_diff = 0.0, sum_sq_u1 = 0.0;
    
    for (size_t p = 0; p < N; p++) {
        double diff = u1[p] - u2[p];
        sum_sq_diff += diff * diff;
        sum_sq_u1 += u1[p] * u1[p];
    }
    
    if (sum_sq_u1 < 1e-20) return 0.0;
    return sqrt(sum_sq_diff / sum_sq_u1);
}

#endif /* HEAT3D_H */
