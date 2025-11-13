# Makefile for 3D Heat Diffusion Solver
#
# Supports: OpenMP, pthreads, MPI, CUDA (naive & optimized), Hybrid MPI+OpenMP
#
# Usage:
#   make              # Build all CPU versions
#   make cuda         # Build CUDA versions
#   make all          # Build everything
#   make clean        # Clean build artifacts

CC = gcc
MPICC = mpicc
NVCC = nvcc

CFLAGS = -O3 -march=native
CFLAGS_OMP = -fopenmp
CFLAGS_PTHREADS = -pthread
LDFLAGS = -lm

NVCCFLAGS = -O3 -arch=sm_70
NVCCLDFLAGS = -lm

# Include paths
INCLUDE = -I./include

# Output directory
BINDIR = ./bin
SRCDIR = ./src

# All targets
TARGETS_CPU = $(BINDIR)/heat_openmp $(BINDIR)/heat_pthreads $(BINDIR)/heat_mpi $(BINDIR)/heat_hybrid
TARGETS_CUDA = $(BINDIR)/heat_cuda_naive $(BINDIR)/heat_cuda_opt
ALL_TARGETS = $(TARGETS_CPU) $(TARGETS_CUDA)

# Default: build CPU versions
.PHONY: all cpu cuda clean help

all: cpu cuda

cpu: $(TARGETS_CPU)

cuda: $(TARGETS_CUDA)

help:
	@echo "3D Heat Diffusion Solver - Build Targets"
	@echo "=========================================="
	@echo "make cpu          - Build OpenMP, pthreads, MPI, Hybrid (default)"
	@echo "make cuda         - Build CUDA versions (requires NVIDIA GPU)"
	@echo "make all          - Build all variants"
	@echo "make clean        - Remove build artifacts"
	@echo ""
	@echo "Individual targets:"
	@echo "  make heat_openmp       - OpenMP version"
	@echo "  make heat_pthreads     - pthreads version"
	@echo "  make heat_mpi          - MPI version"
	@echo "  make heat_hybrid       - Hybrid MPI+OpenMP version"
	@echo "  make heat_cuda_naive   - CUDA naive version"
	@echo "  make heat_cuda_opt     - CUDA optimized version"

# Create bin directory if needed
$(BINDIR):
	mkdir -p $(BINDIR)

# ============================================================================
# CPU VERSIONS
# ============================================================================

$(BINDIR)/heat_openmp: $(SRCDIR)/heat3d_openmp.c | $(BINDIR)
	$(CC) $(CFLAGS) $(CFLAGS_OMP) $(INCLUDE) $< -o $@ $(LDFLAGS)
	@echo "[OK] OpenMP version: $@"

$(BINDIR)/heat_pthreads: $(SRCDIR)/heat3d_pthreads.c | $(BINDIR)
	$(CC) $(CFLAGS) $(CFLAGS_PTHREADS) $(INCLUDE) $< -o $@ $(LDFLAGS)
	@echo "[OK] pthreads version: $@"

$(BINDIR)/heat_mpi: $(SRCDIR)/heat3d_mpi.c | $(BINDIR)
	$(MPICC) $(CFLAGS) $(INCLUDE) $< -o $@ $(LDFLAGS)
	@echo "[OK] MPI version: $@"

$(BINDIR)/heat_hybrid: $(SRCDIR)/heat3d_hybrid.c | $(BINDIR)
	$(MPICC) $(CFLAGS) $(CFLAGS_OMP) $(INCLUDE) $< -o $@ $(LDFLAGS)
	@echo "[OK] Hybrid MPI+OpenMP version: $@"

# ============================================================================
# CUDA VERSIONS
# ============================================================================

$(BINDIR)/heat_cuda_naive: $(SRCDIR)/heat3d_cuda_naive.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $< -o $@ $(NVCCLDFLAGS)
	@echo "[OK] CUDA naive version: $@"

$(BINDIR)/heat_cuda_opt: $(SRCDIR)/heat3d_cuda_optimized.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $< -o $@ $(NVCCLDFLAGS)
	@echo "[OK] CUDA optimized version: $@"

# ============================================================================
# CLEAN
# ============================================================================

clean:
	rm -rf $(BINDIR)
	@echo "Cleaned build artifacts"

.PHONY: help
