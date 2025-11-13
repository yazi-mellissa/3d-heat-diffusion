# Quick Start Guide

## Installation & Setup

### 1. Check Prerequisites

**For CPU versions (OpenMP, pthreads, MPI, Hybrid):**
```bash
# Check compiler
gcc --version          # GCC 7.0+
which mpicc            # MPI compiler (OpenMPI, MPICH, Intel MPI, etc.)
pkg-config --cflags openmp  # OpenMP support
```

**For GPU versions (CUDA):**
```bash
# Check NVIDIA GPU and CUDA toolkit
nvidia-smi
nvcc --version         # CUDA 11.0+
```

### 2. Clone/Navigate to Project

```bash
cd /path/to/HPC_3d_heat_diffusion
ls -la
# You should see: Makefile, README.md, TECHNICAL_NOTES.md, include/, src/, tests/
```

### 3. Build

```bash
# Build all CPU versions
make cpu

# Build only OpenMP
make bin/heat_openmp

# Build CUDA versions (requires NVIDIA CUDA)
make cuda

# Build everything
make all

# Clean
make clean
```

---

## Running Your First Test

### Small Correctness Test (32³, T=10)

```bash
# OpenMP (easiest to start)
OMP_NUM_THREADS=4 ./bin/heat_openmp 32 32 32 10

# Expected output:
# [OpenMP] Grid: 32 x 32 x 32, T=10, alpha=0.00e+00, beta=1.67e-01
# [OpenMP] Threads: 4, Memory: 0.00 GB
# [OpenMP] checksum=9.216000e+02 time=0.000123 s
# [OpenMP] GFLOPS estimate: 5.43
```

### Validate All Implementations

```bash
# Small test on tiny grid
bash tests/validate.sh

# Manually run all variants on same problem
Nx=32; Ny=32; Nz=32; T=10

echo "=== OpenMP ==="
OMP_NUM_THREADS=2 ./bin/heat_openmp $Nx $Ny $Nz $T

echo "=== pthreads ==="
./bin/heat_pthreads $Nx $Ny $Nz $T 2

echo "=== MPI (1 process) ==="
mpirun -np 1 ./bin/heat_mpi $Nx $Ny $Nz $T

echo "=== Hybrid (1 rank, 2 threads) ==="
OMP_NUM_THREADS=2 mpirun -np 1 ./bin/heat_hybrid $Nx $Ny $Nz $T
```

**All checksums should match (up to floating-point precision).**

---

## Common Use Cases

### Scenario 1: Shared-Memory Single Node (Laptop/Workstation)

**Best choice:** OpenMP

```bash
# Find number of cores
nproc

# Run OpenMP with all available threads
OMP_NUM_THREADS=$(nproc) ./bin/heat_openmp 256 256 256 100

# Or explicitly
OMP_NUM_THREADS=8 ./bin/heat_openmp 512 512 512 50
```

### Scenario 2: Multi-Socket System (Dual-Socket Server)

**Best choices:** Hybrid MPI+OpenMP or pthreads with pinning

```bash
# Hybrid: 2 MPI ranks (one per socket), 8 OpenMP threads each
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
mpirun -np 2 --bind-to socket ./bin/heat_hybrid 256 256 512 100

# Or pthreads with NUMA pinning
numactl --cpubind=0 ./bin/heat_pthreads 256 256 128 100 8 &
numactl --cpubind=1 ./bin/heat_pthreads 256 256 128 100 8 &
wait
```

### Scenario 3: Multi-Node HPC Cluster

**Best choice:** MPI or Hybrid MPI+OpenMP

```bash
# Pure MPI: 4 processes (one per socket across 2 nodes)
mpirun -np 4 -hostfile hostfile --bind-to socket ./bin/heat_mpi 512 512 512 100

# Hybrid: 2 ranks per node (4 nodes), 8 threads per rank
OMP_NUM_THREADS=8 mpirun -np 8 -npernode 2 ./bin/heat_hybrid 512 512 1024 100
```

### Scenario 4: GPU-Accelerated Machine

**Best choice:** CUDA optimized (or hybrid with CUDA)

```bash
# CUDA naive (baseline)
./bin/heat_cuda_naive 256 256 256 100

# CUDA optimized (10-50× faster)
./bin/heat_cuda_opt 512 512 512 100

# Check GPU memory usage
nvidia-smi

# Profile with Nsight Compute
ncu --set full ./bin/heat_cuda_opt 128 128 128 10
```

---

## Performance Tuning Tips

### OpenMP

```bash
# 1. Use all cores
export OMP_NUM_THREADS=$(nproc)

# 2. Pin threads to cores (avoid NUMA migration)
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# 3. Disable dynamic teams
export OMP_DYNAMIC=false

# 4. Monitor thread affinity
export OMP_DISPLAY_AFFINITY=true
./bin/heat_openmp 128 128 128 10 2>&1 | head -20
```

### MPI

```bash
# 1. Bind ranks to sockets
mpirun -np 4 --bind-to socket ./bin/heat_mpi 256 256 256 100

# 2. Enable process pinning (MPICH)
MPIR_CVAR_ALLOC_MASK=all mpirun -np 4 ./bin/heat_mpi 256 256 256 100

# 3. Check network performance
mpirun -np 2 ./bin/heat_mpi 64 64 64 100  # Small grid to see communication overhead

# 4. Adjust message packing (tuning file)
# See OpenMPI tuning documentation for --mca parameters
```

### Hybrid MPI+OpenMP

```bash
# Avoid oversubscription:
# If CPU has 16 cores total:
#   Option A: 2 ranks × 8 threads each
#   Option B: 4 ranks × 4 threads each
#   NOT: 8 ranks × 8 threads (16 × 8 = 128 threads!)

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
mpirun -np 2 --bind-to socket ./bin/heat_hybrid 256 256 512 100
```

### CUDA

```bash
# 1. Check available GPU memory
nvidia-smi | grep "Memory-Usage"

# 2. Monitor power & temperature
nvidia-smi -l 1

# 3. Profile memory transfers
nvprof --metrics all ./bin/heat_cuda_opt 64 64 64 10

# 4. Adjust tile size if needed (TECHNICAL_NOTES.md)
# Edit TILE_X, TILE_Y in heat3d_cuda_optimized.cu and rebuild
```

---

## Memory Size Estimation

```bash
# Grid memory = 2 buffers × Nx × Ny × Nz × 8 bytes (double)
# Example: 256 × 256 × 256 grid
# Memory = 2 × 256³ × 8 bytes ≈ 2 GB

# Calculate before running:
python3 -c "Nx=512; Ny=512; Nz=512; print(f'Memory: {2*Nx*Ny*Nz*8/1e9:.2f} GB')"

# Available memory check:
free -h        # Linux
Get-ComputerInfo | Select TotalPhysicalMemory  # Windows
```

---

## Troubleshooting

### "heat_openmp: command not found"

```bash
# Did you build?
make all

# Is bin/ in PATH?
./bin/heat_openmp 32 32 32 10

# Or add to PATH
export PATH=$PATH:$(pwd)/bin
heat_openmp 32 32 32 10
```

### "gcc: command not found"

```bash
# Install GCC
sudo apt-get install build-essential  # Debian/Ubuntu
brew install gcc                      # macOS
```

### "mpicc: command not found"

```bash
# Install MPI library
sudo apt-get install libopenmpi-dev   # Debian/Ubuntu (OpenMPI)
sudo apt-get install libmpich-dev     # Debian/Ubuntu (MPICH)
brew install open-mpi                # macOS
```

### "error: CUDA/cutorch not found"

```bash
# Install CUDA Toolkit from nvidia.com
# Set environment variables
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

### Mismatch in checksums

```bash
# Expected: all implementations give same checksum
# If different:

# 1. Check input parameters match
./bin/heat_openmp 32 32 32 10
./bin/heat_pthreads 32 32 32 10 2

# 2. Check floating-point precision
# Small differences (<1e-10) are OK due to rounding

# 3. Verify correctness with tiny grid
./bin/heat_openmp 5 5 5 1
# For 5³ grid, only one interior cell u[2,2,2] should be updated
# u_new[2,2,2] = 0 + (1/6)*6 = 1.0 (all 6 neighbors = 1.0)
```

### Out of Memory

```bash
# Reduce grid size
./bin/heat_openmp 128 128 128 100  # Instead of 512³

# Check memory usage during execution
watch -n 1 'ps aux | grep heat_openmp'

# Or for GPU
watch -n 1 nvidia-smi
```

### Slow MPI performance

```bash
# Check if communication dominates:
# Run same problem with 1 process vs 2 processes

mpirun -np 1 ./bin/heat_mpi 256 256 128 100
mpirun -np 2 ./bin/heat_mpi 256 256 128 100

# If time increases significantly: communication overhead
# Solution: Use larger grids or hybrid approach

# Profile MPI communication
mpirun -np 2 --profile=all ./bin/heat_mpi 128 128 128 10
```

---

## Benchmarking Template

Save as `bench.sh`:

```bash
#!/bin/bash

echo "=== 3D Heat Diffusion Benchmark Suite ==="

# Parameters
NX=${1:-256}
NY=${2:-256}
NZ=${3:-256}
T=${4:-100}

echo "Grid: ${NX}x${NY}x${NZ}, Time steps: ${T}"
echo ""

# OpenMP scaling
echo "=== OpenMP Scaling ==="
for threads in 1 2 4 8; do
    echo -n "Threads=$threads: "
    OMP_NUM_THREADS=$threads ./bin/heat_openmp $NX $NY $NZ $T 2>&1 | grep "time="
done

# MPI scaling
echo ""
echo "=== MPI Scaling ==="
for procs in 1 2 4; do
    echo -n "Procs=$procs: "
    mpirun -np $procs ./bin/heat_mpi $NX $NY $NZ $T 2>&1 | grep "time="
done

# GPU (if available)
if [ -f ./bin/heat_cuda_opt ]; then
    echo ""
    echo "=== GPU Performance ==="
    echo -n "CUDA optimized: "
    ./bin/heat_cuda_opt $NX $NY $NZ $T 2>&1 | grep "time="
fi

echo ""
echo "=== Benchmark Complete ==="
```

```bash
bash bench.sh 256 256 256 100
```

---

## Next Steps

1. **Read README.md** for detailed documentation
2. **Review TECHNICAL_NOTES.md** for implementation details
3. **Try different implementations** on your hardware
4. **Modify problem parameters** to find bottlenecks
5. **Profile with** perf, nvprof, or Nsight for detailed metrics
6. **Adapt code** to your specific problem (different boundary conditions, etc.)

---

**Happy computing!** 🚀
