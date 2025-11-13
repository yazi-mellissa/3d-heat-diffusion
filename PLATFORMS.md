# Platform-Specific Build Guide

## Linux (Recommended for HPC)

### Ubuntu / Debian

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y libopenmpi-dev openmpi-bin  # or libmpich-dev
sudo apt-get install -y valgrind  # For profiling

# Build
make cpu
make cuda  # If NVIDIA GPU present
```

### Red Hat / CentOS / Rocky Linux

```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y openmpi-devel  # or mpich-devel
sudo yum install -y valgrind

make cpu
```

### Fedora

```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y openmpi-devel
sudo dnf install -y valgrind

make cpu
```

---

## macOS

### Using Homebrew

```bash
# Install compiler & tools
brew install gcc@13          # GCC with OpenMP
brew install open-mpi        # MPI

# Verify
gcc-13 --version
mpicc --version

# Build (may need to adjust CC/MPICC in Makefile)
CC=gcc-13 MPICC=mpicc make cpu
```

### Using MacPorts

```bash
sudo port install gcc13 +universal
sudo port install openmpi +gcc13

make cpu
```

### Note for M1/M2 Mac Users

```bash
# May need native compilation instead of Rosetta
# Check Homebrew configuration for ARM64
arch  # Should show arm64e (native) not i386

# Verify OpenMP works
export OMP_NUM_THREADS=4
./bin/heat_openmp 32 32 32 10
```

---

## Windows (WSL2 or Cygwin)

### WSL2 (Windows Subsystem for Linux) - Recommended

1. **Install WSL2:**
   ```powershell
   # In PowerShell as Administrator
   wsl --install -d Ubuntu-22.04
   ```

2. **Inside WSL2 terminal:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libopenmpi-dev
   cd /mnt/c/Users/PICOS/Downloads/HPC_3d_heat_diffusion
   make cpu
   ```

### Cygwin (Alternative)

1. **Install Cygwin with packages:**
   - gcc-core, gcc-g++
   - libopenmpi-devel
   - libopenmp-devel

2. **Build:**
   ```bash
   make cpu
   ```

### Native Windows (MinGW-w64)

```bash
# Download MinGW-w64 from mingw-w64.org
# Add to PATH
# Compile manually (Makefile may need adjustment)
gcc -O3 -march=native -fopenmp -I./include src/heat3d_openmp.c -o bin/heat_openmp.exe -lm
```

---

## HPC Clusters

### SLURM-based Cluster

```bash
# Load modules
module load gcc     # or intel/2022
module load openmpi # or mpich
module load cuda    # If GPU nodes available

# Build
make all

# Run interactively
srun -N 1 -n 4 ./bin/heat_openmp 256 256 256 100

# Or submit job
cat > job.slurm <<EOF
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

module load gcc openmpi

export OMP_NUM_THREADS=4
mpirun -np 8 ./bin/heat_hybrid 512 512 512 50
EOF

sbatch job.slurm
```

### PBS/Torque Cluster

```bash
# Build (may need module system)
make all

# Job submission
cat > job.pbs <<EOF
#!/bin/bash
#PBS -l nodes=2:ppn=16
#PBS -l walltime=00:10:00

module load intel openmpi

export OMP_NUM_THREADS=8
mpirun -np 4 ./bin/heat_hybrid 512 512 512 50
EOF

qsub job.pbs
```

### LSF Cluster (IBM)

```bash
cat > job.lsf <<EOF
#!/bin/bash
#BSUB -n 16
#BSUB -W 00:10
#BSUB -q normal

module load gcc openmpi

export OMP_NUM_THREADS=8
jsrun -n 4 -r 1 -c 8 ./bin/heat_hybrid 512 512 512 50
EOF

bsub < job.lsf
```

---

## GPU Systems

### NVIDIA GPU with CUDA

#### Docker Container (Easiest)

```bash
# Use NVIDIA CUDA base image
docker run --rm --gpus all -it nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Inside container:
apt-get update && apt-get install -y build-essential
cd /home && git clone <repo>
make cuda
./bin/heat_cuda_opt 256 256 256 100
```

#### Local CUDA Installation

```bash
# Download from developer.nvidia.com/cuda-toolkit
# Install to /usr/local/cuda (default)

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

nvcc --version  # Verify

make cuda
```

#### Conda/Mamba Environment

```bash
conda create -n heat3d -c conda-forge python=3.10 cudatoolkit=11.8 gcc gxx
conda activate heat3d

make cuda
```

---

## Compiler Optimization Flags

### Intel Compiler (icc, icpc)

```bash
# Modify Makefile:
CC = icc
CFLAGS = -O3 -march=core-avx2 -xHost -qopenmp
MPICC = mpiicc

make cpu
```

### NVIDIA HPC SDK (nvc, nccUPLICATE)

```bash
# Alternative to traditional GCC
CC = nvc
CFLAGS = -O3 -march=native -fopenmp -stdpar=gpu
MPICC = mpicc  # wrapper

make cpu
```

### ARM Compiler (Raspberry Pi, A64FX)

```bash
CC = armclang
CFLAGS = -O3 -march=native -fopenmp

make cpu
```

---

## MPI Implementation Specifics

### OpenMPI

```bash
# Check if installed
which mpicc
mpicc --showme

# Common issues: ABI mismatch
# Solution: Rebuild with same compiler as OpenMPI

# Test
mpirun -np 2 ./bin/heat_mpi 32 32 32 10
```

### MPICH

```bash
which mpicc
mpicc -v

# May be faster than OpenMPI on some systems
# Otherwise API-compatible
```

### Intel MPI

```bash
source /opt/intel/oneapi/mpi/latest/env/vars.sh
make cpu
```

---

## Troubleshooting Build Issues

### Issue: "gcc: command not found"

```bash
# Linux
sudo apt-get install build-essential

# macOS
brew install gcc@13

# Windows (WSL2)
sudo apt-get install build-essential
```

### Issue: "mpicc: command not found"

```bash
# Linux
sudo apt-get install libopenmpi-dev

# macOS
brew install open-mpi

# Then verify
mpicc --version
```

### Issue: "nvcc: command not found"

```bash
# Download CUDA Toolkit from nvidia.com
# Then
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

### Issue: "error: conflicting types for 'struct timespec'"

```bash
# On some systems, add compile flag
CFLAGS += -D_POSIX_C_SOURCE=199309L
```

### Issue: CUDA architecture mismatch

```bash
# Find your GPU architecture
nvidia-smi  # Lists GPU model

# Modify NVCCFLAGS in Makefile
NVCCFLAGS = -O3 -arch=sm_86  # For RTX 3080 (sm_86)
                              # For V100 (sm_70)
                              # For A100 (sm_80)

make cuda
```

---

## Continuous Integration (GitHub Actions)

```yaml
# .github/workflows/build.yml
name: Build and Test

on: [push, pull_request]

jobs:
  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libopenmpi-dev
      - name: Build
        run: make cpu
      - name: Test
        run: bash tests/validate.sh

  build-cuda:
    runs-on: self-hosted  # Requires GPU runner
    steps:
      - uses: actions/checkout@v2
      - name: Build CUDA
        run: make cuda
      - name: Test CUDA
        run: ./bin/heat_cuda_opt 64 64 64 10
```

---

## Docker Build

```dockerfile
# Dockerfile for complete environment
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenmpi-dev \
    wget \
    git

WORKDIR /workspace

COPY . /workspace/heat3d/
WORKDIR /workspace/heat3d

RUN make all

# Run tests
CMD ["bash", "tests/validate.sh"]
```

Build and run:
```bash
docker build -t heat3d:latest .
docker run --rm --gpus all heat3d:latest
```

---

## Makefile Customization

### For your specific system:

```makefile
# Edit these variables
CC = gcc-13          # Your compiler
MPICC = mpicc        # Your MPI compiler
NVCC = nvcc          # Your CUDA compiler

CFLAGS = -O3 -march=native -Wall -Wextra
CFLAGS_OMP = -fopenmp

# Add platform-specific flags
ifeq ($(shell uname),Darwin)
    # macOS
    CFLAGS += -march=native
else ifeq ($(shell uname),Linux)
    # Linux
    CFLAGS += -march=native
endif

# Custom targets
debug:
	$(CC) -g -O0 -fopenmp src/heat3d_openmp.c -o bin/heat_openmp_debug
	gdb ./bin/heat_openmp_debug

profile:
	$(CC) -O3 -fopenmp -pg src/heat3d_openmp.c -o bin/heat_openmp_prof
	./bin/heat_openmp_prof 64 64 64 10
	gprof bin/heat_openmp_prof gmon.out > profile.txt
```

---

## Performance Optimization Flags

### Aggressive optimization (careful!)

```makefile
CFLAGS = -O3 -march=native -flto -ffast-math
# -flto: link-time optimization
# -ffast-math: may sacrifice numerical accuracy
```

### Conservative (reproducible results)

```makefile
CFLAGS = -O2 -march=haswell
# -O2: safer than -O3
# -march=haswell: assume Haswell or newer (widely available)
```

---

**Document Version:** 1.0  
**Last Updated:** November 13, 2025
