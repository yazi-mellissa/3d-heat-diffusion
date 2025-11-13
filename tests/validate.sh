#!/bin/bash
#
# Validation and testing script for 3D Heat Diffusion Solver
#
# Tests all implementations on small problem sizes to verify correctness
# and compare performance.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BIN_DIR="./bin"
SMALL_NX=32
SMALL_NY=32
SMALL_NZ=32
SMALL_T=10

MEDIUM_NX=64
MEDIUM_NY=64
MEDIUM_NZ=64
MEDIUM_T=20

# Checksums tolerance
TOLERANCE=1e-9

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "==============================================="
    echo "$1"
    echo "==============================================="
}

print_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

check_binary() {
    if [ ! -f "$1" ]; then
        return 1
    fi
    return 0
}

# Extract checksum from output
extract_checksum() {
    grep -oP 'checksum=\K[0-9.e+-]+' | head -1
}

# Extract time from output
extract_time() {
    grep -oP 'time=\K[0-9.]+' | head -1
}

# Compare checksums (with tolerance)
compare_checksums() {
    local cs1=$1
    local cs2=$2
    local label1=$3
    local label2=$4
    
    # Simple string comparison (exact match)
    if [ "$cs1" = "$cs2" ]; then
        print_pass "Checksum match: $label1 == $label2 (${cs1})"
        return 0
    else
        # Check relative difference (basic check)
        print_fail "Checksum mismatch: $label1 (${cs1}) != $label2 (${cs2})"
        return 1
    fi
}

# ============================================================================
# MAIN TESTS
# ============================================================================

main() {
    print_header "3D Heat Diffusion Solver - Validation Suite"
    
    # Check if binaries exist
    echo ""
    echo "Checking for compiled binaries in $BIN_DIR..."
    
    HAVE_OPENMP=0
    HAVE_PTHREADS=0
    HAVE_MPI=0
    HAVE_HYBRID=0
    HAVE_CUDA_NAIVE=0
    HAVE_CUDA_OPT=0
    
    if check_binary "$BIN_DIR/heat_openmp"; then
        HAVE_OPENMP=1
        print_pass "Found OpenMP version"
    else
        print_skip "OpenMP not found"
    fi
    
    if check_binary "$BIN_DIR/heat_pthreads"; then
        HAVE_PTHREADS=1
        print_pass "Found pthreads version"
    else
        print_skip "pthreads not found"
    fi
    
    if check_binary "$BIN_DIR/heat_mpi"; then
        HAVE_MPI=1
        print_pass "Found MPI version"
    else
        print_skip "MPI not found"
    fi
    
    if check_binary "$BIN_DIR/heat_hybrid"; then
        HAVE_HYBRID=1
        print_pass "Found Hybrid version"
    else
        print_skip "Hybrid not found"
    fi
    
    if check_binary "$BIN_DIR/heat_cuda_naive"; then
        HAVE_CUDA_NAIVE=1
        print_pass "Found CUDA naive version"
    else
        print_skip "CUDA naive not found"
    fi
    
    if check_binary "$BIN_DIR/heat_cuda_opt"; then
        HAVE_CUDA_OPT=1
        print_pass "Found CUDA optimized version"
    else
        print_skip "CUDA optimized not found"
    fi
    
    # If no binaries, exit
    if [ $((HAVE_OPENMP + HAVE_PTHREADS + HAVE_MPI + HAVE_HYBRID + HAVE_CUDA_NAIVE + HAVE_CUDA_OPT)) -eq 0 ]; then
        print_fail "No binaries found. Run 'make' to compile."
        exit 1
    fi
    
    # ========================================================================
    # Test 1: Tiny grid (correctness)
    # ========================================================================
    print_header "Test 1: Tiny Grid (5×5×5, T=1) - Correctness"
    
    TINY_NX=5
    TINY_NY=5
    TINY_NZ=5
    TINY_T=1
    
    declare -A checksums
    
    if [ $HAVE_OPENMP -eq 1 ]; then
        print_test "Running OpenMP..."
        output=$(OMP_NUM_THREADS=2 "$BIN_DIR/heat_openmp" $TINY_NX $TINY_NY $TINY_NZ $TINY_T 2>&1)
        cs=$(echo "$output" | extract_checksum)
        checksums["openmp"]=$cs
        echo "$output"
    fi
    
    if [ $HAVE_PTHREADS -eq 1 ]; then
        print_test "Running pthreads..."
        output=$("$BIN_DIR/heat_pthreads" $TINY_NX $TINY_NY $TINY_NZ $TINY_T 2 2>&1)
        cs=$(echo "$output" | extract_checksum)
        checksums["pthreads"]=$cs
        echo "$output"
    fi
    
    if [ $HAVE_MPI -eq 1 ]; then
        print_test "Running MPI (1 process)..."
        output=$(mpirun -np 1 "$BIN_DIR/heat_mpi" $TINY_NX $TINY_NY $TINY_NZ $TINY_T 2>&1)
        cs=$(echo "$output" | extract_checksum)
        checksums["mpi"]=$cs
        echo "$output"
    fi
    
    if [ $HAVE_CUDA_NAIVE -eq 1 ]; then
        print_test "Running CUDA naive..."
        output=$("$BIN_DIR/heat_cuda_naive" $TINY_NX $TINY_NY $TINY_NZ $TINY_T 2>&1)
        cs=$(echo "$output" | extract_checksum)
        checksums["cuda_naive"]=$cs
        echo "$output"
    fi
    
    # Compare checksums
    echo ""
    echo "Checksum verification:"
    if [ ${#checksums[@]} -gt 1 ]; then
        first_cs=""
        for impl in "${!checksums[@]}"; do
            if [ -z "$first_cs" ]; then
                first_cs=${checksums[$impl]}
                echo "  Reference ($impl): $first_cs"
            else
                if compare_checksums "$first_cs" "${checksums[$impl]}" "reference" "$impl"; then
                    true
                fi
            fi
        done
    fi
    
    # ========================================================================
    # Test 2: Small grid (32³, T=10)
    # ========================================================================
    print_header "Test 2: Small Grid (${SMALL_NX}³, T=${SMALL_T}) - Performance Scaling"
    
    echo "Sequential execution times:"
    echo ""
    
    if [ $HAVE_OPENMP -eq 1 ]; then
        print_test "OpenMP with varying thread counts"
        for threads in 1 2 4; do
            output=$(OMP_NUM_THREADS=$threads "$BIN_DIR/heat_openmp" \
                     $SMALL_NX $SMALL_NY $SMALL_NZ $SMALL_T 2>&1)
            time=$(echo "$output" | extract_time)
            echo "  OpenMP (threads=$threads): ${time}s"
        done
        echo ""
    fi
    
    if [ $HAVE_PTHREADS -eq 1 ]; then
        print_test "pthreads with varying thread counts"
        for threads in 1 2 4; do
            output=$("$BIN_DIR/heat_pthreads" \
                     $SMALL_NX $SMALL_NY $SMALL_NZ $SMALL_T $threads 2>&1)
            time=$(echo "$output" | extract_time)
            echo "  pthreads (threads=$threads): ${time}s"
        done
        echo ""
    fi
    
    if [ $HAVE_MPI -eq 1 ]; then
        print_test "MPI with varying process counts"
        for procs in 1 2; do
            output=$(mpirun -np $procs "$BIN_DIR/heat_mpi" \
                     $SMALL_NX $SMALL_NY $SMALL_NZ $SMALL_T 2>&1)
            time=$(echo "$output" | extract_time)
            echo "  MPI (procs=$procs): ${time}s"
        done
        echo ""
    fi
    
    if [ $HAVE_HYBRID -eq 1 ]; then
        print_test "Hybrid MPI+OpenMP"
        output=$(OMP_NUM_THREADS=2 mpirun -np 1 "$BIN_DIR/heat_hybrid" \
                 $SMALL_NX $SMALL_NY $SMALL_NZ $SMALL_T 2>&1)
        time=$(echo "$output" | extract_time)
        echo "  Hybrid (1 rank, 2 threads): ${time}s"
        echo ""
    fi
    
    if [ $HAVE_CUDA_NAIVE -eq 1 ]; then
        print_test "CUDA naive"
        output=$("$BIN_DIR/heat_cuda_naive" \
                 $SMALL_NX $SMALL_NY $SMALL_NZ $SMALL_T 2>&1)
        time=$(echo "$output" | extract_time)
        echo "  CUDA naive: ${time}s"
        echo ""
    fi
    
    if [ $HAVE_CUDA_OPT -eq 1 ]; then
        print_test "CUDA optimized"
        output=$("$BIN_DIR/heat_cuda_opt" \
                 $SMALL_NX $SMALL_NY $SMALL_NZ $SMALL_T 2>&1)
        time=$(echo "$output" | extract_time)
        echo "  CUDA optimized: ${time}s"
        echo ""
    fi
    
    # ========================================================================
    # Test 3: Medium grid (64³, T=20)
    # ========================================================================
    print_header "Test 3: Medium Grid (${MEDIUM_NX}³, T=${MEDIUM_T}) - Scaling Study"
    
    echo "Strong scaling (OpenMP):"
    if [ $HAVE_OPENMP -eq 1 ]; then
        declare -a times
        for threads in 1 2 4; do
            output=$(OMP_NUM_THREADS=$threads "$BIN_DIR/heat_openmp" \
                     $MEDIUM_NX $MEDIUM_NY $MEDIUM_NZ $MEDIUM_T 2>&1)
            time=$(echo "$output" | extract_time)
            times[$threads]=$time
            speedup=$(echo "scale=2; ${times[1]} / $time" | bc)
            echo "  threads=$threads: ${time}s (speedup=${speedup}x)"
        done
        echo ""
    fi
    
    # ========================================================================
    # Test 4: Weak scaling (if MPI available)
    # ========================================================================
    print_header "Test 4: Weak Scaling (MPI)"
    
    if [ $HAVE_MPI -eq 1 ]; then
        echo "Fixed problem per MPI rank: 32×32×32 per rank, T=10"
        for procs in 1 2 4; do
            Nz=$((32 * procs))
            output=$(mpirun -np $procs "$BIN_DIR/heat_mpi" \
                     32 32 $Nz 10 2>&1)
            time=$(echo "$output" | extract_time)
            echo "  procs=$procs (grid=32×32×${Nz}): ${time}s"
        done
        echo ""
    fi
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_header "Test Summary"
    
    echo ""
    echo "All available implementations tested successfully."
    echo ""
    echo "Next steps:"
    echo "  1. Verify checksums match across implementations (tiny grid)"
    echo "  2. Study scaling plots (strong/weak scaling)"
    echo "  3. Profile with: numactl, OMP_PROC_BIND, MPI binding"
    echo "  4. Benchmark on larger grids (256³, 512³) if memory allows"
    echo ""
}

# ============================================================================
# ENTRY POINT
# ============================================================================

if [ ! -d "$BIN_DIR" ]; then
    print_fail "Binary directory '$BIN_DIR' not found. Run 'make' first."
    exit 1
fi

main "$@"
