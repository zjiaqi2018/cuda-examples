#include <Kokkos_Core.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "timer.h"

using ValueType = double;

void
print_stat(const std::string &method,
           const int          size,
           const double       flops,
           const double       secs,
           const double       fnorm)
{
  const double bytes = 1.0 * sizeof(double) * size * size;

  std::cout << method << " N= " << size
            << " mem: " << std::round(bytes / 1024 / 1024 * 100) / 100 << "mb"
            << " Mflops/s: " << std::round(flops / secs / 1000000)
            << " mem: " << std::round(bytes / secs / 1024 / 1024) << "mb/s"
            << " took " << secs << "s"
            << " Frobenius norm: " << fnorm << std::endl
            << std::endl;
}

// source: git@github.com:CUDA-Tutorial/CodeSamples.git

// Declare a GPU-visible floating point variable in global memory.
__device__ ValueType dResult;

__global__ void
reduceAtomicGlobal(const ValueType *input, int N)
{
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  /*
  Since all blocks must have the same number of threads,
  we may have to launch more threads than there are
  inputs. Superfluous threads should not try to read
  from the input (out of bounds access!)
  */
  if (id < N)
    {
      const ValueType t = fabs(input[id]);
      atomicAdd(&dResult, t * t);
    }
}


/*
 First improvement: shared memory is much faster than global
 memory. Each block can accumulate partial results in isolated
 block-wide visible memory. This relieves the contention on
 a single global variable that all threads want access to.
*/
__global__ void
reduceAtomicShared(const ValueType *input, int N)
{
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  // Declare a shared float for each block
  __shared__ ValueType x;

  // Only one thread should initialize this shared value
  if (threadIdx.x == 0)
    {
      x = 0.0;
    }
  /*
  Before we continue, we must ensure that all threads
  can see this update (initialization) by thread 0
  */
  __syncthreads();

  /*
  Every thread in the block adds its input to the
  shared variable of the block.
  */
  if (id < N)
    {
      const ValueType t = fabs(input[id]);
      atomicAdd(&x, t * t);
    }

  // Wait until all threads have done their part
  __syncthreads();

  /*
  Once they are all done, only one thread must add
  the block's partial result to the global variable.
  */
  if (threadIdx.x == 0)
    {
      atomicAdd(&dResult, x);
    }
}


/*
 Second improvement: choosing a more suitable algorithm.
 We can exploit the fact that the GPU is massively parallel
 and come up with a fitting procedure that uses multiple
 iterations. In each iteration, threads accumulate partial
 results from the previous iteration. Before, the contented
 accesses to one location forced the GPU to perform updates
 sequentially O(N). Now, each thread can access its own,
 exclusive shared variable in each iteration in parallel,
 giving an effective runtime that is closer to O(log N).
*/
template <unsigned int BLOCK_SIZE>
__global__ void
reduceShared(const ValueType *input, int N)
{
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  /*
  Use a larger shared memory region so that each
  thread can store its own partial results
  */
  __shared__ ValueType data[BLOCK_SIZE];
  /*
  Use a new strategy to handle superfluous threads.
  To make sure they stay alive and can help with
  the reduction, threads without an input simply
  produce a '0', which has no effect on the result.
  */

  const ValueType t = fabs(input[id]);
  data[threadIdx.x] = (id < N ? t * t : 0);

  /*
  log N iterations to complete. In each step, a thread
  accumulates two partial values to form the input for
  the next iteration. The sum of all partial results
  eventually yields the full result of the reduction.
  */
  for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
      /*
      In each iteration, we must make sure that all
      threads are done writing the updates of the
      previous iteration / the initialization.
      */
      __syncthreads();
      if (threadIdx.x < s)
        data[threadIdx.x] += data[threadIdx.x + s];
    }

  /*
  Note: thread 0 is the last thread to combine two
  partial results, and the one who writes to global
  memory, therefore no synchronization is required
  after the last iteration.
  */
  if (threadIdx.x == 0)
    atomicAdd(&dResult, data[0]);
}

// A very simple matrix class
class matrix
{
public:
  matrix(int N)
    : N(N)
  {
    data.resize(N * N);
  }

  double &
  operator()(int i, int j)
  {
    return data[j * N + i];
  }

  int                 N;
  std::vector<double> data;
};

void
fill(matrix &mat)
{
  for (int i = 0; i < mat.N; ++i)
    for (int j = 0; j < mat.N; ++j)
      {
        mat(i, j) = 1.0 * (i + j);
      }
}

// compute the Frobenius norm
double
frob(matrix &mat)
{
  double result = 0.0;
  for (int i = 0; i < mat.N; ++i)
    for (int j = 0; j < mat.N; ++j)
      {
        // try what happens when you do mat(j,i) instead
        double t = std::abs(mat(j, i));
        result += t * t;
      }
  return std::sqrt(result);
}

void
test_host(int size, int nrepeat)
{
  matrix mat(size);
  fill(mat);
  Timer<> clock; // Timer is defined in timer.h in the same folder

  // Let's time it:
  clock.tick();
  // int runs = 1000000/size;
  double r = 0.0;
  for (int i = 0; i < nrepeat; ++i)
    r = frob(mat);
  clock.tock();

  const double secs  = clock.duration().count() / (nrepeat * 1000.0);
  const double flops = 3.0 * size * size;
  print_stat("CPU", size, flops, secs, r);
}

void
test_kokkos(const int M, const int N, const int nrepeat)
{
  using MemSpace = Kokkos::CudaSpace;

  using ExecSpace    = MemSpace::execution_space;
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  // Allocate Matrix A on device.
  typedef Kokkos::View<double *, Kokkos::LayoutLeft, MemSpace>  ViewVectorType;
  typedef Kokkos::View<double **, Kokkos::LayoutLeft, MemSpace> ViewMatrixType;
  ViewMatrixType                                                A("A", M, N);

  // Create host mirrors of device views.
  ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view(A);

  matrix mat(N);
  fill(mat);
  // Initialize A matrix on host.
  for (int j = 0; j < N; ++j)
    {
      for (int i = 0; i < M; ++i)
        {
          h_A(j, i) = mat(j, i);
        }
    }

  // Deep copy host views to device views.
  Kokkos::deep_copy(A, h_A);

  // Timer products.
  Kokkos::Timer timer;

  double result;

  for (int repeat = 0; repeat < nrepeat; repeat++)
    {
      result = 0;

      Kokkos::parallel_reduce(
        "Frobenius norm",
        range_policy(0, N),
        KOKKOS_LAMBDA(int j, double &update) {
          double temp2 = 0;

          for (int i = 0; i < M; ++i)
            {
              double t = fabs(A(j, i));
              temp2 += t * t;
            }

          update += temp2;
        },
        result);
    }

  result = sqrt(result);

  // Calculate time.
  double       time         = timer.seconds();
  const double average_time = time / (nrepeat * 1.0);

  const double flops = 3.0 * M * N;
  print_stat("Kokkos", N, flops, average_time, result);

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  double Gbytes = 1.0e-9 * double(sizeof(double) * (M * N));

  //   // Print results (problem size, time and bandwidth in GB/s).
  //   printf( "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s )
  //   bandwidth( %g GB/s )\n",
  //           N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );
}



void
test_cuda(const int M, const int N, const int nrepeat)
{
  ValueType *mat, *frob_out;
  ValueType *d_mat, *dfrob_out;

  // Allocate host memory
  const int matrix_length = M * N;

  mat      = (ValueType *)malloc(sizeof(ValueType) * matrix_length);
  frob_out = (ValueType *)malloc(sizeof(ValueType) * matrix_length);

  // Initialize host arrays
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < M; j++)
        {
          mat[j * N + i] = 1.0 * (i + j);
        }
    }


  // Allocate device memory
  cudaMalloc((void **)&d_mat, sizeof(ValueType) * matrix_length);
  cudaMalloc((void **)&dfrob_out, sizeof(ValueType) * matrix_length);

  // Transfer data from host to device memory
  cudaMemcpy(d_mat,
             mat,
             sizeof(ValueType) * matrix_length,
             cudaMemcpyHostToDevice);

  // Executing kernel
  constexpr int block_size = 256;
  int           grid_size  = ((matrix_length + block_size) / block_size);
  double        Gbytes     = 1.0e-9 * double(sizeof(double) * (matrix_length));

  {
    ValueType reduce_result = 0.;

    Timer<> clock;
    clock.tick();
    for (int repeat = 0; repeat < nrepeat; repeat++)
      {
        cudaMemcpyToSymbol(dResult, &reduce_result, sizeof(ValueType));
        reduceAtomicGlobal<<<grid_size, block_size>>>(d_mat, matrix_length);
      }
    cudaDeviceSynchronize();
    clock.tock();
    const double secs = clock.duration().count() / (nrepeat * 1000.0);

    cudaMemcpyFromSymbol(&reduce_result, dResult, sizeof(ValueType));

    const double flops = 3.0 * M * N;
    ;
    print_stat("ReduceAtomicGlobal", N, flops, secs, sqrt(reduce_result));
    // printf("Reduce Frobenius norm = %lf\n",
    // std::sqrt(reduce_result/(nrepeat*1.0))); printf("  N( %d ) M( %d )
    // nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n", N,
    // M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time);
  }

  {
    ValueType reduce_result = 0.;
    Timer<>   clock;
    clock.tick();
    for (int repeat = 0; repeat < nrepeat; repeat++)
      {
        cudaMemcpyToSymbol(dResult, &reduce_result, sizeof(ValueType));
        reduceAtomicShared<<<grid_size, block_size>>>(d_mat, matrix_length);
        // reduceShared<block_size><<<grid_size, block_size>>>(d_mat,
        // matrix_length);
      }
    cudaDeviceSynchronize();
    clock.tock();
    const double secs = clock.duration().count() / (nrepeat * 1000.0);

    cudaMemcpyFromSymbol(&reduce_result, dResult, sizeof(ValueType));

    const double flops = 3.0 * M * N;
    ;
    print_stat("ReduceAtomicShared", N, flops, secs, sqrt(reduce_result));
    // printf("Reduce Frobenius norm = %lf\n",
    // std::sqrt(reduce_result/(nrepeat*1.0))); printf("  N( %d ) M( %d )
    // nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n", N,
    // M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time);
  }

  {
    ValueType reduce_result = 0.;
    Timer<>   clock;
    clock.tick();
    for (int repeat = 0; repeat < nrepeat; repeat++)
      {
        cudaMemcpyToSymbol(dResult, &reduce_result, sizeof(ValueType));
        reduceShared<block_size>
          <<<grid_size, block_size>>>(d_mat, matrix_length);
      }
    cudaDeviceSynchronize();

    clock.tock();

    const double secs = clock.duration().count() / 1000.0 / nrepeat;

    cudaMemcpyFromSymbol(&reduce_result, dResult, sizeof(ValueType));

    const double flops = 3.0 * M * N;
    ;
    print_stat("ActomicShare", N, flops, secs, sqrt(reduce_result));
    // printf("Reduce Frobenius norm = %lf\n",
    // std::sqrt(reduce_result/(nrepeat*1.0))); printf("  N( %d ) M( %d )
    // nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n", N,
    // M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time);
  }

  // Deallocate device memory
  cudaFree(d_mat);
  cudaFree(dfrob_out);

  // Deallocate host memory
  free(mat);
  free(frob_out);
}

int
main(int argc, char *argv[])
{
  const int N       = 10000;
  const int M       = N;
  const int nrepeat = 100; // number of repeats of the test

  // for (int i=1;i<3;++i)
  {
    int size = N;
    test_host(size, nrepeat);
  }

  {
    Kokkos::initialize(argc, argv);
    test_kokkos(M, N, nrepeat);
    Kokkos::finalize();
  }

  {
    test_cuda(M, N, nrepeat);
  }

  return 0;
}
