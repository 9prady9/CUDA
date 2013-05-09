#include <arrayfire.h>
#include <curand_kernel.h>
using namespace af;

#include <stdio.h>
#include <time.h>

typedef unsigned int IntegerType;

// generate millions of random samples
static IntegerType samples = 30e6;
static float g_h_elapsedtime;
static IntegerType g_h_PerThreadLoad;
static IntegerType g_h_BlkCount;
static IntegerType g_h_ThrdCount;
static IntegerType repetitions;
static IntegerType leftOverSize;

const int BLOCKS_PER_SM = 8;
const int DEFAULT_UNWIND_COUNT = 8;
const float g_h_fraction = 0.649161;

IntegerType *g_d_blockCounts;
IntegerType *g_h_output;

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
	{
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/*
  Self-contained code to run each implementation of PI estimation.
  Note that each is generating its own random values, so the
  estimates of PI will differ.
*/
static double pi_cpu()
{
    IntegerType count = 0;
    for (IntegerType i = 0; i < samples; ++i) {
        float x = float(rand()) / RAND_MAX;
        float y = float(rand()) / RAND_MAX;
        if (x*x + y*y < 1)
            count++;
    }
    return 4.0 * count / samples;
}

static double pi_af()
{
    array x = randu(samples,f32), y = randu(samples,f32);
    return 4 * sum<float>(x*x + y*y <= 1) / samples;
}

/**
 * Below kernel is used for finding rough estimation
 * of time taken for one single iteration on the device
 * The resulting estimation is used to find per thread work load
 */
__global__ void opTimeEstimation(IntegerType fSamples)
{
	for(IntegerType i=0;i<fSamples;++i)
	{
		IntegerType seed = i;
		curandState s;
		curand_init(seed, 0, 0, &s);
		float x = curand_uniform(&s);
		float y = curand_uniform(&s);
		bool value = ( x*x + y*y < 1 ? 1 : 0 );
	}
}

static void pi_init_cuda()
{
    /*
      TODO any initialization code you need goes here, e.g. random
      number seeding, cudaMalloc allocations, etc.  Random number
      _generation_ should still go in pi_cuda().
    */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	printf("-------------------------------------------\n");
    printf("Device name: %s\n", prop.name);
	printf("Shared memory per block = %d KB\n",prop.sharedMemPerBlock/1024);
	printf("Multiprocessor count : %d\n", prop.multiProcessorCount);
	printf("Warp size : %d\n", prop.warpSize);
	printf("Max blocks per (x,y,z) dimension =  (%d,%d,%d)\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
	printf("Max threads per (x,y,z) dimension =  (%d,%d,%d)\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
	printf("-------------------------------------------\n");
	g_h_ThrdCount = prop.maxThreadsPerBlock/2;
	/**
	 * consider the following operations as one single task
	 * generate two random numbers
	 * find sum or their squares
	 * compare if less than < 1
	 * estimate time for one such task on device by launching a <<<1,1>>> thread
	 */
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	opTimeEstimation<<<1,1>>>(samples);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&g_h_elapsedtime,start,stop);
	printf ("Time for the kernel: %f ms\n", g_h_elapsedtime);
	g_h_PerThreadLoad = g_h_fraction*samples/(g_h_elapsedtime*1.0);
	printf("Max possible Per thread work load %d \n",g_h_PerThreadLoad);

	g_h_BlkCount = (float)samples/(g_h_ThrdCount*g_h_PerThreadLoad)+0.5f;
	g_h_BlkCount = g_h_BlkCount + (prop.multiProcessorCount - (g_h_BlkCount%prop.multiProcessorCount));
	g_h_PerThreadLoad = (float)samples/(g_h_BlkCount*g_h_ThrdCount)+0.5f;
	samples = g_h_BlkCount*g_h_ThrdCount*g_h_PerThreadLoad;

	printf("Number of blocks : %d\n",g_h_BlkCount);
	printf("Number of threads per block : %d\n",g_h_ThrdCount);
	printf("Per thread load : %d\n",g_h_PerThreadLoad);
	printf("Global array size : %d\n",g_h_BlkCount*sizeof(IntegerType));
	HANDLE_ERROR( cudaMalloc((void**)&g_d_blockCounts, g_h_BlkCount*sizeof(IntegerType)) );
	g_h_output = (IntegerType*)malloc(sizeof(IntegerType));
	
	repetitions = g_h_PerThreadLoad/DEFAULT_UNWIND_COUNT;
	leftOverSize = g_h_PerThreadLoad%DEFAULT_UNWIND_COUNT;
}

__global__ void pointTest(IntegerType* fBlockCounts, IntegerType fNumThreads, IntegerType fPerThreadLoad,
							unsigned long fSeed, const IntegerType fRepeats, const IntegerType fLeftOverSize)
{
	extern __shared__ volatile IntegerType cache[];	
	curandState myState;;
	IntegerType myId = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(fSeed, myId, 0, &myState);
	IntegerType count = 0;

	// unroll the loop DEFAULT_UNWIND_COUNT times
	for(IntegerType unwind_k=0; unwind_k<fRepeats; unwind_k++)
	{
		float x,y;
		/* 8 times */
		x = curand_uniform(&myState); y = curand_uniform(&myState); if((x*x + y*y) < 1.0) count++;
		x = curand_uniform(&myState); y = curand_uniform(&myState); if((x*x + y*y) < 1.0) count++;
		x = curand_uniform(&myState); y = curand_uniform(&myState); if((x*x + y*y) < 1.0) count++;
		x = curand_uniform(&myState); y = curand_uniform(&myState); if((x*x + y*y) < 1.0) count++;
		x = curand_uniform(&myState); y = curand_uniform(&myState); if((x*x + y*y) < 1.0) count++;
		x = curand_uniform(&myState); y = curand_uniform(&myState); if((x*x + y*y) < 1.0) count++;
		x = curand_uniform(&myState); y = curand_uniform(&myState); if((x*x + y*y) < 1.0) count++;
		x = curand_uniform(&myState); y = curand_uniform(&myState); if((x*x + y*y) < 1.0) count++;
	}
	// loop rest over elements
	for(IntegerType leftOver_k=0; leftOver_k<fLeftOverSize; ++leftOver_k)
	{
		float x = curand_uniform(&myState);
		float y = curand_uniform(&myState);
		if((x*x + y*y) < 1.0) count++;
	}
	cache[threadIdx.x] = count;
	__syncthreads();
	// Reduction of this cache.
	while( (fNumThreads>>=1)>0 )
	{
		if(threadIdx.x<fNumThreads)
			cache[threadIdx.x] += cache[threadIdx.x+fNumThreads];
		__syncthreads();
	}
	/*if(threadIdx.x<32)
	{
		cache[threadIdx.x] += cache[threadIdx.x+32];
		cache[threadIdx.x] += cache[threadIdx.x+16];
		cache[threadIdx.x] += cache[threadIdx.x+8];
		cache[threadIdx.x] += cache[threadIdx.x+4];
		cache[threadIdx.x] += cache[threadIdx.x+2];
	}*/
	// End of reduction: thread-id 0 puts in cache[0]
	if(threadIdx.x == 0)
		fBlockCounts[blockIdx.x] = cache[0];
}

__global__ void sumUpBlkCounts(IntegerType* fBlockCounts, IntegerType fSize)
{
	for(IntegerType k=1;k<fSize;++k)
	{
		fBlockCounts[0] += fBlockCounts[k];
	}
}

static double pi_cuda()
{
    /*
      TODO Put your code here.  You can use anything in the CUDA
      Toolkit, including libraries, Thrust, or your own device
      kernels, but do not use ArrayFire functions here.  If you have
      initialization code, see pi_init_cuda().
    */
	pointTest<<<g_h_BlkCount,g_h_ThrdCount,g_h_ThrdCount*sizeof(IntegerType)>>>(g_d_blockCounts,g_h_ThrdCount,g_h_PerThreadLoad,time(NULL),repetitions,leftOverSize);
	HANDLE_ERROR( cudaPeekAtLastError() );
	sumUpBlkCounts<<<1,1>>>(g_d_blockCounts,g_h_BlkCount);
	HANDLE_ERROR( cudaDeviceSynchronize() );
	HANDLE_ERROR( cudaMemcpy(g_h_output,g_d_blockCounts,sizeof(IntegerType),cudaMemcpyDeviceToHost) );
    return 4.0 * g_h_output[0] / samples;
}


// void wrappers for timeit()
static void wrap_cpu()  { pi_cpu();  }
static void wrap_af()   { pi_af();   }
static void wrap_cuda() { pi_cuda(); }


static void experiment(const char *method, double time, double error, double cpu_time)
{
    printf("%10s: %7.5f seconds, error=%.8f", method, time, error);
    if (time > cpu_time)  printf(" ... needs speed!");
    if (error > 1e-3)     printf(" ... needs accuracy!");
    putchar('\n');
}

int main(int argc, char** argv)
{
    try {
        // perform timings and calculate error from reference af::Pi
        info();
        double t_cpu  = timeit(wrap_cpu),  e_cpu  = fabs(af::Pi - pi_cpu());
        double t_af   = timeit(wrap_af),   e_af   = fabs(af::Pi - pi_af());
        pi_init_cuda();
        double t_cuda = timeit(wrap_cuda), e_cuda = fabs(af::Pi - pi_cuda());
		cudaFree(g_d_blockCounts);

        // print results
        experiment("cpu",       t_cpu,  e_cpu,  t_cpu);
        experiment("arrayfire", t_af,   e_af,   t_cpu);
        experiment("cuda",      t_cuda, e_cuda, t_cpu);
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    #ifdef WIN32 // pause in Windows
    if (!(argc == 2 && argv[1][0] == '-')) {
        printf("hit [enter]...");
        getchar();
    }
    #endif
    return 0;
}
