#include <malloc.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cuPrintf.cu"

#define MAX_BLKS 65535
#define MAX_THRDS_P_BLK 512

struct Matrix
{
    unsigned int rows;
    unsigned int cols;
    float *elems;
};

__device__ float MAX(float op1, float op2)
{
  return ((op1<op2) ? op2 : op1);
}

__device__ float MIN(float op1, float op2)
{
  return ((op1>op2) ? op2 : op1);
}

__host__ int ciel(float value)
{
  float mantissa = value - (int)value;
  return ((int)value + (mantissa==0 ? 0 : 1));
}

/**
 * First lanuch of this kernel is such that number is blocks equals number of rows
 * Each row is then divided by multiple threads
 * Thus, the variables 'tid' , 'PerThreadLimit' and 'Block_ColLimit' are computed in below device function
 * max_offset variable is essentially used to prevent off range access in memory.
 * Once each thread knows the range of coloumns it should process in a given row. It computes min and max locally
 * and places it in shared memory space 'threadResults[]'. This array is later used for running
 * reduction process on the local minimums and local maximums to get row wise minimum and maximum
 * Second call of the kernel is made with row wise mins and maxs to get a global minimum and global maximum
 */
__global__ void MinMax(Matrix mat, float* localMin, float* localMax, int ColsPerThread)
{
  extern __shared__ float threadResults[];
  
  int max_offset = blockDim.x;   
  int tid = blockIdx.x * mat.cols + threadIdx.x * ColsPerThread;
  int PerThreadLimit = tid + ColsPerThread;
  int Block_ColLimit = (blockIdx.x+1) * mat.cols;
 
  float max, min;

  int reduceDim = blockDim.x;
  int i = reduceDim/2;
  int start = threadIdx.x;
  int op2;

  if( tid < Block_ColLimit )
  {
    min = mat.elems[tid];
    max = mat.elems[tid];
    while( tid < PerThreadLimit )
    {
      min = MIN(mat.elems[tid],min);
      max = MAX(mat.elems[tid],max);
      tid++;
    }
    threadResults[threadIdx.x] = min;
    threadResults[max_offset+threadIdx.x] = max;
    __syncthreads();
    
    //process results in threadResults in a binary fashion 
    while(i!=0)
    {
      if( threadIdx.x < i )
      {
        if( reduceDim%2 != 0 )
        {
          reduceDim--;
          if( threadIdx.x == 0 )
          {
            op2 = start + reduceDim;
            threadResults[start] = MIN(threadResults[start],threadResults[op2]);
            threadResults[max_offset+start] = MAX(threadResults[max_offset+start],threadResults[max_offset+op2]);
          }
        }
        op2 = start + i;
        threadResults[start] = MIN(threadResults[start],threadResults[op2]);
        threadResults[max_offset+start] = MAX(threadResults[max_offset+start],threadResults[max_offset+op2]);
      }
      __syncthreads();
      reduceDim = reduceDim/2;
      i = i/2;
    }
    if( threadIdx.x == 0 )
    {
      localMin[blockIdx.x] = threadResults[start];
      localMax[blockIdx.x] = threadResults[max_offset+start];
    }
  }
}

/**
 * Program requires two numerical inputs: Matrix dimensions
 * Maximum dimension allowed in any direction is MAX_BLKS = 65535
 */
int main(int argc, char* argv[])
{
 if(argc == 3)
 {
  int RowDim = atoi(argv[1]);
  int ColDim = atoi(argv[2]);
  if( !(RowDim <= MAX_BLKS) )
  {
    printf("Please pass a matrix of row dimension not exceeding 65535\n");
    return -1;
  }
  if( !(ColDim <= MAX_BLKS) )
  {
    printf("Please pass a matrix of coloumn dimension not exceeding 65535\n");
    return -1;
  }

  int iter;
  int mn 		= RowDim * ColDim;
  size_t size 	= RowDim * ColDim * sizeof(float);
  size_t RSIZE 	= RowDim * sizeof(float);

  float 		*host_localMin, *host_localMax, host_Minimum, host_Maximum;
  Matrix 		host_mat, temp_mat;
  Matrix 		dev_mat;
  float 		*dev_localMin, *dev_localMax;
  cudaEvent_t 	startminmax1, stopminmax1, startminmax2, stopminmax2;
  cudaEvent_t	startT1, startT2, startT3, startT4, stopT1, stopT2, stopT3, stopT4, startl, stopl;
  float 		timeT1, timeT2, timeT3, timeT4, timeminmax1, timeminmax2, host_min, host_max;

  /**
   * Check whether Row Dimension < Maximum # of blocks allowed:
   * if YES then we launch blocks equal to number of rows
   * if NO then we take dividend of RowDim/MAX_BlOCKS as number 
   * and launch MAX_BLOCKS blocks
   */
  int blocksPerGrid = RowDim;
  int half = ColDim/2 + ( ColDim%2==0 ? 0 : 1 );
  int ColsPerThread = ( half/MAX_THRDS_P_BLK==0 ? 2 : ciel((float)ColDim/MAX_THRDS_P_BLK) );
  int threadsPerBlock = ( half < MAX_THRDS_P_BLK ? half : ceil((float)ColDim/ColsPerThread) );
  
  /* for max and min; Set of mins followed by Set of maxs*/
  int perBlockSize = 2*threadsPerBlock*sizeof(float);   
  printf("Blocks per Grid: %d\nThreads per Block: %d\nColoumns per Thread: %d\n", blocksPerGrid, threadsPerBlock, ColsPerThread);
  srand(time(NULL));
  
  /* Allocate host memory to prepare data */
  host_mat.elems = (float*)malloc(size);
  host_localMin = (float*)malloc(RSIZE);
  host_localMax = (float*)malloc(RSIZE);  
  host_mat.rows = RowDim;
  host_mat.cols = ColDim;
  for( iter=0; iter < mn; iter++ )
    host_mat.elems[iter] = (rand()%100+1.2)*(rand()%100+4.2);
  
  /* Allocate device memory, GPU memory */
  cudaMalloc((void**)&dev_mat.elems, size);
  cudaMalloc((void**)&dev_localMax, RSIZE);
  cudaMalloc((void**)&dev_localMin, RSIZE);
  
  cudaEventCreate( &startminmax1 );
  cudaEventCreate( &stopminmax1 );
  cudaEventCreate( &startminmax2 );
  cudaEventCreate( &stopminmax2 );
  cudaEventCreate( &startT1 ); cudaEventCreate( &startT2 );
  cudaEventCreate( &stopT1 ); cudaEventCreate( &stopT2 );
  cudaEventCreate( &startT3 ); cudaEventCreate( &startT4 );
  cudaEventCreate( &stopT3 ); cudaEventCreate( &stopT4 );
  cudaEventCreate( &startl ); cudaEventCreate( &stopl );

  /* Copy data from host to device */
  dev_mat.rows = host_mat.rows;
  dev_mat.cols = host_mat.cols;
  cudaEventRecord( startT1, 0 );
  cudaMemcpy(dev_mat.elems, host_mat.elems, size, cudaMemcpyHostToDevice );
  cudaEventRecord( stopT1, 0 );
   
  cudaEventRecord( startminmax1, 0 );
  cudaPrintfInit();
  /* This kernel will compute min and max along each row */
  MinMax<<<blocksPerGrid, threadsPerBlock, perBlockSize>>>(dev_mat, dev_localMin, dev_localMax, ColsPerThread);
  cudaEventRecord( stopminmax1, 0 );
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();

   /* copt result from device to host */
  cudaEventRecord( startT2, 0 );
  cudaMemcpy( host_localMax, dev_localMax, RSIZE, cudaMemcpyDeviceToHost );
  cudaMemcpy( host_localMin, dev_localMin, RSIZE, cudaMemcpyDeviceToHost );
  cudaEventRecord( stopT2, 0 );

  /* A second Kernel call follows that computes the global min and max from previous kernel call results */
  cudaFree(dev_mat.elems);
  cudaFree(dev_localMax);
  cudaFree(dev_localMin);
  cudaMalloc((void**)&dev_mat.elems, 2*RSIZE);
  cudaMalloc((void**)&dev_localMax, sizeof(float));
  cudaMalloc((void**)&dev_localMin, sizeof(float)); 
  
  temp_mat.elems = (float*)malloc(2*RSIZE);
  temp_mat.rows = 1;
  temp_mat.cols = 2*RowDim;
  mn = temp_mat.rows*temp_mat.cols;
  for( iter=0; iter < mn; iter++ )
  {
    if( iter < RowDim )
      temp_mat.elems[iter] = host_localMin[iter];
    else
      temp_mat.elems[iter] = host_localMax[iter-RowDim];     
  }
  dev_mat.rows = temp_mat.rows;
  dev_mat.cols = temp_mat.cols;
  cudaEventRecord( startT3, 0 );
  cudaMemcpy(dev_mat.elems, temp_mat.elems, 2*RSIZE, cudaMemcpyHostToDevice );
  cudaEventRecord( stopT3, 0 );
  
  blocksPerGrid = temp_mat.rows;
  half = temp_mat.cols/2 + ( temp_mat.cols%2==0 ? 0 : 1 );
  ColsPerThread = ( half/MAX_THRDS_P_BLK==0 ? 2 : ciel((float)(temp_mat.cols+MAX_THRDS_P_BLK-1)/MAX_THRDS_P_BLK) );
  threadsPerBlock = ( half < MAX_THRDS_P_BLK ? half : ceil((float)temp_mat.cols/ColsPerThread) );
  perBlockSize = 2*threadsPerBlock*sizeof(float);
  
  cudaEventRecord( startminmax2, 0 );
  cudaPrintfInit();
  MinMax<<<blocksPerGrid,threadsPerBlock, perBlockSize>>>(dev_mat, dev_localMin, dev_localMax, ColsPerThread);
  cudaEventRecord( stopminmax2, 0 );
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();

   /* copt result from device to host */
  cudaEventRecord( startT4, 0 );
  cudaMemcpy( &host_Maximum, dev_localMax, sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( &host_Minimum, dev_localMin, sizeof(float), cudaMemcpyDeviceToHost );
  cudaEventRecord( stopT4, 0 );

  cudaEventSynchronize( stopT1 );
  cudaEventSynchronize( stopT2 );
  cudaEventSynchronize( stopT3 );
  cudaEventSynchronize( stopT4 );
  cudaEventElapsedTime( &timeT1, startT1, stopT1 );
  cudaEventElapsedTime( &timeT2, startT2, stopT2 );
  cudaEventElapsedTime( &timeT3, startT3, stopT3 );
  cudaEventElapsedTime( &timeT4, startT4, stopT4 );
  cudaEventSynchronize( stopminmax1 );
  cudaEventSynchronize( stopminmax2 );
  cudaEventElapsedTime( &timeminmax1, startminmax1, stopminmax1 );
  cudaEventElapsedTime( &timeminmax2, startminmax2, stopminmax2 );

  printf("Final Device(Min, Max)=((%f, %f)\n", host_Minimum, host_Maximum);
  printf("Data transfer time : %f ms\n", timeT1+timeT2+timeT3+timeT4);
  printf("Computation time   : %f ms\n", timeminmax1+timeminmax2);
 
  /* Compute on host for comparison */
  cudaEventRecord( startminmax1, 0 );
  host_max = host_mat.elems[0];
  host_min = host_mat.elems[0];
  mn = host_mat.rows * host_mat.cols;
  //printf("Matrix given is : ");
  for (iter = 0; iter < mn; iter++)
  { 
     /*if( iter%ColDim == 0)
       printf("\n");
     printf(" %.1f ",host_mat.elems[iter]);*/
     if( host_mat.elems[iter] > host_max )
       host_max = host_mat.elems[iter];
     if( host_mat.elems[iter] < host_min )
       host_min = host_mat.elems[iter];
  }
  printf("\n");
  cudaEventRecord( stopminmax1, 0 );

  cudaEventSynchronize( stopminmax1 ); 
  cudaEventElapsedTime( &timeminmax1, startminmax1, stopminmax1 );
  printf("Host(Min, Max)=((%f, %f)\n", host_min, host_max);
  printf("Time (computation done on host only): %f ms\n", timeminmax1);
  
  /* clear all memory  */
  cudaFree(dev_mat.elems);
  cudaFree(dev_localMax);
  cudaFree(dev_localMin);
 cudaEventDestroy( startT1 );
  cudaEventDestroy( stopT1 );
  cudaEventDestroy( startT2 );
  cudaEventDestroy( stopT2 );
  cudaEventDestroy( startT3 );
  cudaEventDestroy( stopT3 );
  cudaEventDestroy( startT4 );
  cudaEventDestroy( stopT4 );
  cudaEventDestroy( startminmax1 );
  cudaEventDestroy( stopminmax1 );
  cudaEventDestroy( startminmax2 );
  cudaEventDestroy( stopminmax2 );
  free(host_mat.elems);
  free(temp_mat.elems);
  free(host_localMax);
  free(host_localMin);
 }
 else
 {
   printf("Please provide atleast one argument.\n");
   return 1;
 }
 return 0;
}
