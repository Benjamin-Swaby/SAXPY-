#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 20us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */


__global__ void fill(float *a , float x)
{
   int index =  blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
   
   for(int i = index; i < N; i += stride)
   {
       a[i] = x;
   }
}

__global__ void saxpy(float *x, float *y, float *result)
{
   int index = threadIdx.x + blockIdx.x * blockDim.x;
   int stride = blockDim.x * gridDim.x;
   
   for(int i = index; i < N; i += stride)
   {
       result[i] = 2 * x[i] + y[i];
   }
}


cudaDeviceProp getDetails(int deviceId)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    return props;
}



#define multi 20
int main()
{
    float *x, *y, *result;
    int size = N * sizeof (int); // The total number of bytes per vector
    
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props = getDetails(deviceId);
    

    cudaMallocManaged(&result, size);
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);
    
    cudaMemPrefetchAsync(result, size, deviceId);
    cudaMemPrefetchAsync(x, size, deviceId);
    cudaMemPrefetchAsync(y, size, deviceId);
	
    int threads_per_block = 512;
    printf("number of sms :%d \n", props.multiProcessorCount);
    int number_of_blocks = props.multiProcessorCount * multi;
	
	cudaStream_t stream_result; cudaStreamCreate(&stream_result);
	cudaStream_t stream_x; cudaStreamCreate(&stream_x);
	cudaStream_t stream_y; cudaStreamCreate(&stream_y);

    fill<<<threads_per_block,number_of_blocks, 0 , stream_result>>>(result, 0.0); //result
    fill<<<threads_per_block,number_of_blocks, 0 , stream_x>>>(x, 1.0); // array x 
    fill<<<threads_per_block,number_of_blocks, 0 , stream_y>>>(y, 2.0); // array y
	
	cudaStreamDestroy(stream_result); cudaStreamDestroy(stream_x); cudaStreamDestroy(stream_y);	
    

    //error variables
    cudaError_t addVectorsErr;
    cudaError_t asyncErr;

    saxpy <<< number_of_blocks, threads_per_block >>> ( x, y, result );
    cudaMemPrefetchAsync(result, size, cudaCpuDeviceId);
    
	
	addVectorsErr = cudaGetLastError();

    if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    
    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("y[%d] = %f, ", i, result[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("y[%d] = %f, ", i, result[i]);
    printf ("\n");

    cudaFree( result ); cudaFree( x ); cudaFree( y );
}
