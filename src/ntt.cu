#include <cmath>		/* log2(), pow() */
#include <cstdint>		/* uint64_t */
#include <cstdlib> 		/* malloc() */
#include <iostream>
#include <cooperative_groups.h>
#include "../include/utils.cuh"	/* bit_reverse(), modExp(), modulo() */
#include "../include/ntt.h" 	//INCLUDE HEADER FILE

using namespace std;

__global__ void cuda_ntt_parallel_kernel(uint64_t*,uint64_t*,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t*) ;
void cuda_ntt_parallel(uint64_t* , uint64_t,uint64_t,uint64_t,uint64_t ,uint64_t,uint64_t*) ;

void cpuToGpuMemcpy(uint64_t* h_data,uint64_t* d_data,int size)
{
    cudaError_t err = cudaMemcpy(d_data,h_data,size,cudaMemcpyHostToDevice) ;
    if(err != cudaSuccess)
    {
            fprintf(stderr,"Failed to copy vector from host device! - %s",cudaGetErrorString(err)) ;
            exit(EXIT_FAILURE) ;
    }
}

void gpuToCpuMemcpy(uint64_t* d_data,uint64_t* h_data,int size)
{
    cudaError_t err = cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost) ;
    if(err != cudaSuccess)
    {
            fprintf(stderr,"Failed to copy vector from gpu device! - %s",cudaGetErrorString(err)) ;
            exit(EXIT_FAILURE) ;
    }
    cudaFree(d_data) ;
}


/**
 * Perform an in-place iterative breadth-first decimation-in-time Cooley-Tukey NTT on an input vector and return the result
 *
 * @param vec 	The input vector to be transformed
 * @param n	The size of the input vector
 * @param p	The prime to be used as the modulus of the transformation
 * @param r	The primitive root of the prime
 * @param rev	Whether to perform bit reversal on the input vector
 * @return 	The transformed vector
 */



uint64_t *inPlaceNTT_DIT(uint64_t *vec,uint64_t batchSize,  uint64_t n, uint64_t p, uint64_t r,uint64_t* twiddleFactorArray,bool rev){

	uint64_t *result,*result_cpu;

	uint64_t m, k_, a ;
        uint64_t factor1, factor2 ;
	result = (uint64_t*)malloc(n*batchSize*sizeof(uint64_t));
	result_cpu = (uint64_t*)malloc(n*batchSize*sizeof(uint64_t));

	/*
	if(rev){
		result = bit_reverse(vec, n);
		result_cpu = bit_reverse(vec, n);
	}else{
		for(uint64_t i = 0; i < n; i++){	
			result[i] = vec[i];
			result_cpu[i] = vec[i];
		}
	}
	*/

	for(uint64_t i = 0; i < n*batchSize; i++){
                        result[i] = vec[i];
                        result_cpu[i] = vec[i];
        }


	//CPU implementation
	for(int y=0;y < batchSize;y++){

	   for(uint64_t i = 1; i <= log2(n); i++){ 

		m = pow(2,i);
		k_ = (p - 1)/m;
		a = modExp(r,k_,p);

        
		for(uint64_t j = 0; j < n; j+=m){

			for(uint64_t k = 0; k < m/2; k++){

				factor1 = result_cpu[y*n + j + k];
				factor2 = modulo(modExp(a,k,p)*result_cpu[y*n + j + k + m/2],p);
			
				result_cpu[y*n + j + k] 	= modulo(factor1 + factor2, p);
				result_cpu[y*n + j + k+m/2] 	= modulo(factor1 - factor2, p);

			}
		}
	   }
	}
	//GPU implementation
	cuda_ntt_parallel(result,batchSize,n,p,r,log2(n),twiddleFactorArray) ;

	//Comparison
	bool compCPUGPUResult = compVec(result,result_cpu,batchSize*n,false) ;
	std::cout<<"\nComparing output of cpu and gpu :"<<compCPUGPUResult ;
	return result;

}

void cuda_ntt_parallel(uint64_t* res,uint64_t batchSize,uint64_t n,uint64_t p,uint64_t r,uint64_t log2n,uint64_t* twiddleFactorArray)
{
    uint64_t *cuda_result, *cuda_output  ;
    uint64_t sizeOfRes = batchSize*n*sizeof(uint64_t) ;
    uint64_t *preComputeTFarray ;
    cudaMalloc(&cuda_result,sizeOfRes) ;
    cudaMalloc(&cuda_output,sizeOfRes) ;
    cudaMalloc(&preComputeTFarray,log2(n)*(n/2)*sizeof(uint64_t)) ;
    cpuToGpuMemcpy(res,cuda_result,sizeOfRes) ;
    cpuToGpuMemcpy(twiddleFactorArray,preComputeTFarray,log2(n)*(n/2)*sizeof(uint64_t)) ;

  
    // Number of threads my_kernel will be launched with
    int tpb = 128; // Threads per block
    int bpg = (batchSize*n -1 + tpb)/tpb ; // Blocks per grid

    if(bpg>256)
	  bpg=256;

//    cout<<"bpg: "<<bpg<<endl;
    
    dim3 dimGrid(bpg,1,1) ;
    dim3 dimBlock(tpb,1,1) ;
    void* kernelArgs[] = {
	(void*)&cuda_result, (void*)&cuda_output, (void*)&batchSize, (void*)&n,(void*)&p, (void*)&r, (void*)&log2n,
	(void*)&preComputeTFarray
    } ;
    
    cudaLaunchCooperativeKernel((void*)cuda_ntt_parallel_kernel,dimGrid,dimBlock,kernelArgs) ;
    cudaDeviceSynchronize() ;
    cudaError_t err = cudaGetLastError() ;

	if(err != cudaSuccess)
	{
	    fprintf(stderr,"Issues in running the kernel. (%s)",cudaGetErrorString(err)) ;
            exit(EXIT_FAILURE) ;
	}

    gpuToCpuMemcpy(cuda_output,res,sizeOfRes) ;
    cudaFree(cuda_result) ;
    cudaFree(preComputeTFarray) ;
}

__global__ void cuda_ntt_parallel_kernel(uint64_t* result, uint64_t* output,uint64_t batchSize, uint64_t n,uint64_t p,uint64_t r,uint64_t log2n,uint64_t* twiddleFactorArray)
{
    

    uint64_t mini_batch_size = blockDim.x*gridDim.x/n;
    uint64_t num_mini_batches = (batchSize+mini_batch_size-1)/mini_batch_size;
    uint64_t mini_batch_offset = mini_batch_size*n;

    uint64_t global_idx= blockDim.x*blockIdx.x+threadIdx.x ;
    uint64_t vec_idx = (blockDim.x*blockIdx.x+threadIdx.x)%n;

    uint64_t k ;
    uint64_t factor1,factor2 ;
    uint64_t m = 1;
 
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    uint64_t maxTwiddleCols = n>>1 ;
 
    for(uint64_t i = 0; i < log2n; i++){

        m = m<<1;

        if(vec_idx < n)
        {
            k = vec_idx & (m-1) ;//idx%m ;
            if(k < (m>>1))
            {
		for(int l=0;l<num_mini_batches;l++){
                	factor1 = result[global_idx+mini_batch_offset*l] ;
                	factor2 = modulo(twiddleFactorArray[i*maxTwiddleCols+k]*result[global_idx+mini_batch_offset*l+(m>>1)],p);
                	output[global_idx+mini_batch_offset*l] = modulo(factor1+factor2,p) ;
           	} 
	   }
           else
           {
		for(int l=0;l<num_mini_batches;l++){
                	factor1 = result[global_idx+mini_batch_offset*l - (m>>1)] ;
                	factor2 = modulo(twiddleFactorArray[i*maxTwiddleCols+k-(m>>1)]*result[global_idx+mini_batch_offset*l],p) ;
                	output[global_idx+mini_batch_offset*l] = modulo(factor1-factor2,p) ;
		}
           }
        }
        grid.sync() ;
        if(vec_idx < n)
		for(int l=0;l<num_mini_batches;l++)
                	result[global_idx+mini_batch_offset*l] = output[global_idx+mini_batch_offset*l] ;
        grid.sync();
    }
}
