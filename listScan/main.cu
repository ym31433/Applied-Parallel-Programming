// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  //load input to shared memory
  __shared__ float T[BLOCK_SIZE];
  int id = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  T[threadIdx.x] = (id < len)? input[id]: 0;
  __syncthreads();
  //pre scan
  int stride = 1;
  while(stride < BLOCK_SIZE) {
    id = (threadIdx.x+1)*stride*2-1;
    if(id < BLOCK_SIZE) {
      T[id] += T[id-stride];
    }
    stride *= 2;
    __syncthreads();
  }
  //post scan
  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    id = (threadIdx.x+1)*stride*2-1;
    if(id+stride < BLOCK_SIZE) {
      T[id+stride] += T[id];
    }
    stride /= 2;
    __syncthreads();
  }
  //write to output
  id = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  if(id < len) {
    output[id] = T[threadIdx.x];
  }
}
__global__ void blockSumInit(float *output, int len, float *blockSum, int blockSumLen) {
  int id = blockIdx.x*BLOCK_SIZE + threadIdx.x; // the number of threads launched is numBlocks
  int id_out = (id+1)*BLOCK_SIZE-1;
  if(id < blockSumLen) {
    blockSum[id] = (id_out < len)? output[id_out]: output[len-1];
  }
}
__global__ void addBack(float *output, int len, float *scanSum, int scanLen) {
  int id = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  if(blockIdx.x > 0 && blockIdx.x < scanLen && id < len) {
    output[id] += scanSum[blockIdx.x - 1];
  }
}

void scan(float *input, float *output, int len, int numBlocks) {
  //wbLog(TRACE, "Len: ", len);
  //wbLog(TRACE, "NumBlocks: ", numBlocks);
  scan<<<dim3((unsigned)numBlocks, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(input, output, len);
  //cudaDeviceSynchronize();
  if(len <= BLOCK_SIZE) return;
  float *deviceBlockSum;
  float *deviceScanBlockSum;
  cudaMalloc((void **)&deviceBlockSum, numBlocks*sizeof(float));
  cudaMalloc((void **)&deviceScanBlockSum, numBlocks*sizeof(float));
  blockSumInit<<<dim3((unsigned)((numBlocks-1)/BLOCK_SIZE+1), 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(output, len, deviceBlockSum, numBlocks);
  /*debug//
  float *hostBlockSum = (float *)malloc(numBlocks*sizeof(float));
  cudaMemcpy(hostBlockSum, deviceBlockSum, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);
  wbLog(TRACE, "BlockSum[0]: ", hostBlockSum[0]);
  //end_debug*/
  //recursive
  scan(deviceBlockSum, deviceScanBlockSum, numBlocks, (numBlocks-1)/BLOCK_SIZE+1);
  /*debug//
  float *hostScanBlockSum = (float *)malloc(numBlocks*sizeof(float));
  cudaMemcpy(hostScanBlockSum, deviceScanBlockSum, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);
  wbLog(TRACE, "ScanBlockSum[0]: ", hostScanBlockSum[0]);
  //end_debug*/
  
  addBack<<<dim3((unsigned)numBlocks, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(output, len, deviceScanBlockSum, numBlocks);
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int numBlocks = (numElements-1)/BLOCK_SIZE+1;
  //dim3 dimGrid(numBlocks, 1, 1);
  //dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  /*debug//
  wbLog(TRACE, "input[511]: ", hostInput[511]);
  wbLog(TRACE, "input[512]: ", hostInput[512]);
  //end_debug*/
  
  scan(deviceInput, deviceOutput, numElements, numBlocks);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  
  /*debug//
  wbLog(TRACE, "output[511]: ", hostOutput[511]);
  wbLog(TRACE, "output[512]: ", hostOutput[512]);
  //end_debug*/

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
