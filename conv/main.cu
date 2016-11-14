#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 8
#define N MASK_WIDTH/2
//@@ Define constant memory for device kernel here
__constant__ float kernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float tile[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
  
  const int idx_out = blockIdx.x*TILE_WIDTH + threadIdx.x;
  const int idy_out = blockIdx.y*TILE_WIDTH + threadIdx.y;
  const int idz_out = blockIdx.z*TILE_WIDTH + threadIdx.z;
  
  if(idx_out < x_size && idy_out < y_size && idz_out < z_size) {
    tile[threadIdx.x][threadIdx.y][threadIdx.z] = input[idz_out*(y_size*x_size) + idy_out*x_size + idx_out];
  }
  
  __syncthreads();
  
  const int idx_in = idx_out - N;
  const int idy_in = idy_out - N;
  const int idz_in = idz_out - N;
  
  const int idx_this_block = blockIdx.x*TILE_WIDTH;
  const int idy_this_block = blockIdx.y*TILE_WIDTH;
  const int idz_this_block = blockIdx.z*TILE_WIDTH;
  
  const int idx_next_block = (blockIdx.x+1)*TILE_WIDTH;
  const int idy_next_block = (blockIdx.y+1)*TILE_WIDTH;
  const int idz_next_block = (blockIdx.z+1)*TILE_WIDTH;
  
  int idx, idy, idz;
  float result = 0.0f;
  
  for(int i = 0; i != MASK_WIDTH; ++i) {
    idx = idx_in+i;
    for(int j = 0; j != MASK_WIDTH; ++j) {
      idy = idy_in+j;
      for(int k = 0; k != MASK_WIDTH; ++k) {
        idz = idz_in+k;
        if(idx >= 0 && idx < x_size &&
           idy >= 0 && idy < y_size &&
           idz >= 0 && idz < z_size) { //check if the index is in input
          if(idx >= idx_this_block && idx < idx_next_block &&
             idy >= idy_this_block && idy < idy_next_block &&
             idz >= idz_this_block && idz < idz_next_block) { //check if the index is in tile
            result += tile[threadIdx.x-N+i][threadIdx.y-N+j][threadIdx.z-N+k] * kernel[i][j][k];
          }
          else {
            result += input[idz*(y_size*x_size) + idy*x_size + idx] * kernel[i][j][k];
          }
        }
      }
    }
  }
  
  if(idx_out < x_size && idy_out < y_size && idz_out < z_size) {
    output[idz_out*(y_size*x_size) + idy_out*x_size + idx_out] = result;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbCheck(cudaMalloc((void**) &deviceInput, z_size*y_size*x_size*sizeof(float)));
  wbCheck(cudaMalloc((void**) &deviceOutput, z_size*y_size*x_size*sizeof(float)));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  //float* hostInputPtr = &hostInput[3];
  wbCheck(cudaMemcpy(deviceInput, &hostInput[3], z_size*y_size*x_size*sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpyToSymbol(kernel, hostKernel, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float)));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid((x_size-1)/TILE_WIDTH+1, (y_size-1)/TILE_WIDTH+1, (z_size-1)/TILE_WIDTH+1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  //float* hostOutputPtr = &hostOutput[3];
  wbCheck(cudaMemcpy(&hostOutput[3], deviceOutput, z_size*y_size*x_size*sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
