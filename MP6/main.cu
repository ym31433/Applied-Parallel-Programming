// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
//#define BLOCK_SIZE 1024

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ insert code here
__global__ void hist(float* input, unsigned char* ucharImage, int size, unsigned int* hist){
  unsigned char r, g, b, gray;
  int stride = blockDim.x * gridDim.x;
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ unsigned int private_hist[256];
  
  if(threadIdx.x < 256) {
    private_hist[threadIdx.x] = 0;
  }
  __syncthreads();
  
  while(id < size) {
    r = (unsigned char)(input[id*3]*255);
    g = (unsigned char)(input[id*3+1]*255);
    b = (unsigned char)(input[id*3+2]*255);
    ucharImage[id*3] = r;
    ucharImage[id*3+1] = g;
    ucharImage[id*3+2] = b;
    
    gray = (unsigned char)(0.21*(float)r + 0.71*(float)g + 0.07*(float)b);
    if(gray > 255) gray = 255;
    atomicAdd(&(private_hist[gray]), 1);
    
    id += stride;
  }
  __syncthreads();
  
  if(threadIdx.x < 256) {
    atomicAdd(&(hist[threadIdx.x]), private_hist[threadIdx.x]);
  }
}

__global__ void scan(unsigned int *input, float *output, int len, int size) {
  //load input to shared memory
  __shared__ float T[HISTOGRAM_LENGTH];
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  T[threadIdx.x] = (id < len)? ( ((float)input[id]) / ((float)size) ): 0;
  __syncthreads();
  //pre scan
  int stride = 1;
  while(stride < blockDim.x) {
    id = (threadIdx.x+1)*stride*2-1;
    if(id < blockDim.x) {
      T[id] += T[id-stride];
    }
    stride *= 2;
    __syncthreads();
  }
  //post scan
  stride = blockDim.x/2;
  while(stride > 0) {
    id = (threadIdx.x+1)*stride*2-1;
    if(id+stride < blockDim.x) {
      T[id+stride] += T[id];
    }
    stride /= 2;
    __syncthreads();
  }
  //write to output
  id = blockIdx.x*blockDim.x + threadIdx.x;
  if(id < len) {
    output[id] = T[threadIdx.x];
  }
}

__global__ void correct(unsigned char* ucharImage, float* cdf, float* output, int size) {
  unsigned char correct;
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  __shared__ float cdf_s[HISTOGRAM_LENGTH];
  
  if(threadIdx.x < HISTOGRAM_LENGTH) {
    cdf_s[threadIdx.x] = cdf[threadIdx.x];
  }
  __syncthreads();
  
  float cdfmin = cdf_s[0];
  while(id < size) {
    correct = (unsigned char)( (cdf_s[ucharImage[id]]-cdfmin)*255 / (1.0-cdfmin) );
    if(correct > 255) correct = 255;
    output[id] = (float)(correct/255.0);
    
    id += stride;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char *deviceUcharInputImageData;
  unsigned int *deviceHist;
  float *deviceCdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  //hostOutputImageData = (float*)malloc(imageWidth*imageHeight*imageChannels*sizeof(float));
  
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void**)&deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceUcharInputImageData, imageWidth*imageHeight*imageChannels*sizeof(unsigned char)));
  wbCheck(cudaMalloc((void**)&deviceHist, 256*sizeof(unsigned int)));
  wbCheck(cudaMalloc((void**)&deviceCdf, 256*sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");
  
  wbTime_start(GPU, "Clearing histogram memory.");
  wbCheck(cudaMemset(deviceHist, 0, 256*sizeof(unsigned int)));
  wbTime_stop(GPU, "Clearing histogram memory.");
  
  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  
  //@@compute dim
  dim3 dimGrid((imageWidth*imageHeight-1)/1024+1, 1, 1); // this may exceed the capacity
  dim3 dimBlock(1024, 1, 1);
  dim3 dimGrid2((imageWidth*imageHeight*imageChannels-1)/1024+1, 1, 1); // this may exceed the capacity
  dim3 dimBlock2(1024, 1, 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@insert code
  hist<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceUcharInputImageData, imageWidth*imageHeight, deviceHist);
  scan<<<dim3(1, 1, 1), dim3(HISTOGRAM_LENGTH, 1, 1)>>>(deviceHist, deviceCdf, 256, imageWidth*imageHeight);
  correct<<<dimGrid2, dimBlock2>>>(deviceUcharInputImageData, deviceCdf, deviceOutputImageData, imageWidth*imageHeight*imageChannels);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
  
  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceUcharInputImageData);
  cudaFree(deviceHist);
  cudaFree(deviceCdf);
  wbTime_stop(GPU, "Freeing GPU Memory");
  
  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}

