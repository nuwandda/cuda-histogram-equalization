#include <cstdio>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wb.h"

using namespace std;

#define BLOCK_SIZE 1024
#define HISTOGRAM_SIZE 256

/**
 * File Operations
 */
struct Image {
	int width;
	int height;
	int channels;
	int colors;
	float* data;

    // Used 1 for grayscale images.
    Image(int imageWidth = 0, int imageHeight = 0, int imageChannels = 1, int imageColors = wbInternal::kImageColorLimit) : width(imageWidth), height(imageHeight), channels(imageChannels), colors(imageColors), data(NULL) {
        const int numElements = width * height * channels;

        // Prevent zero-length memory allocation
        if (numElements > 0)
            data = new float[numElements];
    }
};


Image importImage(const char* fName)
{
	ifstream inFile(fName, ios::binary);
	
	if (!inFile.is_open())
	{
		cerr << "Error opening image file " << fName << ". " << wbInternal::wbStrerror(errno) <<  endl;
		exit(EXIT_FAILURE);		
	}
	// Read PGM image header
	string magic;
	getline(inFile, magic);
	
	// use P2 format	
	// if (magic != "P2") 
	// {
	// 	cerr << "Error reading image file " << fName << ". " << "Expecting 'P2' image format but got '" << magic << "'" << endl;
	// 	inFile.close();
	// 	exit(EXIT_FAILURE);
	// }
	
	// Filter image comments
	if (inFile.peek() == '#')
	{
		string commentStr;
		getline(inFile, commentStr);
	}

	Image image;	
	inFile >> image.width;
	
	if (inFile.fail() || 0 >= image.width)
	{
		 cerr << "Error reading width of image in file " << fName <<  endl;
		 inFile.close();
		 exit(EXIT_FAILURE);
	}
	
	inFile >> image.height;
	
	if (inFile.fail() || 0 >= image.height)
	{
		 cerr << "Error reading height of image in file " << fName <<  endl;
		 inFile.close();
		 exit(EXIT_FAILURE);
	}

	inFile >> image.colors;
	if (inFile.fail() || image.colors > wbInternal::kImageColorLimit)
	{
		 cerr << "Error reading colors value of image in file " << fName <<  endl;
		inFile.close();
		 exit(EXIT_FAILURE);
	}
	
	while (isspace(inFile.peek()))
    {
        inFile.get();
    }
	 // not need raw data 
	const int numElements = image.width * image.height * image.channels;

	float* data = new float[numElements];
	
	for (int i = 0; i < numElements; i++)
	{
		inFile >> data[i];		
	}
	
	inFile.close();	
	image.data = data;
	return image;
}

void  saveImage(const Image& image, const char* fName) {
	 ostringstream oss;
     oss << "P2\n" << "#  Created by applying histogram "  << "\n" <<  image.width << " " << image.height << "\n" << image.colors << "\n";
	 string headerStr(oss.str());

	 ofstream outFile(fName,  ios::binary);
	 outFile.write(headerStr.c_str(), headerStr.size());

	 const int numElements = image.width * image.height * image.channels;

	for (int i = 0; i < numElements; ++i)
	{
		outFile << (int)image.data[i] << " ";
	}

	outFile.close();
}

int wbImage_getWidth(const Image& image)
{
	return image.width;
}

int wbImage_getHeight(const Image& image)
{
	return image.height;
}

int wbImage_getChannels(const Image& image)
{
	return image.channels;
}

float* wbImage_getData(const Image& image)
{
	return image.data;
}

Image createImage(const int imageWidth, const int imageHeight, const int imageChannels)
{
	Image image(imageWidth, imageHeight, imageChannels);
	return image;
}

void wbImage_delete(Image& image)
{
	delete[] image.data;
}

/**
 * Kernels 
 */
__global__ void calculateHistogram(bool usingPrivatization, float *buffer, float *histogram, int size) {
    if (usingPrivatization) {
        // The idea of privatization is to replicate highly contended data structures into private copies
        // so that each thread (or each subset of threads) can access a private copy
        // The benefit is that the private copies can be accessed with much less contention and often at much lower latency.
        __shared__ unsigned int cache[HISTOGRAM_SIZE];
        if (threadIdx.x < HISTOGRAM_SIZE) {
            cache[threadIdx.x] = 0;
        }
        __syncthreads();

        // Used interleaved partitioning.
        // All threads process a contiguous section of elements.
        // They all move to the next section and repeat.
        // The memory accesses are coalesced.
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        // Define stride. Stride is the total number of threads.
        int stride = blockDim.x * gridDim.x;

        while (tid < size) {
            atomicAdd(&(cache[(int)buffer[tid]]), 1);
            tid += stride;
        }
        __syncthreads();

        if (threadIdx.x < HISTOGRAM_SIZE) {
            atomicAdd(&(histogram[threadIdx.x]), cache[threadIdx.x]);
        }
    }
    else {
        // Used interleaved partitioning.
        // All threads process a contiguous section of elements.
        // They all move to the next section and repeat.
        // The memory accesses are coalesced.
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        // Define stride. Stride is the total number of threads.
        int stride = blockDim.x * gridDim.x;

        while (tid < size) {
            atomicAdd(&(histogram[(int)buffer[tid]]), 1);
            tid += stride;
        }
    }
}

__global__ void createCumulativeDistributionFunction(int scanType, float *cdf, float *histogram, int size) {
    // In probability theory and statistics, the cumulative distribution function (CDF) of a real-valued random variable X, 
    // or just distribution function of X, evaluated at x, is the probability that X will take a value less than or equal to x.
    if (scanType == 0) {
        // Used Kogge-Stone Scan to create CDF.
        __shared__ float scan[HISTOGRAM_SIZE];
	
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) {	
            scan[threadIdx.x] = histogram[tid];
        }

        for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
            __syncthreads();
            
            if (threadIdx.x >= stride) {	
                scan[threadIdx.x] += scan[threadIdx.x - stride];
            }
        }
                
        __syncthreads();

        for(int i = 0 ; i< HISTOGRAM_SIZE ; i++ ) {
            cdf[i] = scan[i];		 
        }	
    }
    else {
        // Used Brent–Kung Scan to create CDF.
        __shared__ float scan[2 * HISTOGRAM_SIZE];
        int tid = threadIdx.x + blockDim.x * blockIdx.x;

        if (tid < HISTOGRAM_SIZE) {
            scan[threadIdx.x] = histogram[tid];
        }
        __syncthreads();

        // Reduction
        for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
            __syncthreads();
            int index = (threadIdx.x + 1) * stride * 2 - 1;

            if (index < blockDim.x) {
                scan[index] += scan[index - stride];
            }
        }

        // Post Reduction
        for (unsigned int stride = HISTOGRAM_SIZE / 4; stride > 0; stride /= 2) {
            __syncthreads();
            int index = (threadIdx.x + 1) * stride * 2 - 1;

            if (index + stride < HISTOGRAM_SIZE) {
                scan[index + stride] += scan[index];
            }
        }

        __syncthreads();
        for (int i = 0; i < HISTOGRAM_SIZE; i ++) {
            cdf[i] = scan[i];
        }
    }
}

__global__ void calculateCumulativeDistributionFunction(float *cdf, float *cdfClone, int size) {
    float minimum = cdf[0];
    cdfClone[threadIdx.x] = ((cdf[threadIdx.x] - minimum) / (size - minimum) * (HISTOGRAM_SIZE - 1));
}

__global__ void equalizeHistogram(float *outputImageData, float *inputImageData, float *cdf, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size){
        outputImageData[tid] = cdf[(int)inputImageData[tid]];
    }
    __syncthreads();
}


int main(int argc, char ** argv) {
    if (argc != 5) {
        printf("Usage error. Program expects four argument. \n");
    	printf("Usage: ./histogram IMAGENAME PRIVATIZATION SCANTYPE OUTPUTIMAGENAME \n");
		printf("Usage Example: ./histogram 1.pgm 1 1 output.pgm\n");
    	exit(1);
    }

    // System specifications
    printf("-->\n");
    printf("System Specifications:\n");
    printf("\tAzure NC6\n");
    printf("\tCores: 6\n");
    printf("\tGPU: Tesla K80\n");
    printf("\tMemory: 56 GB\n");
    printf("\tDisk: 380 GB SSD\n");
    printf("-->\n");

    const char * inputImageFile;
    const char * outputImageFile;
	Image inputImage;
	Image outputImage;
	float * hostInputImageData;
	float * hostOutputImageData;
	float * hostHistogram; 
    float *deviceInputImageData; 
	float *deviceOutputImageData;
	float *deviceHistogram;
	float *cdfClone;
	float *cdf;
    bool usingPrivatization;
    int scanType;
    const char * strScanType;
    const char * strPrivatization;

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    inputImageFile = argv[1];
    outputImageFile = argv[4];
    if (string(argv[2]) == "0") {
        usingPrivatization = false;
        strPrivatization = "Privatization is not included.";
        printf("Privatization was set to false.\n");
    }
    else {
        usingPrivatization = true;
        strPrivatization = "Privatization is included.";
        printf("Privatization was set to true.\n");
    }

    if (string(argv[3]) == "0") {
        scanType = 0;
        strScanType = "Kogge-Stone is used.";
        printf("Scan type was set to Kogge-Stone.\n");
    }
    else {
        scanType = 1;
        strScanType = "Brent-Kung is used.";
        printf("Scan type was set to Brent-Kung.\n");
    }

    inputImage = importImage(inputImageFile);
	hostHistogram = NULL;
	hostInputImageData = wbImage_getData(inputImage);

    int imageWidth = wbImage_getWidth(inputImage);
	int imageHeight = wbImage_getHeight(inputImage);
	int imageChannels = wbImage_getChannels(inputImage);
    int imageSize = imageWidth * imageHeight * imageChannels; 
    printf("------\n");
    printf("Image Info:\n");
    printf("\tImage Size:%dx%d\n", imageWidth, imageHeight);
    printf("\tImage Channel:%d\n", imageChannels);
    printf("------\n");

    outputImage = createImage(imageWidth, imageHeight, imageChannels);
	hostOutputImageData = wbImage_getData(outputImage);

    size_t sizeImage = imageWidth * imageHeight * imageChannels * sizeof(float);	 
	size_t size = HISTOGRAM_SIZE * sizeof(float);

    // Allocate GPU
    err = cudaMalloc((void **)&deviceInputImageData, sizeImage);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device input image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceOutputImageData, sizeImage);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device output image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceHistogram, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device histogram (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&cdfClone, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate clone cdf (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&cdf, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate cdf (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy image to device memory
    printf("Copying image data from the host memory to the CUDA device...\n");
    err = cudaMemcpy(deviceInputImageData, hostInputImageData, sizeImage, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy host histogram to device histogram
    printf("Copying host histogram to device histogram...\n");
    cudaMemcpy(deviceHistogram, hostHistogram, size, cudaMemcpyHostToDevice);	

	int gridDim = (imageSize + 1) / BLOCK_SIZE;

    cudaEventRecord(start);
	
    calculateHistogram <<< gridDim, BLOCK_SIZE >>>(usingPrivatization, deviceInputImageData, deviceHistogram, imageSize);

    createCumulativeDistributionFunction <<< 1, HISTOGRAM_SIZE >>>(scanType, cdf, deviceHistogram, imageSize);
			
    calculateCumulativeDistributionFunction <<< 1, HISTOGRAM_SIZE  >>>(cdf, cdfClone, imageSize);

    equalizeHistogram <<< gridDim, BLOCK_SIZE >>>(deviceOutputImageData, deviceInputImageData, cdfClone, imageSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    printf("Copying device output image data to host output image data...\n");
    err = cudaMemcpy(hostOutputImageData, deviceOutputImageData, sizeImage, cudaMemcpyDeviceToHost);	
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);

    // Timings
    printf("Execution time:\n");
    printf("\tPrivatization: %s Scan type: %s \n\tElapsed time: %f milliseconds.\n", strPrivatization, strScanType, ms);

	saveImage(outputImage, outputImageFile);
	 
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceHistogram);
	cudaFree(cdfClone);

	return 0;
}