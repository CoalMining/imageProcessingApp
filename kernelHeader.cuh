#pragma once

#include "header.h"

__global__
void sharpeningKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOfColumns);

__global__
void gaussianBlurKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOnCols);

__global__
void powerLawHistogramKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOnCols);

__global__
void meanBlurKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOnCols);

__global__
void histogramEqualizationKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOfCols, int stride);

__global__
void linearStretchingContrastKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOfCols, int stride);

__global__
void gaussianRGB2GrayKernel(uchar* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOnCols);

__global__
void convolution1DKernel(float* inputArray, int inputArrayLength, float* outputArray, float* maskArray, int maskLength);

__global__
void embossingKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOfColumns);

__global__
void laplaceEdgeDetectionKernel(uchar4* outputPixels, uchar4* inputPixels, uchar* tempPixel, size_t noOfRows, size_t noOfColumns);
__global__
void laplaceEdgeDetectionKernelWithSmoothing(uchar4* outputPixels, uchar4* inputPixels, uchar* tempPixel, uchar* tempPixels2,size_t noOfRows, size_t noOfColumns);

__global__
void convolution2DKernel(uchar4* inputPixels, uchar4* outputPixels, uchar* tempPixels, float* maskKernel2D, int noOfRows, int noOfColumns, int maskWidth, int maskHeight);

__global__
void prewittEdgeDetectionDxKernel(uchar4* outputPixels, uchar4* inputPixels, uchar* tempPixels, size_t noOfRows, size_t noOnCols);
__global__
void prewittEdgeDetectionDyKernel(uchar4* outputPixels, uchar4* inputPixels, uchar* tempPixels, size_t noOfRows, size_t noOnCols);
__global__
void prewittEdgeDetectionDmKernel(uchar4* outputPixels, uchar4* inputPixels, uchar* tempPixels, size_t noOfRows, size_t noOnCols);
__global__
void sobelEdgeDetectionDxKernel(uchar4* outputPixels, uchar4* inputPixels,uchar* tempPixels, size_t noOfRows, size_t noOnCols);
__global__
void sobelEdgeDetectionDyKernel(uchar4* outputPixels, uchar4* inputPixels,uchar* tempUchar, size_t noOfRows, size_t noOnCols);
__global__
void sobelEdgeDetectionDmKernel(uchar4* outputPixels, uchar4* inputPixels,uchar* tempChar, size_t noOfRows, size_t noOnCols);