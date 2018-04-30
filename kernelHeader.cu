#include "kernelHeader.cuh"


__global__
void meanBlurKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOfColumns)
{
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int div = 9;

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns - 1 && col>0))
	{
		outputPixels[row*noOfColumns + col].x = (inputPixels[row*noOfColumns + col].x / div + inputPixels[(row + 1)*noOfColumns + col].x / div + inputPixels[(row - 1)*noOfColumns + col].x / div + inputPixels[row*noOfColumns + (col + 1)].x / div + inputPixels[row*noOfColumns + (col - 1)].x / div + inputPixels[(row - 1)*noOfColumns + (col - 1)].x / div + inputPixels[(row - 1)*noOfColumns + (col + 1)].x / div + inputPixels[(row + 1)*noOfColumns + (col - 1)].x / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].x / div);
		outputPixels[row*noOfColumns + col].y = (inputPixels[row*noOfColumns + col].y/div + inputPixels[(row + 1)*noOfColumns + col].y/ div + inputPixels[(row - 1)*noOfColumns + col].y/ div + inputPixels[row*noOfColumns + (col + 1)].y/ div + inputPixels[row*noOfColumns + (col - 1)].y/ div + inputPixels[(row - 1)*noOfColumns + (col - 1)].y / div + inputPixels[(row - 1)*noOfColumns + (col + 1)].y / div + inputPixels[(row + 1)*noOfColumns + (col - 1)].y / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].y / div);
		outputPixels[row*noOfColumns + col].z = (inputPixels[row*noOfColumns + col].z/div + inputPixels[(row + 1)*noOfColumns + col].z/ div + inputPixels[(row - 1)*noOfColumns + col].z/ div + inputPixels[row*noOfColumns + (col + 1)].z/ div + inputPixels[row*noOfColumns + (col - 1)].z/ div + inputPixels[(row - 1)*noOfColumns + (col - 1)].z / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].z / div + inputPixels[(row + 1)*noOfColumns + (col - 1)].z / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].z / div);
	}
	else if (row < noOfRows && col < noOfColumns)
	{
		outputPixels[row*noOfColumns + col].x = inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = inputPixels[row*noOfColumns + col].z;
	}
}

__global__
void powerLawHistogramKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOnCols)
{
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;

	double c = 1;
	double r = 3;

	if (row < noOfRows && col < noOnCols)
	{
		outputPixels[row*noOnCols + col].x = c * pow((double)inputPixels[row*noOnCols + col].x, r);
		outputPixels[row*noOnCols + col].y = c * pow((double)inputPixels[row*noOnCols + col].y, r);
		outputPixels[row*noOnCols + col].z = c * pow((double)inputPixels[row*noOnCols + col].z, r);
	}
}

__global__
void gaussianBlurKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOfColumns)
{
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int div = 16;

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns - 1 && col>0))
	{
		uint8_t tempx = (uint8_t) (4*inputPixels[row*noOfColumns + col].x / div + 2*inputPixels[(row + 1)*noOfColumns + col].x / div + 2*inputPixels[(row - 1)*noOfColumns + col].x / div + 2*inputPixels[row*noOfColumns + (col + 1)].x / div + 2*inputPixels[row*noOfColumns + (col - 1)].x / div + inputPixels[(row - 1)*noOfColumns + (col - 1)].x / div + inputPixels[(row - 1)*noOfColumns + (col + 1)].x / div + inputPixels[(row + 1)*noOfColumns + (col - 1)].x / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].x / div);
		outputPixels[row*noOfColumns + col].x = (uchar)tempx;
		uint8_t tempy = (uint8_t)(4 * inputPixels[row*noOfColumns + col].y / div + 2 * inputPixels[(row + 1)*noOfColumns + col].y / div + 2 * inputPixels[(row - 1)*noOfColumns + col].y / div + 2 * inputPixels[row*noOfColumns + (col + 1)].y / div + 2 * inputPixels[row*noOfColumns + (col - 1)].y / div + inputPixels[(row - 1)*noOfColumns + (col - 1)].y / div + inputPixels[(row - 1)*noOfColumns + (col + 1)].y / div + inputPixels[(row + 1)*noOfColumns + (col - 1)].y / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].y / div);
		outputPixels[row*noOfColumns + col].y = (uchar)tempy;
		uint8_t tempz = (uint8_t)(4 * inputPixels[row*noOfColumns + col].z / div + 2 * inputPixels[(row + 1)*noOfColumns + col].z / div + 2 * inputPixels[(row - 1)*noOfColumns + col].z / div + 2 * inputPixels[row*noOfColumns + (col + 1)].z / div + 2 * inputPixels[row*noOfColumns + (col - 1)].z / div + inputPixels[(row - 1)*noOfColumns + (col - 1)].z / div + inputPixels[(row - 1)*noOfColumns + (col + 1)].z / div + inputPixels[(row + 1)*noOfColumns + (col - 1)].z / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].z / div);
		outputPixels[row*noOfColumns + col].z = (uchar)tempz;

		//outputPixels[row*noOfColumns + col].x = (4 * inputPixels[row*noOfColumns + col].x / div + 2 * inputPixels[(row + 1)*noOfColumns + col].x / div + 2 * inputPixels[(row - 1)*noOfColumns + col].x / div + 2 * inputPixels[row*noOfColumns + (col + 1)].x / div + 2 * inputPixels[row*noOfColumns + (col - 1)].x / div + inputPixels[(row - 1)*noOfColumns + (col - 1)].x / div + inputPixels[(row - 1)*noOfColumns + (col + 1)].x / div + inputPixels[(row + 1)*noOfColumns + (col - 1)].x / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].x / div);
		//outputPixels[row*noOfColumns + col].y = (4 * inputPixels[row*noOfColumns + col].z / div + 2 * inputPixels[(row + 1)*noOfColumns + col].z / div + 2 * inputPixels[(row - 1)*noOfColumns + col].z / div + 2 * inputPixels[row*noOfColumns + (col + 1)].z / div + 2 * inputPixels[row*noOfColumns + (col - 1)].z / div + inputPixels[(row - 1)*noOfColumns + (col - 1)].z / div + inputPixels[(row - 1)*noOfColumns + (col + 1)].z / div + inputPixels[(row + 1)*noOfColumns + (col - 1)].z / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].z / div);
		//outputPixels[row*noOfColumns + col].z = (4 * inputPixels[row*noOfColumns + col].z / div + 2 * inputPixels[(row + 1)*noOfColumns + col].z / div + 2 * inputPixels[(row - 1)*noOfColumns + col].z / div + 2 * inputPixels[row*noOfColumns + (col + 1)].z / div + 2 * inputPixels[row*noOfColumns + (col - 1)].z / div + inputPixels[(row - 1)*noOfColumns + (col - 1)].z / div + inputPixels[(row - 1)*noOfColumns + (col + 1)].z / div + inputPixels[(row + 1)*noOfColumns + (col - 1)].z / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].z / div);
	}
	else if (row < noOfRows && col < noOfColumns)
	{
		outputPixels[row*noOfColumns + col].x = inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = inputPixels[row*noOfColumns + col].z;
	}
}

__global__
void sharpeningKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOfColumns)
{
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns - 1 && col>0))
	{
		uint16_t tempx = (uint16_t)(9 * (float)inputPixels[row*noOfColumns + col].x - inputPixels[(row + 1)*noOfColumns + col].x  - inputPixels[(row - 1)*noOfColumns + col].x - inputPixels[row*noOfColumns + (col + 1)].x  - inputPixels[row*noOfColumns + (col - 1)].x  - inputPixels[(row - 1)*noOfColumns + (col - 1)].x  - inputPixels[(row - 1)*noOfColumns + (col + 1)].x- inputPixels[(row + 1)*noOfColumns + (col - 1)].x - inputPixels[(row + 1)*noOfColumns + (col + 1)].x);
		outputPixels[row*noOfColumns + col].x = (uchar)tempx;
		uint16_t tempy = (uint16_t)(9 * (float)inputPixels[row*noOfColumns + col].y - inputPixels[(row + 1)*noOfColumns + col].y  - inputPixels[(row - 1)*noOfColumns + col].y  - inputPixels[row*noOfColumns + (col + 1)].y  - inputPixels[row*noOfColumns + (col - 1)].y  - inputPixels[(row - 1)*noOfColumns + (col - 1)].y  - inputPixels[(row - 1)*noOfColumns + (col + 1)].y- inputPixels[(row + 1)*noOfColumns + (col - 1)].y  - inputPixels[(row + 1)*noOfColumns + (col + 1)].y );
		outputPixels[row*noOfColumns + col].y = (uchar)tempy;
		uint16_t tempz = (uint16_t)(9 * (float)inputPixels[row*noOfColumns + col].z  - inputPixels[(row + 1)*noOfColumns + col].z  - inputPixels[(row - 1)*noOfColumns + col].z  - inputPixels[row*noOfColumns + (col + 1)].z  - inputPixels[row*noOfColumns + (col - 1)].z  - inputPixels[(row - 1)*noOfColumns + (col - 1)].z  - inputPixels[(row - 1)*noOfColumns + (col + 1)].z  - inputPixels[(row + 1)*noOfColumns + (col - 1)].z  - inputPixels[(row + 1)*noOfColumns + (col + 1)].z );
		outputPixels[row*noOfColumns + col].z = (uchar)tempz;
	}
	else if (row < noOfRows && col < noOfColumns)
	{
		outputPixels[row*noOfColumns + col].x = inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = inputPixels[row*noOfColumns + col].z;
	}

}

__global__
void gaussianRGB2GrayKernel(uchar* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOnCols)
{
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;

	if (row < noOfRows &&col < noOnCols)
	{
		float temp = 0.299*inputPixels[row*noOnCols + col].x + 0.587*inputPixels[row*noOnCols + col].y + .114*inputPixels[row*noOnCols + col].z;
		outputPixels[row*noOnCols + col] = temp;
	}
}

__global__
void linearStretchingContrastKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOfCols, int stride)
{
	__shared__ int3 maxVal;
	__shared__ int3 minVal;

	int index = threadIdx.x;

	if (index == 0)
	{
		maxVal.x = 0;
		maxVal.y = 0;
		maxVal.z = 0;
	}else if(index == 1)
	{
		minVal.x = 255;
		minVal.y = 255;
		minVal.z = 255;
	}
	__syncthreads();

	int i = index;
	//find the minimum and maximum value of the intensity in each channel
	while (i < noOfCols*noOfRows)
	{
		atomicMin(&minVal.x, (int)inputPixels[i].x);
		atomicMin(&minVal.y, (int)inputPixels[i].y);
		atomicMin(&minVal.z, (int)inputPixels[i].z);

		atomicMax(&maxVal.x, (int)inputPixels[i].x);
		atomicMax(&maxVal.y, (int)inputPixels[i].y);
		atomicMax(&maxVal.z, (int)inputPixels[i].z);
		i += stride;
	}
	__syncthreads();

	i = index;
	while (i<noOfCols*noOfRows)
	{
		outputPixels[i].x = (uchar)(((float)255)*((float)inputPixels[i].x - (float)minVal.x) / ((float)maxVal.x - (float)minVal.x)); 
		outputPixels[i].y = (uchar)(((float)255)*((float)inputPixels[i].y - (float)minVal.y) / ((float)maxVal.y - (float)minVal.y));
		outputPixels[i].z = (uchar)(((float)255)*((float)inputPixels[i].z - (float)minVal.z) / ((float)maxVal.z - (float)minVal.z));
		i += stride;
	}
}

__global__
void histogramEqualizationKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOfCols, int stride)
{
	__shared__ int histogramR[256];
	__shared__ int histogramG[256];
	__shared__ int histogramB[256];
	__shared__ int cumulativeHistoR[256];
	__shared__ int cumulativeHistoG[256];
	__shared__ int cumulativeHistoB[256];
	__shared__ int minVal[3];//0->r,1->g,2->b

	//index is the id of the thread, will be in the range 0-1023
	int index =threadIdx.x;
	if (index > 255)
		return;
	if (index == 0)
	{
		minVal[0] = minVal[1] = minVal[2] = INT_MAX;
	}
	//first before making the histogram, lets make the data 0
	//histogram contains 0-255, so first 256 will populate them
	if (index < 256)
	{
		histogramR[index] = 0;
		histogramG[index] = 0;
		histogramB[index] = 0;
	}
	__syncthreads();

	//perform histogram calculation here
	//atomicAdd will be used
	do
	{
		atomicAdd(&histogramR[inputPixels[index].x], (int)1);
		atomicAdd(&histogramG[inputPixels[index].y], (int)1);
		atomicAdd(&histogramB[inputPixels[index].z], (int)1);
		index += stride;
	} while (index<noOfCols*noOfRows);
__syncthreads();

//index = threadIdx.x;
//printf("HistR[%d]=%d,HistG[%d]=%d,HistB[%d]=%d\n",index,histogramR[index],index, histogramG[index],index, histogramB[index]);
//__syncthreads();

//---------- histogram is calculated already
//hillis-steele inclusive scan algorithm
index = threadIdx.x;
if (index < 256)
{
	cumulativeHistoR[index] = histogramR[index];
	cumulativeHistoG[index] = histogramG[index];
	cumulativeHistoB[index] = histogramB[index];
	__syncthreads();

	int j = 1;
	while (j <= 256 / 2)
	{
		if (index + j < 256)
		{
			cumulativeHistoR[index + j] = cumulativeHistoR[index + j] + cumulativeHistoR[index];
			cumulativeHistoG[index + j] = cumulativeHistoG[index + j] + cumulativeHistoG[index];
			cumulativeHistoB[index + j] = cumulativeHistoB[index + j] + cumulativeHistoB[index];
		}
		j *= 2;
		__syncthreads();
	}
}
__syncthreads();
//index = threadIdx.x;
//printf("%d %d %d %d %d %d %d",index,histogramR[index],cumulativeHistoR[index],histogramG[index], cumulativeHistoG[index],histogramB[index], cumulativeHistoB[index]);
//__syncthreads();

//finds the minimum non-zero cumulative value for each R G and B
index = threadIdx.x;
if (index < 256)
{
	if (cumulativeHistoR[index]>0) atomicMin(&minVal[0], cumulativeHistoR[index]);
	if (cumulativeHistoG[index]>0) atomicMin(&minVal[1], cumulativeHistoG[index]);
	if (cumulativeHistoB[index]>0) atomicMin(&minVal[2], cumulativeHistoB[index]);
}
__syncthreads();


do
{
	outputPixels[index].x = (uchar)(((float)cumulativeHistoR[inputPixels[index].x] /*- minVal[0])*/ * 255) / (noOfRows*noOfCols - 1));
	outputPixels[index].y = (uchar)(((float)cumulativeHistoR[inputPixels[index].y] /*- minVal[1])*/ * 255 )/ (noOfRows*noOfCols - 1));
	outputPixels[index].z = (uchar)(((float)cumulativeHistoR[inputPixels[index].z] /*- minVal[2]) */* 255 )/ (noOfRows*noOfCols - 1));
	index += stride;
} while (index < noOfCols*noOfRows);

}

__global__
void convolution1DKernel(float* inputArray, int inputArrayLength, float* outputArray, float* maskArray, int maskLength)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float tempVal = 0;
	int calculationStartPoint = index - (maskLength / 2);
	for (int j = 0; j < maskLength; j++)
	{
		if (calculationStartPoint + j >= 0 && calculationStartPoint + j < inputArrayLength)
		{
			tempVal += inputArray[calculationStartPoint + j] * maskArray[j];
		}
	}
	outputArray[index] = tempVal;
}

__global__
void embossingKernel(uchar4* outputPixels, uchar4* inputPixels, size_t noOfRows, size_t noOfColumns)
{
	//2  0  0
	//0 -1  0
	//0  0 -1
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns - 1 && col>0))
	{
		uint8_t xtemp = (uint8_t)(inputPixels[(row - 1)*noOfColumns + (col - 1)].x * 2 - inputPixels[(row)*noOfColumns + (col)].x - inputPixels[(row + 1)*noOfColumns + (col + 1)].x);
		outputPixels[row*noOfColumns + col].x = (uchar)xtemp;
		uint8_t ytemp = (uint8_t)(inputPixels[(row - 1)*noOfColumns + (col - 1)].y * 2 - inputPixels[(row)*noOfColumns + (col)].y - inputPixels[(row + 1)*noOfColumns + (col + 1)].y);
		outputPixels[row*noOfColumns + col].x = (uchar)ytemp;
		uint8_t ztemp = (uint8_t)(inputPixels[(row - 1)*noOfColumns + (col - 1)].z * 2 - inputPixels[(row)*noOfColumns + (col)].z - inputPixels[(row + 1)*noOfColumns + (col + 1)].z);
		outputPixels[row*noOfColumns + col].z = (uchar)ztemp;
	}
	else if (row < noOfRows && col < noOfColumns)
	{
		outputPixels[row*noOfColumns + col].x = inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = inputPixels[row*noOfColumns + col].z;
	}
}

__global__
void prewittEdgeDetectionDxKernel(uchar4* outputPixels, uchar4* inputPixels,uchar* tempPixels, size_t noOfRows, size_t noOfColumns)
{
	/* -1  0  1
 1/3   -1  0  1
	   -1  0  1*/

	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int div = 1;
	float temp;
	if (row < noOfRows &&col < noOfColumns)
	{
		temp = 0.299*inputPixels[row*noOfColumns + col].x + 0.587*inputPixels[row*noOfColumns + col].y + .114*inputPixels[row*noOfColumns + col].z;
		tempPixels[row*noOfColumns + col] = temp;
	}
	__syncthreads();

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns-1 && col>0))
	{
		float xtemp = ((float)abs(((float)tempPixels[(row - 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row - 1)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row - 1)*noOfColumns + (col + 1)]
			+ (float)tempPixels[(row)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col + 1)] * 1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row + 1)*noOfColumns + (col + 1)] * 1)));


		if (xtemp > 200) {
			outputPixels[row*noOfColumns + col].x = (uchar)xtemp;
			outputPixels[row*noOfColumns + col].y = (uchar)xtemp;
			outputPixels[row*noOfColumns + col].z = (uchar)xtemp;
		}
		else
		{
			outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
			outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
			outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
		}
	}
	else
	{
		outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
	}

}



__global__
void prewittEdgeDetectionDyKernel(uchar4* outputPixels, uchar4* inputPixels,uchar* tempPixels, size_t noOfRows, size_t noOfColumns)
{
	//		1  1  1
	//1/3	0  0  0
	//		-1 -1 -1
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int div = 1;
	float temp;
	if (row < noOfRows &&col < noOfColumns)
	{
		temp = 0.299*inputPixels[row*noOfColumns + col].x + 0.587*inputPixels[row*noOfColumns + col].y + .114*inputPixels[row*noOfColumns + col].z;
		tempPixels[row*noOfColumns + col] = temp;
	}
	__syncthreads();

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns-1 && col>0))
	{
		float xtemp = ((float)abs((float)tempPixels[(row - 1)*noOfColumns + (col - 1)] * 1
			+ (float)tempPixels[(row - 1)*noOfColumns + (col)] * 1
			+ (float)tempPixels[(row - 1)*noOfColumns + (col + 1)]
			+ (float)tempPixels[(row)*noOfColumns + (col - 1)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col + 1)] * 0
			+ (float)tempPixels[(row + 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col + 1)] * -1));


		if (xtemp > 200) {
			outputPixels[row*noOfColumns + col].x = (uchar)xtemp;
			outputPixels[row*noOfColumns + col].y = (uchar)xtemp;
			outputPixels[row*noOfColumns + col].z = (uchar)xtemp;
		}
		else
		{
			outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
			outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
			outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
		}
	}
	else
	{
		outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
	}

}

__global__
void prewittEdgeDetectionDmKernel(uchar4* outputPixels, uchar4* inputPixels, uchar* tempPixels, size_t noOfRows, size_t noOfColumns)
{
	/* -1  0  1
	1/3   -1  0  1
	-1  0  1*/

	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int div = 1;
	float temp;
	if (row < noOfRows &&col < noOfColumns)
	{
		temp = 0.299*inputPixels[row*noOfColumns + col].x + 0.587*inputPixels[row*noOfColumns + col].y + .114*inputPixels[row*noOfColumns + col].z;
		tempPixels[row*noOfColumns + col] = temp;
	}
	__syncthreads();

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns - 1 && col>0))
	{
		float xtemp = ((float)abs(((float)tempPixels[(row - 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row - 1)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row - 1)*noOfColumns + (col + 1)]
			+ (float)tempPixels[(row)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col + 1)] * 1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row + 1)*noOfColumns + (col + 1)] * 1)));

		float ytemp = ((float)abs((float)tempPixels[(row - 1)*noOfColumns + (col - 1)] * 1
			+ (float)tempPixels[(row - 1)*noOfColumns + (col)] * 1
			+ (float)tempPixels[(row - 1)*noOfColumns + (col + 1)]
			+ (float)tempPixels[(row)*noOfColumns + (col - 1)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col + 1)] * 0
			+ (float)tempPixels[(row + 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col + 1)] * -1));

		float temp = abs(sqrt((float)(xtemp*xtemp) + (float)(ytemp*ytemp)));

		if (xtemp > 200) {
			outputPixels[row*noOfColumns + col].x = (uchar)temp;
			outputPixels[row*noOfColumns + col].y = (uchar)temp;
			outputPixels[row*noOfColumns + col].z = (uchar)temp;
		}
		else
		{
			outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
			outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
			outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
		}
	}
	else
	{
		outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
	}

}


__global__
void sobelEdgeDetectionDxKernel(uchar4* outputPixels, uchar4* inputPixels,uchar* tempPixels, size_t noOfRows, size_t noOfColumns)
{
	//-1  0  1
	//-2  0  2
	//-1  0  1
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int div = 1;
	float temp;
	if (row < noOfRows &&col < noOfColumns)
	{
		temp = 0.299*inputPixels[row*noOfColumns + col].x + 0.587*inputPixels[row*noOfColumns + col].y + .114*inputPixels[row*noOfColumns + col].z;
		tempPixels[row*noOfColumns + col] = temp;
	}
	__syncthreads();

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns - 1 && col>0))
	{
		float xtemp = ((float)abs((float)tempPixels[(row - 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row - 1)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row - 1)*noOfColumns + (col + 1)]
			+ (float)tempPixels[(row)*noOfColumns + (col - 1)] * -2
			+ (float)tempPixels[(row)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col + 1)] * 2
			+ (float)tempPixels[(row + 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col + 1)] * 1));


		if (xtemp > 200) {
			outputPixels[row*noOfColumns + col].x = (uchar)xtemp;
			outputPixels[row*noOfColumns + col].y = (uchar)xtemp;
			outputPixels[row*noOfColumns + col].z = (uchar)xtemp;
		}
		else
		{
			outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
			outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
			outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
		}
	}
	else
	{
		outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
	}
}

__global__
void sobelEdgeDetectionDyKernel(uchar4* outputPixels, uchar4* inputPixels, uchar* tempPixels,size_t noOfRows, size_t noOfColumns)
{
	//1  2  1
	//0  0  0
	//-1 -2 -1
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;

	int div = 1;
	float temp;

	if (row < noOfRows &&col < noOfColumns)
	{
		temp = 0.299*inputPixels[row*noOfColumns + col].x + 0.587*inputPixels[row*noOfColumns + col].y + .114*inputPixels[row*noOfColumns + col].z;
		tempPixels[row*noOfColumns + col] = temp;
	}
	__syncthreads();

	if ((row < (noOfRows - 1) && row>0) && (col < (noOfColumns - 1) && col>0))
	{
		float xtemp = ((float)abs((float)tempPixels[(row - 1)*noOfColumns + (col - 1)] * 1
			+ (float)tempPixels[(row - 1)*noOfColumns + (col)] * 2
			+ (float)tempPixels[(row - 1)*noOfColumns + (col + 1)]
			+ (float)tempPixels[(row)*noOfColumns + (col - 1)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col + 1)] * 0
			+ (float)tempPixels[(row + 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col)] * -2
			+ (float)tempPixels[(row + 1)*noOfColumns + (col + 1)] * -1));


		if (xtemp > 200) {
		outputPixels[row*noOfColumns + col].x = (uchar)xtemp;
		outputPixels[row*noOfColumns + col].y = (uchar)xtemp;
		outputPixels[row*noOfColumns + col].z = (uchar)xtemp;
		}
		else
		{
			outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
			outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
			outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
		}
	}
	else
	{
		outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
	}
}

__global__
void sobelEdgeDetectionDmKernel(uchar4* outputPixels, uchar4* inputPixels,uchar* tempPixels, size_t noOfRows, size_t noOfColumns)
{
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int div = 1;
	float temp;
	if (row < noOfRows &&col < noOfColumns)
	{
		temp = 0.299*inputPixels[row*noOfColumns + col].x + 0.587*inputPixels[row*noOfColumns + col].y + .114*inputPixels[row*noOfColumns + col].z;
		tempPixels[row*noOfColumns + col] = temp;
	}
	__syncthreads();

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns - 1 && col>0))
	{
		float xtemp = ((float)abs((float)tempPixels[(row - 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row - 1)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row - 1)*noOfColumns + (col + 1)]
			+ (float)tempPixels[(row)*noOfColumns + (col - 1)] * -2
			+ (float)tempPixels[(row)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col + 1)] * 2
			+ (float)tempPixels[(row + 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col + 1)] * 1));

		float ytemp = ((float)abs((float)tempPixels[(row - 1)*noOfColumns + (col - 1)] * 1
			+ (float)tempPixels[(row - 1)*noOfColumns + (col)] * 2
			+ (float)tempPixels[(row - 1)*noOfColumns + (col + 1)]
			+ (float)tempPixels[(row)*noOfColumns + (col - 1)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col)] * 0
			+ (float)tempPixels[(row)*noOfColumns + (col + 1)] * 0
			+ (float)tempPixels[(row + 1)*noOfColumns + (col - 1)] * -1
			+ (float)tempPixels[(row + 1)*noOfColumns + (col)] * -2
			+ (float)tempPixels[(row + 1)*noOfColumns + (col + 1)] * -1));

		float temp = abs(sqrt((float)(xtemp*xtemp) + (float)(ytemp*ytemp)));

		if (xtemp > 200) {
		outputPixels[row*noOfColumns + col].x = (uchar)temp;
		outputPixels[row*noOfColumns + col].y = (uchar)temp;
		outputPixels[row*noOfColumns + col].z = (uchar)temp;
		}
		else
		{
			outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
			outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
			outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
		}
	}
	else
	{
		outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
	}

}
__global__
void laplaceEdgeDetectionKernelWithSmoothing(uchar4* outputPixels, uchar4* inputPixels, uchar* tempPixels,uchar* tempPixels2, size_t noOfRows, size_t noOfColumns)
{

	//0  1  0
	//1 -4  1
	//0  1  0

	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;

	if (row < noOfRows && col < noOfColumns)
	{
		float temp = 0.299*inputPixels[row*noOfColumns + col].x + 0.587*inputPixels[row*noOfColumns + col].y + .114*inputPixels[row*noOfColumns + col].z;
		tempPixels[row*noOfColumns + col] = temp;
	}
	__syncthreads();

	int div = 9;

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns - 1 && col>0))
	{
		tempPixels2[row*noOfColumns + col] = (tempPixels[row*noOfColumns + col] / div + tempPixels[(row + 1)*noOfColumns + col] / div + tempPixels[(row - 1)*noOfColumns + col] / div + tempPixels[row*noOfColumns + (col + 1)] / div + tempPixels[row*noOfColumns + (col - 1)] / div + tempPixels[(row - 1)*noOfColumns + (col - 1)] / div + tempPixels[(row - 1)*noOfColumns + (col + 1)] / div + tempPixels[(row + 1)*noOfColumns + (col - 1)] / div + tempPixels[(row + 1)*noOfColumns + (col + 1)] / div);
	}
	else
	{
		tempPixels2[row*noOfColumns + col] = 255;// tempPixels[row*noOfColumns + col];
	}

	__syncthreads();
	div = 1;

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns - 1 && col>0))
	{
		float xtemp = (float)abs(int((tempPixels2[(row - 1)*noOfColumns + (col)]
			+ tempPixels2[(row)*noOfColumns + (col - 1)]
			+ tempPixels2[(row + 1)*noOfColumns + (col)]
			+ tempPixels2[(row)*noOfColumns + (col + 1)]
			- 4 * tempPixels2[(row)*noOfColumns + (col)])) / div);

		if (xtemp > 50) {
			outputPixels[row*noOfColumns + col].x = 255;// tempPixels[row*noOfColumns + col];// (uchar)xtemp;
			outputPixels[row*noOfColumns + col].y = 255;// tempPixels[row*noOfColumns + col];//(uchar)xtemp;
		outputPixels[row*noOfColumns + col].z = 255;// tempPixels[row*noOfColumns + col];//(uchar)xtemp;
		}
		else
		{
			outputPixels[row*noOfColumns + col].x = 0;
			outputPixels[row*noOfColumns + col].y = 0;
			outputPixels[row*noOfColumns + col].z = 0;
		}
	}
	else
	{
		outputPixels[row*noOfColumns + col].x = 0;// tempPixels[row*noOfColumns + col];
		outputPixels[row*noOfColumns + col].y = 0;//tempPixels[row*noOfColumns + col];
		outputPixels[row*noOfColumns + col].z = 0;//tempPixels[row*noOfColumns + col];
	}
}
__global__
void laplaceEdgeDetectionKernel(uchar4* outputPixels, uchar4* inputPixels,uchar* tempPixels, size_t noOfRows, size_t noOfColumns)
{

	//0  1  0
	//1 -4  1
	//0  1  0

	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int div = 1;
	float temp;
	if (row < noOfRows &&col < noOfColumns)
	{
		temp = 0.299*inputPixels[row*noOfColumns + col].x + 0.587*inputPixels[row*noOfColumns + col].y + .114*inputPixels[row*noOfColumns + col].z;
		tempPixels[row*noOfColumns + col] = temp;
	}
	__syncthreads();

	if ((row < noOfRows - 1 && row>0) && (col < noOfColumns - 1 && col>0))
	{
		float xtemp = (float)abs((float)(tempPixels[(row - 1)*noOfColumns + (col)]
						+ (float)tempPixels[(row)*noOfColumns + (col - 1)]
						+ (float)tempPixels[(row + 1)*noOfColumns + (col)]
						+ (float)tempPixels[(row)*noOfColumns + (col + 1)]
						- 4 * (float)tempPixels[(row)*noOfColumns + (col)])/div);
		 
		if (xtemp > 200) {
			outputPixels[row*noOfColumns + col].x = 255;// (uchar)xtemp;
			outputPixels[row*noOfColumns + col].y = 255;//(uchar)xtemp;
			outputPixels[row*noOfColumns + col].z = 255;//(uchar)xtemp;
		}
		else
		{
			outputPixels[row*noOfColumns + col].x = 0;
			outputPixels[row*noOfColumns + col].y = 0;
			outputPixels[row*noOfColumns + col].z = 0;
		}
	}
	else  //(row < noOfRows && col < noOfColumns)
	{
		outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
	}
}

__global__
void convolution2DKernel(uchar4* inputPixels,uchar4* outputPixels,uchar* tempPixels,float* maskKernel2D,int noOfRows,int noOfColumns,int maskWidth,int maskHeight)
{
	int row = threadIdx.y + blockDim.y*blockIdx.y;	//row means y
	int col = threadIdx.x + blockDim.x*blockIdx.x;	//col means x

	//if ((i *noOfColumns+j)< (noOfColumns*noOfRows))
	//{
	//	float tempX = 0;
	//	float tempY = 0;
	//	float tempZ = 0;
	//		int iStart = i-(int)(maskHeight / 2);
	//		int jStart = j-(int)(maskWidth / 2);

	//		for (int k = 0; k < maskHeight; k++)
	//			for (int l = 0; l < maskWidth; l++)
	//			{
	//				int index = ((iStart + k)*noOfColumns + (jStart) + l);
	//				if ((iStart + k) >= 0 && (jStart)+l>=0 && (jStart)+l<noOfColumns && iStart+k<noOfRows)
	//				{
	//					tempX += (float)inputPixels[index].x * maskKernel2D[(k*maskWidth + j)];
	//					tempY += (float)inputPixels[index].y * maskKernel2D[(k*maskWidth + j)];
	//					tempZ += (float)inputPixels[index].z * maskKernel2D[(k*maskWidth + j)];
	//					//if(i==200&&j==200)
	//					//printf("%d %d %d %d,",k,l, (iStart + k), (jStart)+l);
	//				}
	//			}

	//	outputPixels[i*noOfColumns + j].x = (uchar)tempX;
	//	outputPixels[i*noOfColumns + j].z = (uchar)tempZ;
	//	outputPixels[i*noOfColumns + j].y = (uchar)tempY;
	//}
	float temp;
	if (maskWidth != 3 || maskHeight != 3)
		return;
	if (row < noOfRows &&col < noOfColumns)
	{
		temp = 0.299*inputPixels[row*noOfColumns + col].x + 0.587*inputPixels[row*noOfColumns + col].y + .114*inputPixels[row*noOfColumns + col].z;
		tempPixels[row*noOfColumns + col] = temp;
	}
	__syncthreads();

	if ((row < (noOfRows - 1) && row>0) && (col < (noOfColumns - 1) && col>0))
	{
		float xtemp = ((float)abs((float)tempPixels[(row - 1)*noOfColumns + (col - 1)] * maskKernel2D[0]
			+ (float)tempPixels[(row - 1)*noOfColumns + (col)] * maskKernel2D[1]
			+ (float)tempPixels[(row - 1)*noOfColumns + (col + 1)]*maskKernel2D[2] 
			+ (float)tempPixels[(row)*noOfColumns + (col - 1)] * maskKernel2D[3]
			+ (float)tempPixels[(row)*noOfColumns + (col)] * maskKernel2D[4]
			+ (float)tempPixels[(row)*noOfColumns + (col + 1)] * maskKernel2D[5]
			+ (float)tempPixels[(row + 1)*noOfColumns + (col - 1)] * maskKernel2D[6]
			+ (float)tempPixels[(row + 1)*noOfColumns + (col)] * maskKernel2D[7]
			+ (float)tempPixels[(row + 1)*noOfColumns + (col + 1)] * maskKernel2D[8]));


		if (xtemp > 200) {
			outputPixels[row*noOfColumns + col].x = (uchar)xtemp;
			outputPixels[row*noOfColumns + col].y = (uchar)xtemp;
			outputPixels[row*noOfColumns + col].z = (uchar)xtemp;
		}
		else
		{
			outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
			outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
			outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
		}
	}
	else
	{
		outputPixels[row*noOfColumns + col].x = 0;// inputPixels[row*noOfColumns + col].x;
		outputPixels[row*noOfColumns + col].y = 0;// inputPixels[row*noOfColumns + col].y;
		outputPixels[row*noOfColumns + col].z = 0;// inputPixels[row*noOfColumns + col].z;
	}
}