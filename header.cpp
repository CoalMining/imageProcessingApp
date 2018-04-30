#include "header.h"
#include "kernelHeader.cuh"

using namespace std;
using namespace cv;

int mainApplication::loadImage(string path)
{
	int returnStatus = 0;

	//reading the image from the path given
	inputImageMat = imread(path, CV_LOAD_IMAGE_COLOR);

	//error checking if the read is done properly
	if (!inputImageMat.data)
	{
		cout << "Error in reading the image file. \n\t Please check the file and try again" << endl;
		returnStatus = -1;
	}
	return returnStatus;
}

int mainApplication::displayInputImage()
{
	if (!inputImageMat.data)
	{
		cout << "No input image data to load" << endl;
		return -1;
	}
	cout << "Press any key to continue" << endl;
	//display input image
	destroyWindow("Input Image");
	namedWindow("Input Image", WINDOW_AUTOSIZE);
	imshow("Input Image", inputImageMat);
	waitKey(1);

	return 0;
}
int mainApplication::displayOutputImage()
{
	if (!outputImageMat.data)
	{
		cout << "No output image data to load" << endl;
		return -1;
	}
	cout << "Press any key to continue" << endl;
	//display input image
	destroyWindow("Output Image");
	namedWindow("Output Image", WINDOW_AUTOSIZE);
	imshow("Output Image", outputImageMat);
	waitKey(1);

	return 0;
}
bool mainApplication::choiceAndAct()
{
	int selectionInput;

	//prompt for user action
	cout << "Please select the following options for various operations in the loaded image" << endl;
	cout << "[10]: Convolution2D (Single Kernel)\t\t[11]: RGB To Gray Conversion" << endl <<
		"[20]: Laplace Edge Detection\t\t[21]: Embossing" << endl <<
		"[22]: Prewitt Edge Detection Dx\t\t[23]: Prewitt Edge Detection Dy" << endl <<
		"[24]: Prewitt Edge Detection Dm\t\t[25]: Sobel Edge Detection Dx" << endl <<
		"[26]: Sobel Edge Detection Dy\t\t[27]: Sobel Edge Detection Dm" << endl <<
		"[28]: Laplace Edge Detection with Smoothing\t\t[30]: Histogram Equalization" << endl <<
		"[31]: Contrast Enhancement (Power Law)\t\t[32]: Contrast Enhancement (Linear  Streatching)" << endl <<
		"[33]: (Gaussian) Blur Image\t\t[34]: (Mean) Blur Image" << endl <<
		"[35]: Sharpening Image" << endl <<
		"**********************Serial Operations*****************************" << endl <<
		"[120]: Laplace Edge Detection Serial\t\t[121]: Embossing Serial" << endl <<
		"[122]: Prewitt Edge Detection Dx Serial\t\t[123]: Prewitt Edge Detection Dy Serial" << endl <<
		"[124]: Prewitt Edge Detection Dm Serial\t\t[125]: Sobel Edge Detection Dx Serial" << endl <<
		"[126]: Sobel Edge Detection Dy Serial\t\t[127]: Sobel Edge Detection Dm Serial" << endl <<
		"[128]: Laplace Edge Detection with Smoothing Serial\t\t[130]: Histogram Equalization Serial" << endl <<
		"[131]: Contrast Enhancement (Power Law) Serial\t\t[132]: Contrast Enhancement (Linear  Streatching) Serial" << endl <<
		"[133]: (Gaussian) Blur Image Serial\t\t[134]: (Mean) Blur Image Serial" << endl <<
		"[135]: Sharpening Image Serial\t\t[136]: Prewitt Edge Detection With Smoothing Serial" << endl<<
		"[137]: Sobel Edge Detection With Smoothing Serial\t\t[110]: Convolution2D Serial"<<endl;
	std::cin >> selectionInput;


	/*
	10-19 are for general filtering usage
	20-29 are for edge based operation
	30-39 are for histogram/contrast based operation
	*/
	switch (selectionInput)
	{
	case 10:
		cout << "You have selected option 10" << endl;
		return callConvolution2D();
	case 11:
		cout << "You have selected option 11" << endl;
		return rgbToGray();
	case 20:
		cout << "You have selected option 20" << endl;
		return laplaceEdgeDetection();
	case 21:
		cout << "You have selected option 21" << endl;
		return embossing();
	case 22:
		cout << "You have selected option 22" << endl;
		return prewittEdgeDetectionDx();
	case 23:
		cout << "You have selected option 23" << endl;
		return prewittEdgeDetectionDy();
	case 24:
		cout << "You have selected option 24" << endl;
		return prewittEdgeDetectionDm();
	case 25:
		cout << "You have selected option 25" << endl;
		return sobelEdgeDetectionDx();
	case 26:
		cout << "You have selected option 26" << endl;
		return sobelEdgeDetectionDy();
	case 27:
		cout << "You have selected option 27" << endl;
		return sobelEdgeDetectionDm();
	case 28:
		cout << "You have selected option 20" << endl;
		return laplaceEdgeDetectionWithSmoothing();
	case 30:
		cout << "You have selected option 30" << endl;
		return histogramEqualization();
	case 31:
		cout << "You have selected option 31" << endl;
		return powerLawHistogram();
	case 32:
		cout << "You have selected option 32" << endl;
		return linearStretchingContrast();
	case 33:
		cout << "You have selected option33" << endl;
		return gaussianBlur();
	case 34:
		cout << "You have selected option 34" << endl;
		return meanBlurImage();
	case 35:
		cout << "You have selected option 35" << endl;
		return sharpening();
	case 110:
		cout << "You have selected option 10" << endl;
		return callConvolution2DSerial();
	case 111:
		cout << "You have selected option 11" << endl;
		return rgbToGraySerial();
	case 120:
		cout << "You have selected option 20" << endl;
		return laplaceEdgeDetectionSerial();
	case 121:
		cout << "You have selected option 21" << endl;
		return embossingSerial();
	case 122:
		cout << "You have selected option 22" << endl;
		return prewittEdgeDetectionDxSerial();
	case 123:
		cout << "You have selected option 23" << endl;
		return prewittEdgeDetectionDySerial();
	case 124:
		cout << "You have selected option 24" << endl;
		return prewittEdgeDetectionDmSerial();
	case 125:
		cout << "You have selected option 25" << endl;
		return sobelEdgeDetectionDxSerial();
	case 126:
		cout << "You have selected option 26" << endl;
		return sobelEdgeDetectionDySerial();
	case 127:
		cout << "You have selected option 27" << endl;
		return sobelEdgeDetectionDmSerial();
	case 128:
		cout << "You have selected option 20" << endl;
		return laplaceEdgeDetectionWithSmoothingSerial();
	case 130:
		cout << "You have selected option 30" << endl;
		return histogramEqualizationSerial();
	case 131:
		cout << "You have selected option 31" << endl;
		return powerLawHistogramSerial();
	case 132:
		cout << "You have selected option 32" << endl;
		return linearStretchingContrastSerial();
	case 133:
		cout << "You have selected option33" << endl;
		return gaussianBlurSerial();
	case 134:
		cout << "You have selected option 34" << endl;
		return meanBlurImageSerial();
	case 135:
		cout << "You have selected option 35" << endl;
		return sharpeningSerial();
	case 136:
		return prewittEdgeDetectionDmSerialWithSmoothing();
	case 137:
		return sobelEdgeDetectionDmSerialWithSmoothing();
	default:
		cout << "Invalid input" << endl;
		return false;
		break;
	}
}

bool mainApplication::sobelEdgeDetectionDmSerial()
{
	QueryPerformanceFrequency(&Frequency);

	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	QueryPerformanceCounter(&StartingTime);

	uchar* tempPixels;
	tempPixels = new uchar[numberOfColumns*numberOfRows];

	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			tempPixels[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < numberOfColumns - 1; col++)
		{
			//this is fine
			uint8_t tempX = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * -2
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 2
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * 1));

			uint8_t tempY = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * 1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 2
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * -2
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * -1));

			float temp = abs(sqrt((float)(tempX*tempX) + (float)(tempY*tempY)));
			if (temp > 200)
			{
				hostCharOutput[(row)*numberOfColumns + (col)].x = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].y = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].z = (uchar)temp;
			}
			else
			{
				hostCharOutput[(row)*numberOfColumns + (col)].x = 0;
				hostCharOutput[(row)*numberOfColumns + (col)].y = 0;
				hostCharOutput[(row)*numberOfColumns + (col)].z = 0;
			}
		}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cv::cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	delete[] tempPixels;

	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}

bool mainApplication::prewittEdgeDetectionDmSerialWithSmoothing()
{

	cv::cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	uchar *tempPixels,*tempPixelsInitial;
	tempPixels = new uchar[numberOfColumns*numberOfRows];
	tempPixelsInitial = new uchar[numberOfColumns*numberOfRows];

	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			tempPixelsInitial[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);

	//mean smoothing of the image is done here in this section
	for (int i = 1; i < numberOfRows - 1; i++)
		for (int j = 1; j < numberOfColumns - 1; j++)
			tempPixels[i*numberOfColumns + j] = (uchar)((float)tempPixelsInitial[i*numberOfColumns + (j-1)]+ (float)tempPixelsInitial[i*numberOfColumns + j] + (float)tempPixelsInitial[i*numberOfColumns + (j+1)] + (float)tempPixelsInitial[(i-1)*numberOfColumns + (j-1)] + (float)tempPixelsInitial[(i-1)*numberOfColumns + j] + (float)tempPixelsInitial[(i-1)*numberOfColumns + (j+1)] + (float)tempPixelsInitial[(i + 1)*numberOfColumns + (j - 1)] + (float)tempPixelsInitial[(i + 1)*numberOfColumns + j] + (float)tempPixelsInitial[(i + 1)*numberOfColumns + (j + 1)]/9);

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < numberOfColumns - 1; col++)
		{
			//this is fine
			uint8_t tempX = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * -2
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 2
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * 1));

			uint8_t tempY = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * 1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 2
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * -2
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * -1));

			float temp = abs(sqrt((float)(tempX*tempX) + (float)(tempY*tempY)));
			if (temp > 200)
			{
				hostCharOutput[(row)*numberOfColumns + (col)].x =  (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].y = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].z = (uchar)temp;
			}
			else
			{
				hostCharOutput[(row)*numberOfColumns + (col)].x = 0;
				hostCharOutput[(row)*numberOfColumns + (col)].y = 0;
				hostCharOutput[(row)*numberOfColumns + (col)].z = 0;
			}
		}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cv::cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	delete[] tempPixels;

	return true;
}
bool mainApplication::sobelEdgeDetectionDmSerialWithSmoothing()
{
	cv::cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	uchar *tempPixels, *tempPixelsInitial;
	tempPixels = new uchar[numberOfColumns*numberOfRows];
	tempPixelsInitial = new uchar[numberOfColumns*numberOfRows];

	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			tempPixelsInitial[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);

	//mean smoothing of the image is done here in this section
	for (int i = 1; i < numberOfRows - 1; i++)
		for (int j = 1; j < numberOfColumns - 1; j++)
			tempPixels[i*numberOfColumns + j] = (uchar)((float)tempPixelsInitial[i*numberOfColumns + (j - 1)] + (float)tempPixelsInitial[i*numberOfColumns + j] + (float)tempPixelsInitial[i*numberOfColumns + (j + 1)] + (float)tempPixelsInitial[(i - 1)*numberOfColumns + (j - 1)] + (float)tempPixelsInitial[(i - 1)*numberOfColumns + j] + (float)tempPixelsInitial[(i - 1)*numberOfColumns + (j + 1)] + (float)tempPixelsInitial[(i + 1)*numberOfColumns + (j - 1)] + (float)tempPixelsInitial[(i + 1)*numberOfColumns + j] + (float)tempPixelsInitial[(i + 1)*numberOfColumns + (j + 1)] / 9);

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < numberOfColumns - 1; col++)
		{
			//this is fine
			uint8_t tempX = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * -2
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 2
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * 1));

			uint8_t tempY = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * 1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 2
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * -2
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * -1));

			float temp = abs(sqrt((float)(tempX*tempX) + (float)(tempY*tempY)));
			if (temp > 200)
			{
				hostCharOutput[(row)*numberOfColumns + (col)].x = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].y = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].z = (uchar)temp;
			}
			else
			{
				hostCharOutput[(row)*numberOfColumns + (col)].x = 0;
				hostCharOutput[(row)*numberOfColumns + (col)].y = 0;
				hostCharOutput[(row)*numberOfColumns + (col)].z = 0;
			}
		}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	delete[] tempPixels;


	return true;
}
bool mainApplication::meanBlurImageSerial()
{

	QueryPerformanceFrequency(&Frequency);

	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t noOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[noOfColumns*numberOfRows];
	uchar4* inputPixels;
	inputPixels = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	QueryPerformanceCounter(&StartingTime);
	float div = 9;

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < noOfColumns - 1; col++)
		{
			hostCharOutput[row*noOfColumns + col].x = (inputPixels[row*noOfColumns + col].x / div + inputPixels[(row + 1)*noOfColumns + col].x / div + inputPixels[(row - 1)*noOfColumns + col].x / div + inputPixels[row*noOfColumns + (col + 1)].x / div + inputPixels[row*noOfColumns + (col - 1)].x / div + inputPixels[(row - 1)*noOfColumns + (col - 1)].x / div + inputPixels[(row - 1)*noOfColumns + (col + 1)].x / div + inputPixels[(row + 1)*noOfColumns + (col - 1)].x / div + inputPixels[(row + 1)*noOfColumns + (col + 1)].x / div);
		}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, noOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	return true;
}
bool mainApplication::gaussianBlurSerial()
{
	QueryPerformanceFrequency(&Frequency);

	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t noOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[noOfColumns*numberOfRows];
	uchar4* inputPixels;
	inputPixels = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	QueryPerformanceCounter(&StartingTime);
	float div = 9;

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < noOfColumns - 1; col++)
		{
			hostCharOutput[row*noOfColumns + col].x = (4 *(float) inputPixels[row*noOfColumns + col].x / div + 2 * (float)inputPixels[(row + 1)*noOfColumns + col].x / div + 2 *(float) inputPixels[(row - 1)*noOfColumns + col].x / div + 2 *(float) inputPixels[row*noOfColumns + (col + 1)].x / div + 2 * (float)inputPixels[row*noOfColumns + (col - 1)].x / div +(float) inputPixels[(row - 1)*noOfColumns + (col - 1)].x / div + (float)inputPixels[(row - 1)*noOfColumns + (col + 1)].x / div + (float)inputPixels[(row + 1)*noOfColumns + (col - 1)].x / div + (float)inputPixels[(row + 1)*noOfColumns + (col + 1)].x / div);
		}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, noOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}
bool mainApplication::sharpeningSerial()
{
	return true;
}
bool mainApplication::callConvolution2DSerial()
{
	uchar* maskKernel = new uchar[3 * 3];
	cout << "Enter the elements of 3X3 convolution kernel. Row major order. First 3 elements are considered to be of first row and so on" << endl;
	for (int i = 0; i < 9; i++)
		cin >> maskKernel[i];

	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	uchar* tempPixels;
	tempPixels = new uchar[numberOfColumns*numberOfRows];


	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);

	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			tempPixels[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < numberOfColumns - 1; col++)
		{
			//this is fine
			uint8_t temp = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * maskKernel[0] 
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * maskKernel[1]
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)] * maskKernel[2]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * maskKernel[3]
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * maskKernel[4]
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * maskKernel[5]
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * maskKernel[6]
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * maskKernel[7]
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * maskKernel[8]));

			if (temp > 200) {
				hostCharOutput[(row)*numberOfColumns + (col)].x = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].y = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].z = (uchar)temp;
			}
			else
			{
				hostCharOutput[(row)*numberOfColumns + (col)].x = 0;// (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].y = 0;// (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].z = 0;// (uchar)temp;
			}
		}


	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;


	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	delete[] tempPixels;

	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	return true;
}
bool mainApplication::embossingSerial()
{
	return true;
}
bool mainApplication::rgbToGraySerial()
{
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar* hostCharOutput = new uchar[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	
	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			hostCharOutput[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);


	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC1, (void*)hostCharOutput);

	//since the formed result is single channel image the output is directly displayed from
	//here. Output image will contain the input image
	if (!tempOutputMat.data)
	{
		cout << "No data to display. Probably Error from kernel!!!" << endl;
		return false;
	}

	outputImageMat = inputImageMat;

	destroyWindow("Output Image");
	namedWindow("Output Image", WINDOW_AUTOSIZE);
	imshow("Output Image", tempOutputMat);
	waitKey(1);

	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}
bool mainApplication::laplaceEdgeDetectionSerial()
{	//0  1  0
	//1 -4  1
	//0  1  0
	
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;


	uchar4* hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);

	uchar* tempPixels;
	tempPixels = new uchar[numberOfColumns*numberOfRows];

	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			tempPixels[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < numberOfColumns - 1; col++)
		{
			//this is fine
			uint8_t temp = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * 0
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]*0
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * 1
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * -4
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * 1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * 0));

			if (temp > 200) {
				hostCharOutput[(row)*numberOfColumns + (col)].x = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].y = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].z = (uchar)temp;
			}
			else
			{
				hostCharOutput[(row)*numberOfColumns + (col)].x = 0;// (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].y = 0;// (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].z = 0;// (uchar)temp;
			}
		}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	delete[] tempPixels;

	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	return true;
}
bool mainApplication::laplaceEdgeDetectionWithSmoothingSerial()
{
	return true;
}
bool mainApplication::powerLawHistogramSerial()
{
	
	if (!inputImageMat.data)
	{
		return false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);

	uchar4 *hostCharInput, *hostCharOutput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);

	double c = 1;
	double r = 0.3;
	hostCharOutput = new uchar4[numberOfRows*numberOfColumns];
	for (int row = 0; row<numberOfRows; row++)
		for (int col = 0; col < numberOfColumns; col++)
		{
			hostCharOutput[row*numberOfColumns + col].x = c * pow((double)hostCharInput[row*numberOfColumns + col].x, r);
			hostCharOutput[row*numberOfColumns + col].y = c * pow((double)hostCharInput[row*numberOfColumns + col].y, r); 
			hostCharOutput[row*numberOfColumns + col].z = c * pow((double)hostCharInput[row*numberOfColumns + col].z, r);
			//hostCharOutput[row*numberOfColumns + col].x = log10(1+ (double)hostCharInput[row*numberOfColumns + col].x);
			//hostCharOutput[row*numberOfColumns + col].y = log10(1 + (double)hostCharInput[row*numberOfColumns + col].y);
			//hostCharOutput[row*numberOfColumns + col].z = log10(1 + (double)hostCharInput[row*numberOfColumns + col].z);
		}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;

	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}
bool mainApplication::linearStretchingContrastSerial()
{
	if (!inputImageMat.data)
	{
		return false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);

	uchar4 *hostCharInput, *hostCharOutput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;


	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	hostCharOutput = new uchar4[numberOfRows*numberOfColumns];
	
	uchar3 minPixVal,maxPixVal;
	minPixVal.x = 255;
	minPixVal.y = 255;
	minPixVal.z = 255;
	maxPixVal.x = 0;
	maxPixVal.y = 0;
	maxPixVal.z = 0;

	for(int row=0;row<numberOfRows;row++)
		for (int col = 0; col < numberOfColumns; col++)
		{
			if (minPixVal.x > hostCharInput[row*numberOfColumns + col].x) minPixVal.x = hostCharInput[row*numberOfColumns + col].x;
			if (minPixVal.y > hostCharInput[row*numberOfColumns + col].y) minPixVal.y = hostCharInput[row*numberOfColumns + col].y;
			if (minPixVal.z > hostCharInput[row*numberOfColumns + col].z) minPixVal.z = hostCharInput[row*numberOfColumns + col].z;

			if (maxPixVal.x < hostCharInput[row*numberOfColumns + col].x) maxPixVal.x = hostCharInput[row*numberOfColumns + col].x;
			if (maxPixVal.y < hostCharInput[row*numberOfColumns + col].y) maxPixVal.y = hostCharInput[row*numberOfColumns + col].y;
			if (maxPixVal.z < hostCharInput[row*numberOfColumns + col].z) maxPixVal.z = hostCharInput[row*numberOfColumns + col].z;
		}

	for (int row = 0; row < numberOfRows; row++)
	{
		for (int col = 0; col < numberOfColumns; col++)
		{
			hostCharOutput[row*numberOfColumns + col].x = ((((float)hostCharInput[row*numberOfColumns + col].x - (float)minPixVal.x)*(float)255 / ((float)maxPixVal.x - (float)minPixVal.x)));
			hostCharOutput[row*numberOfColumns + col].y = ((((float)hostCharInput[row*numberOfColumns + col].y - (float)minPixVal.y)*(float)255 / ((float)maxPixVal.y - (float)minPixVal.y)));
			hostCharOutput[row*numberOfColumns + col].z = ((((float)hostCharInput[row*numberOfColumns + col].z - (float)minPixVal.z)*(float)255 / ((float)maxPixVal.z - (float)minPixVal.z)));
		}
	}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}
bool mainApplication::gaussianBlur()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	cudaProfilerStart();

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns/32), ceil(numberOfRows/32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput,&numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&gaussianBlurKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}
	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);
	cudaProfilerStop();

	delete[] hostCharOutput;
	//free(hostCharInput);
	return true;

}

bool mainApplication::callConvolution2D()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;
	uchar *tempPixels;

	bool successReading;
	int maskWidth, maskHeight;	//noOfCols and noOfRows of mask matrix
	float *maskMatrix;
	float *maskMatrixD;
	do
	{
		char inY;
		cout << "Enter the dimension of mask matrix\n\tPlease enter the NoOfRows followed by NoOfColumns" << endl;
		cin >> maskHeight >> maskWidth;

		maskMatrix = (float*) malloc(sizeof(float)*maskHeight*maskWidth);
		cout << "Enter the elements of the mask matrix in row major order. First " << maskWidth << " elements will be first row and so  on.." << endl;
		for (int i = 0; i < maskHeight*maskWidth; i++)
			cin >> maskMatrix[i];
		
		cout << "Please confirm the mask matrix you entered" << endl;
		for (int i = 0; i < maskHeight*maskWidth; i++)
		{
			cout << maskMatrix[i]<<" ";
			if (i % (maskWidth) == maskWidth - 1) cout << endl;
		}

		cout << "is it correct?(Y/N)" << endl;
		cin >> inY;
		if (inY == 'Y' || inY == 'y')	successReading = true;
		else successReading = false;
	} while (successReading == false);

	Mat tempOutputRGBA;
	cudaError_t cudaStatus;

	if (!inputImageMat.data)
	{
		return false;
	}
	cvtColor(inputImageMat,tempImageMat,CV_BGR2RGBA);

	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	cudaProfilerStart();
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned char>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&tempPixels, sizeof(uchar)*numberOfRows*numberOfColumns);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of temp data inside the conv function" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&maskMatrixD,sizeof(float)*maskHeight*maskWidth);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc for mask matrix" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(maskMatrixD, maskMatrix, sizeof(float)*maskHeight*maskWidth, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying mask matrix from host to device" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of output data inside the histogram function" << endl;
		return false;
	}

	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying input image from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharInput ,&deviceCharOutput, &tempPixels, &maskMatrixD ,&numberOfRows,&numberOfColumns,&maskWidth,&maskHeight};
	//void convolution2DKernel(uchar4* inputPixels, uchar4* outputPixels, uchar* maskKernel2D, int noOfRows, int noOfColumns, int maskWidth, int maskHeight);
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&convolution2DKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed with status "<<cudaStatus << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);
	cudaFree(maskMatrixD);

	free(maskMatrix);
	delete[] hostCharOutput;
	//free(hostCharInput);
	cudaProfilerStop();

	return true;
}

bool mainApplication::sharpening()
{

	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	cudaProfilerStart();
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput,&numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&sharpeningKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}
	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);
	cudaProfilerStop();


	delete[] hostCharOutput;
	//free(hostCharInput);

	return true;

}

bool mainApplication::meanBlurImage()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput,sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput,sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput,&numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&meanBlurKernel,gridDimm,blockDimm,args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);

	cudaProfilerStop();
	delete[] hostCharOutput;
	//free(hostCharInput);

	return true;	
}

bool mainApplication::laplaceEdgeDetection()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;
	uchar* *deviceTemp;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceTemp, sizeof(uchar*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput, &deviceTemp, &numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&laplaceEdgeDetectionKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);

	cudaProfilerStop();

	delete[] hostCharOutput;
	//free(hostCharInput);

	return true;
}

bool mainApplication::laplaceEdgeDetectionWithSmoothing()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;
	uchar* *deviceTemp,*deviceTemp2;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceTemp, sizeof(uchar*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceTemp2, sizeof(uchar*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput, &deviceTemp,&deviceTemp2, &numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&laplaceEdgeDetectionKernelWithSmoothing, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);

	cudaProfilerStop();

	delete[] hostCharOutput;
	//free(hostCharInput);

	return true;
}

bool mainApplication::prewittEdgeDetectionDx()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;
	uchar* *deviceTemp;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceTemp, sizeof(uchar*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput, &deviceTemp, &numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&prewittEdgeDetectionDxKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);


	cudaProfilerStop();
	delete[] hostCharOutput;
	//free(hostCharInput);

	return true;
}
bool mainApplication::sobelEdgeDetectionDxSerial()
{
	
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);

	uchar* tempPixels;
	tempPixels = new uchar[numberOfColumns*numberOfRows];

	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			tempPixels[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < numberOfColumns - 1; col++)
		{
			//this is fine
			uint8_t temp = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * -2
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 2
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * 1));

			hostCharOutput[(row)*numberOfColumns + (col)].x = (uchar)temp;
			hostCharOutput[(row)*numberOfColumns + (col)].y = (uchar)temp;
			hostCharOutput[(row)*numberOfColumns + (col)].z = (uchar)temp;
		}
	QueryPerformanceCounter(&EndingTime);

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	delete[] tempPixels;

	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}
bool mainApplication::sobelEdgeDetectionDySerial()
{
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);

	uchar* tempPixels;
	tempPixels = new uchar[numberOfColumns*numberOfRows];

	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			tempPixels[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < numberOfColumns - 1; col++)
		{
			uint8_t temp = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * 1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 2
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * -2
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * -1));

			hostCharOutput[(row)*numberOfColumns + (col)].x = (uchar)temp;
			hostCharOutput[(row)*numberOfColumns + (col)].y = (uchar)temp;
			hostCharOutput[(row)*numberOfColumns + (col)].z = (uchar)temp;
		}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	delete[] tempPixels;

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}
bool mainApplication::prewittEdgeDetectionDxSerial()
{
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	uchar* tempPixels;
	tempPixels = new uchar[numberOfColumns*numberOfRows];

	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			tempPixels[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);

	//this section actually does the convolution using the prewitt Dx filter
	for(int row=1; row < numberOfRows -1 ; row++ )
		for (int col = 1; col < numberOfColumns - 1; col++)
		{
			//this is fine
			uint8_t temp = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * 1));
			
			hostCharOutput[(row)*numberOfColumns + (col)].x = (uchar)temp;
			hostCharOutput[(row)*numberOfColumns + (col)].y = (uchar)temp;
			hostCharOutput[(row)*numberOfColumns + (col)].z = (uchar)temp;
		}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	delete[] tempPixels;

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}
bool mainApplication::prewittEdgeDetectionDmSerial()
{
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	uchar* tempPixels;
	tempPixels = new uchar[numberOfColumns*numberOfRows];

	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			tempPixels[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < numberOfColumns - 1; col++)
		{
			//this is fine
			uint8_t tempX = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * 1));
			
			uint8_t tempY = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * 1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * -1));

			float temp = abs(sqrt((float)(tempX*tempX) + (float)(tempY*tempY)));
			if (temp > 200)
			{
				hostCharOutput[(row)*numberOfColumns + (col)].x = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].y = (uchar)temp;
				hostCharOutput[(row)*numberOfColumns + (col)].z = (uchar)temp;
			}
			else
			{
				hostCharOutput[(row)*numberOfColumns + (col)].x = 0;
				hostCharOutput[(row)*numberOfColumns + (col)].y = 0;
				hostCharOutput[(row)*numberOfColumns + (col)].z = 0;
			}
		}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	delete[] tempPixels;

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}
bool mainApplication::prewittEdgeDetectionDySerial()
{
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar4* hostCharOutput = new uchar4[numberOfColumns*numberOfRows];
	uchar4* hostCharInput;
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);

	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	uchar* tempPixels;
	tempPixels = new uchar[numberOfColumns*numberOfRows];

	//following section converts the RGBA into GrayScale image
	for (int i = 0; i < numberOfRows; i++)
		for (int j = 0; j < numberOfColumns; j++)
			tempPixels[i*numberOfColumns + j] = (uchar)(0.299* (float)hostCharInput[i*numberOfColumns + j].x + 0.587* (float)hostCharInput[i*numberOfColumns + j].y + 0.114* (float)hostCharInput[i*numberOfColumns + j].z);

	//this section actually does the convolution using the prewitt Dx filter
	for (int row = 1; row < numberOfRows - 1; row++)
		for (int col = 1; col < numberOfColumns - 1; col++)
		{
			//this is fine
			uint8_t temp = ((uint8_t)abs((float)tempPixels[(row - 1)*numberOfColumns + (col - 1)] * 1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col)] * 1
				+ (float)tempPixels[(row - 1)*numberOfColumns + (col + 1)]
				+ (float)tempPixels[(row)*numberOfColumns + (col - 1)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col)] * 0
				+ (float)tempPixels[(row)*numberOfColumns + (col + 1)] * 0
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col - 1)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col)] * -1
				+ (float)tempPixels[(row + 1)*numberOfColumns + (col + 1)] * -1));

			hostCharOutput[(row)*numberOfColumns + (col)].x = (uchar)temp;
			hostCharOutput[(row)*numberOfColumns + (col)].y = (uchar)temp;
			hostCharOutput[(row)*numberOfColumns + (col)].z = (uchar)temp;
		}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] hostCharOutput;
	delete[] tempPixels;

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}

bool mainApplication::prewittEdgeDetectionDy()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;
	uchar* *deviceTemp;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	cudaProfilerStart();
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceTemp, sizeof(uchar*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput, &deviceTemp, &numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&prewittEdgeDetectionDyKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);


	delete[] hostCharOutput;
	cudaProfilerStop();
	//free(hostCharInput);

	return true;
}
bool mainApplication::prewittEdgeDetectionDm()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;
	uchar* *deviceTemp;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceTemp, sizeof(uchar*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput, &deviceTemp, &numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&prewittEdgeDetectionDmKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);


	delete[] hostCharOutput;
	//free(hostCharInput);
	cudaProfilerStop();

	return true;
}
bool mainApplication::sobelEdgeDetectionDx()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;
	uchar* *deviceTemp;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceTemp, sizeof(uchar*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput, &deviceTemp, &numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&sobelEdgeDetectionDxKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);

	cudaProfilerStop();

	delete[] hostCharOutput;
	//free(hostCharInput);

	return true;
}
bool mainApplication::sobelEdgeDetectionDy()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;
	uchar* *deviceTemp;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceTemp, sizeof(uchar*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput, &deviceTemp, &numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&sobelEdgeDetectionDyKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);


	cudaProfilerStop();
	delete[] hostCharOutput;
	//free(hostCharInput);

	return true;
}
bool mainApplication::sobelEdgeDetectionDm()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;
	uchar* *deviceTemp;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceTemp, sizeof(uchar*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput, &deviceTemp, &numberOfRows ,&numberOfColumns };
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	//call kernel here
	cudaStatus = cudaLaunchKernel((const void*)&sobelEdgeDetectionDmKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);


	delete[] hostCharOutput;
	cudaProfilerStop();

	return true;
}
bool mainApplication::powerLawHistogram()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput,&numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&powerLawHistogramKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}
	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);

	delete[] hostCharOutput;
	cudaProfilerStop();
	//free(hostCharInput);
	return true;
}
bool mainApplication::linearStretchingContrast()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4 *deviceCharInput, *deviceCharOutput;

	//its better to copy the whole input image data into the shared memory
	int stride = 256;	//this will represent the maximum allowable threads in a block

	Mat tempOutputRGBA;
	cudaError_t cudaStatus;
	bool successStatus = true;

	if (!inputImageMat.data)
	{
		successStatus = false;
		cout << "No input data found.\n Program Exiting" << endl;
		return successStatus;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);

	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfRows*numberOfColumns);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in malloc operation in device. Operation exiting" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in malloc operation in device. Operation exiting" << endl;
		return false;
	}

	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in copying input data from host to device" << endl;
		return false;
	}

	dim3 gridDimension(1);
	dim3 blockDimension(stride);

	//shared memory will now contain the input pixels and input pixels
	size_t sharedMemSize = 6 * sizeof(int) * 256;

	void* args[] = { &deviceCharOutput ,&deviceCharInput,&numberOfRows ,&numberOfColumns,&stride };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&linearStretchingContrastKernel, gridDimension, blockDimension, args, sharedMemSize);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);

	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}
	return successStatus;

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);


	delete[] hostCharOutput;
	cudaProfilerStop();
	//free(hostCharInput);

	return true;
}
bool mainApplication::embossing()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4* deviceCharInput, *deviceCharOutput;

	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool successStatus = false;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar4*)*numberOfColumns*numberOfRows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}

	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput,&numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&embossingKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);


	delete[] hostCharOutput;
	//free(hostCharInput);

	cudaProfilerStop();
	return true;
}

bool mainApplication::histogramEqualizationSerial()
{
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	if (!inputImageMat.data)
	{
		cout << "No input data found.\n Program Exiting" << endl;
		return false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);
	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;

	uchar4 *hostCharInput, *hostCharOutput;
	long long *histogramR, *histogramG, *histogramB;
	long long *cumHistogramR, *cumHistogramG, *cumHistogramB;

	histogramR = new long long[256];
	histogramG = new long long[256];
	histogramB = new long long[256];
	cumHistogramR = new long long[256];
	cumHistogramG = new long long[256];
	cumHistogramB = new long long[256];

	//initialize to zero
	for (int i = 0; i < 256; i++)
	{
		histogramR[i] = 0;
		histogramG[i] = 0;
		histogramB[i] = 0;
	}
	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfRows*numberOfColumns];

	//finding histogram
	for (int row = 0; row < numberOfRows; row++)
	{
		for (int col = 0; col < numberOfColumns; col++)
		{
			histogramR[(int)hostCharInput[row*numberOfColumns + col].x] += 1;
			histogramG[(int)hostCharInput[row*numberOfColumns + col].y] += 1;
			histogramB[(int)hostCharInput[row*numberOfColumns + col].z] += 1;
		}
	}
	//finding cumulative histogram inclusive
	cumHistogramR[0] = histogramR[0];
	cumHistogramG[0] = histogramG[0];
	cumHistogramB[0] = histogramB[0];
	for (int i = 1; i < 256; i++)
	{
		cumHistogramR[i] = cumHistogramR[i - 1] + histogramR[i];
		cumHistogramG[i] = cumHistogramG[i - 1] + histogramG[i];
		cumHistogramB[i] = cumHistogramB[i - 1] + histogramB[i];
		//cout << i << " " << histogramR[i] << " " << cumHistogramR[i] << " " << histogramG[i] << " " << cumHistogramG[i] << " " << histogramB[i] << " " << cumHistogramB[i] << endl;
	}

	//finding non zero minimum value
	long long minR = ULONG_MAX;
	long long minG = ULONG_MAX;
	long long minB = ULONG_MAX;

	long long maxR = 0;
	long long maxG = 0;
	long long maxB = 0;

	for (int i = 0; i < 256; i++)
	{
		if (cumHistogramR[i]>0 && cumHistogramR[i] < minR) minR = cumHistogramR[i];
		if (cumHistogramG[i]>0 && cumHistogramG[i] < minG) minG = cumHistogramG[i];
		if (cumHistogramB[i]>0 && cumHistogramB[i] < minB) minB = cumHistogramB[i];
	}
	

	//cdf stretching part
	for (int row = 0; row< numberOfRows; row++)
		for (int col = 0; col < numberOfColumns; col++)
		{
			hostCharOutput[row*numberOfColumns + col].x = ((cumHistogramR[(int)hostCharInput[row*numberOfColumns + col].x] - minR) * (float)255 / (float)(numberOfRows*numberOfColumns- 1));
			hostCharOutput[row*numberOfColumns + col].y = ((cumHistogramG[(int)hostCharInput[row*numberOfColumns + col].y] - minG) * (float)255 / (float)(numberOfRows*numberOfColumns- 1));
			hostCharOutput[row*numberOfColumns + col].z = ((cumHistogramB[(int)hostCharInput[row*numberOfColumns + col].z] - minB) * (float)255 / (float)(numberOfRows*numberOfColumns- 1));
		}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);
	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}

	delete[] histogramR;
	delete[] histogramG;
	delete[] histogramB;

	delete[] cumHistogramR;
	delete[] cumHistogramG;
	delete[] cumHistogramB;
	
	delete[] hostCharOutput;
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	return true;
}
bool mainApplication::histogramEqualization()
{
	uchar4 *hostCharInput, *hostCharOutput;
	uchar4 *deviceCharInput, *deviceCharOutput;

	//its better to copy the whole input image data into the shared memory
	int stride = 256;	//this will represent the maximum allowable threads in a block

	Mat tempOutputRGBA;
	cudaError_t cudaStatus;
	bool successStatus = true;

	if (!inputImageMat.data)
	{
		successStatus = false;
		cout << "No input data found.\n Program Exiting" << endl;
		return successStatus;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);

	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned int>(0);
	hostCharOutput = new uchar4[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4*)*numberOfRows*numberOfColumns);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in malloc operation in device. Operation exiting" << endl;
		return false;
	}

	cudaStatus = cudaMalloc((void**)&deviceCharOutput,sizeof(uchar4)*numberOfRows*numberOfColumns);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in malloc operation in device. Operation exiting" << endl;
		return false;
	}
	
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in copying input data from host to device" << endl;
		return false;
	}

	dim3 gridDimension(1);
	dim3 blockDimension(stride);

	//shared memory will now contain the input pixels and input pixels
	size_t sharedMemSize = 6 * sizeof(int)*256;

	void* args[] = { &deviceCharOutput ,&deviceCharInput,&numberOfRows ,&numberOfColumns,&stride };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&histogramEqualizationKernel, gridDimension, blockDimension, args, sharedMemSize);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;
	
	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC4, (void*)hostCharOutput);

	//converting the format from RGBA to BGR which is standard in OpenCV
	cvtColor(tempOutputMat, outputImageMat, CV_RGBA2BGR);

	if (displayOutputImage() != 0)
	{
		cout << "Error in displaying image" << endl;
		return false;
	}
	return successStatus;

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);


	delete[] hostCharOutput;
	cudaProfilerStop();
	//free(hostCharInput);

	return true;
}
bool mainApplication::rgbToGray()
{
	uchar4 *hostCharInput;
	uchar  *hostCharOutput;
	uchar4 *deviceCharInput;
	uchar  *deviceCharOutput;


	Mat tempOutputRGBA;

	cudaError_t cudaStatus;
	bool  successStatus = true;
	if (!inputImageMat.data)
	{
		successStatus = false;
	}
	cvtColor(inputImageMat, tempImageMat, CV_BGR2RGBA);

	size_t numberOfRows = tempImageMat.rows;
	size_t numberOfColumns = tempImageMat.cols;
	cudaProfilerStart();

	hostCharInput = (uchar4*)tempImageMat.ptr<unsigned char>(0);
	hostCharOutput = new uchar[numberOfColumns*numberOfRows];

	cudaStatus = cudaMalloc((void**)&deviceCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of input data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void**)&deviceCharOutput, sizeof(uchar)*numberOfRows*numberOfColumns);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Malloc of output data inside the histogram function" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(deviceCharInput, hostCharInput, sizeof(uchar4)*numberOfRows*numberOfColumns, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in Copying from host to device" << endl;
		return false;
	}
	//fixing the grid and block dimension to launch the kernel
	dim3 blockDimm(32, 32);	//max threads 
	dim3 gridDimm(ceil(numberOfColumns / 32), ceil(numberOfRows / 32));

	void* args[] = { &deviceCharOutput ,&deviceCharInput,&numberOfRows ,&numberOfColumns };
	//call kernel here
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	cudaStatus = cudaLaunchKernel((const void*)&gaussianRGB2GrayKernel, gridDimm, blockDimm, args);
	//Kernel call ends here
	if (cudaStatus != cudaSuccess)
	{
		cout << "Kernel Launched failed" << endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Device synchronize failed" << endl;
		return false;
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	cout << "Time elapsed is " << ElapsedMicroseconds.QuadPart << " miliseconds" << endl;

	//copying back from device to host
	cudaStatus = cudaMemcpy(hostCharOutput, deviceCharOutput, sizeof(uchar)*numberOfRows*numberOfColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error copying data from Device to Host" << endl;
		return false;
	}

	//creating Mat type object from the data received from Device
	Mat tempOutputMat(numberOfRows, numberOfColumns, CV_8UC1, (void*)hostCharOutput);

	//since the formed result is single channel image the output is directly displayed from
	//here. Output image will contain the input image
	if (!tempOutputMat.data)
	{
		cout << "No data to display. Probably Error from kernel!!!" << endl;
		return false;
	}

	outputImageMat = inputImageMat;

	destroyWindow("Output Image");
	namedWindow("Output Image", WINDOW_AUTOSIZE);
	imshow("Output Image", tempOutputMat);
	waitKey(1);

	cudaFree(deviceCharInput);
	cudaFree(deviceCharOutput);


	delete[] hostCharOutput;
	//free(hostCharInput);
	cudaProfilerStop();

	return true;
}
bool mainApplication::run()
{
	//defining the return status to indicate error state
	int returnStatus = 0;
	char selectionChoice;
	std::string imagePath;
	char loopCount = 0;
	cudaError_t cudaStatus;
	int numDevices;

	//this section checks for the CUDA capable devices
	if (cudaGetDeviceCount(&numDevices))
	{
		cout << "CUDA toolkit might not have been configured correctly" << endl;
		return -1;
	}

	if (numDevices > 0)
	{
		cout << numDevices << " CUDA capable devices found. Using first device" << endl;
		int tempDevice = 0;
		do
		{
			struct cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, tempDevice);
			if (properties.major != 9999)
			{
				totalAvailableThreads = properties.multiProcessorCount*properties.maxThreadsPerMultiProcessor;
				cout << "The maximum number of threads available is " << totalAvailableThreads << endl;
				cudaStatus = cudaSetDevice(tempDevice);
				if (cudaStatus != cudaSuccess)
				{
					cout << "Error in setting this device. Moving on to next one" << endl;
					tempDevice++;
				}
				else
				{
					tempDevice = 0;
				}
			}
			else
			{
				tempDevice++;
			}
		} while (tempDevice>0 && tempDevice<numDevices);


	}
	else
	{
		cout << "No GPU found!!! Please check and try again later" << endl;
		return -1;
	}

	//the do-while loops makes the program continuous
	do
	{
		//this condition is evaluated in successive loops, not in the first loop itself
		if (loopCount)
		{
			//ask if the user wants to load next image or work on the same image
			cout << "Do you want to edit-on the edited image?(Y/N)" << endl;
			std::cin >> selectionChoice;

			//user opted to not edit the old image, ask to save
			if ((selectionChoice == 'n' || selectionChoice == 'N') && outputImageMat.data)
			{
				char selectionTemp;
				bool successWrite = false;
				cout << "Do you want to save the edited image?(Y/N)" << endl;
				do
				{
					std::cin >> selectionTemp;
					if (selectionTemp == 'y' || selectionTemp == 'Y')
					{
						string outImageName;
						cout << "Enter the name of the image to save as?" << endl;
						std::cin >> outImageName;
						if (imwrite(outImageName, outputImageMat))
						{
							successWrite = true;
						}
						else
						{
							cout << "Save Unsuccessful" << endl;
							cout << "Do you want to continue saving the edited image?(Y/N)" << endl;
						}
					}
					else
						break;
				} while (!successWrite);
			}
		}
		do
		{

			if (loopCount == 0 || (loopCount && selectionChoice != 'Y' && selectionChoice != 'y'))
			{
				//prompt for user input
				cout << "Enter the name of the path of the image" << endl;

				//taking the path as input
				std::cin >> imagePath;

				returnStatus = loadImage(imagePath);
				if (returnStatus == 0)
				{
					cout << "The image will be displayed in a next window named \"Input Image\"" << endl;
					displayInputImage();
					break;
				}
				else
				{
					cout << "Error in loading the image." << endl;
					cout << "Do you want to try loading image again?(Y/N)" << endl;
					std::cin >> selectionChoice;
					if (selectionChoice != 'Y' && selectionChoice != 'y')
						return -1;
				}
			}
			else if (loopCount && (selectionChoice == 'Y' || selectionChoice == 'y'))
			{
				inputImageMat = outputImageMat;
				break;
			}
			else
			{
				cout << "You should not be here" << endl;
				return -1;
			}
		} while (returnStatus != 0 && (selectionChoice == 'Y' || selectionChoice == 'y'));

		if (choiceAndAct())
		{
			cout << "Operation successful" << endl;
			//display image will be called from the operations respectively
		}
		else
		{
			cout << "Operation Failed" << endl;
		}

		cout << "Do you wish to continue editing images? (Y/N)" << endl;
		std::cin >> selectionChoice;

		if (selectionChoice != 'Y' && selectionChoice != 'y')
			break;

		loopCount = (!loopCount) ? loopCount + 1 : loopCount;
	} while (selectionChoice == 'y' || selectionChoice == 'Y');

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error in freeing device. Profiling info may not reflect real data" << endl;
		return -1;
	}

	return returnStatus;
}