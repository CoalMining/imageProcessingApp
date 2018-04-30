#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include "cuda_profiler_api.h"
#include <Windows.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>


//#define iceil(num,den) (num+den-1)/den

using namespace std;
using namespace cv;

class mainApplication
{
private:
	Mat inputImageMat;
	Mat outputImageMat;
	Mat tempImageMat;
	unsigned int totalAvailableThreads;
	bool isGray;
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;

public:
	mainApplication()
	{
		isGray = false;
	}
	~mainApplication(){}
	int loadImage(string);
	int displayInputImage();
	int displayOutputImage();
	bool choiceAndAct();
	bool meanBlurImage();
	bool gaussianBlur();
	bool sharpening();
	bool histogramEqualization();
	bool callConvolution2D();
	bool rgbToGray();
	bool run();
	bool embossing();
	bool laplaceEdgeDetection();
	bool laplaceEdgeDetectionWithSmoothing();
	bool prewittEdgeDetectionDx();
	bool prewittEdgeDetectionDy();
	bool prewittEdgeDetectionDm();
	bool sobelEdgeDetectionDx();
	bool sobelEdgeDetectionDy();
	bool sobelEdgeDetectionDm();
	bool powerLawHistogram();
	bool linearStretchingContrast();
	bool prewittEdgeDetectionDxSerial();
	bool prewittEdgeDetectionDySerial();
	bool prewittEdgeDetectionDmSerial();
	bool sobelEdgeDetectionDxSerial();
	bool sobelEdgeDetectionDySerial();
	bool sobelEdgeDetectionDmSerial();
	bool meanBlurImageSerial();
	bool gaussianBlurSerial();
	bool sharpeningSerial();
	bool histogramEqualizationSerial();
	bool callConvolution2DSerial();
	bool rgbToGraySerial();
	bool embossingSerial();
	bool laplaceEdgeDetectionSerial();
	bool laplaceEdgeDetectionWithSmoothingSerial();
	bool powerLawHistogramSerial();
	bool linearStretchingContrastSerial();
	bool sobelEdgeDetectionDmSerialWithSmoothing();
	bool prewittEdgeDetectionDmSerialWithSmoothing();
};

