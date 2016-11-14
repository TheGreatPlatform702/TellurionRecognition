#pragma once
#ifndef DENSE_SIFT_H
#define DENSE_SIFT_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#ifdef _DEBUG
#pragma comment(lib, "opencv_core249d")
#pragma comment(lib, "opencv_highgui249d")
#pragma comment(lib, "opencv_features2d249d")
#pragma comment(lib, "opencv_ml249d")
#pragma comment(lib, "opencv_nonfree249d")
#pragma comment(lib, "opencv_imgproc249d")
#else
#pragma comment(lib, "opencv_core249")
#pragma comment(lib, "opencv_highgui249")
#pragma comment(lib, "opencv_features2d249")
#pragma comment(lib, "opencv_ml249")
#pragma comment(lib, "opencv_nonfree249")
#pragma comment(lib, "opencv_imgproc249")
#endif

#include "../../../../../MLBasic/util/Util.h"

class DenseSIFT{
private:
	Ptr<DenseFeatureDetector> feature_detector;
	//Ptr<SiftFeatureDetector> feature_detector;
	Ptr<DescriptorExtractor> descriptor_extractor;
	Ptr<DescriptorMatcher> descriptor_matcher;

	vector<cv::KeyPoint> keyPoints;
	cv::Mat descriptors;
	bool scale;
	int weight;
	int height;
	
	DenseSIFT(int _minHessian, int _weight, int _height, bool _scale){
		weight = _weight;
		height = _height;
		scale = _scale;
		//Ptr<FeatureDetector> feature_detector(new SurfFeatureDetector(minHessian));
		feature_detector = new DenseFeatureDetector();
		//feature_detector = new SiftFeatureDetector();
		//descriptor_extractor = DescriptorExtractor::create("SURF");
		//descriptor_extractor = new BriefDescriptorExtractor();
		descriptor_extractor = new SiftDescriptorExtractor();
		descriptor_matcher=new FlannBasedMatcher();

		//this.descriptor_extractor = new SurfDescriptorExtractor();
	}

public:
	
	DenseSIFT() {}

	static DenseSIFT build(int minHessian = 400, int weight = 256, int height = 256, bool scale = false) {
		DenseSIFT res = DenseSIFT(minHessian, weight, height, scale);
		return res;
	}

	void extractDescriptors(const cv::Mat& image) {
		cv::Mat new_image = ImageUtil::ImageReSize(image, weight, height, scale);
		keyPoints = vector<cv::KeyPoint>();
		descriptors = cv::Mat();
		feature_detector->detect(new_image, keyPoints);
        descriptor_extractor->compute(new_image, keyPoints, descriptors);
		return ;
	}
	
	vector<cv::KeyPoint> getKeyPoints() {
		return keyPoints;
	}

	cv::Mat getDescriptors() {
		return descriptors;
	}

	cv::Mat getSTDImage(const cv::Mat& image) {
		cv::Mat new_image = ImageUtil::ImageReSize(image, weight, height, scale);
		return new_image;
	}

	Ptr<DenseFeatureDetector> getPtrDenseFeatureDetector() {
		return feature_detector;
	}

	Ptr<DescriptorExtractor> getPtrDescriptorExtractor() {
		return descriptor_extractor;
	}

	Ptr<DescriptorMatcher> getPtrDescriptorMatcher() {
		return descriptor_matcher;
	}
};

#endif