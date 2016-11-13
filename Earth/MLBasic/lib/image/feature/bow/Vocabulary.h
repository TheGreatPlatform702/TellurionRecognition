#pragma once
#ifndef VOCABULARY_H
#define VOCABULARY_H

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
#pragma comment(lib, "opencv_core2413d")
#pragma comment(lib, "opencv_highgui2413d")
#pragma comment(lib, "opencv_features2d2413d")
#pragma comment(lib, "opencv_ml2413d")
#pragma comment(lib, "opencv_nonfree2413d")
#pragma comment(lib, "opencv_imgproc2413d")
#else
#pragma comment(lib, "opencv_core2413")
#pragma comment(lib, "opencv_highgui2413")
#pragma comment(lib, "opencv_features2d2413")
#pragma comment(lib, "opencv_ml2413")
#pragma comment(lib, "opencv_nonfree2413")
#pragma comment(lib, "opencv_imgproc2413")
#endif

#include "../../../../../MLBasic/lib/cluster/kmeans.h"
#include "../../../../../MLBasic/lib/image/feature/bow/DenseSIFT.h"
#include "../../../../../MLBasic/util/Util.h"

class Vocabulary {
private:
	int k;
	DenseSIFT sift;
	Mat vocabulary;
	Vocabulary(vector<string> _ImageListFile, int _k, 
				int minHessian = 400, int width = 255, int height = 255, bool scale = false):
		sift(DenseSIFT::build(minHessian, width, height, scale)){
		k = _k;
		Mat training_descriptors;
		
		int count = 0;
		for (vector<string>::iterator iter = _ImageListFile.begin(); 
			iter != _ImageListFile.end(); ++ iter) {
			count += 1;
			string image_file_path = *iter;
			cout << count << endl;
			Mat image = ImageUtil::load(image_file_path);
			if (image.rows>0 && image.cols>0){
				sift.extractDescriptors(image);
				Mat descriptors = sift.getDescriptors();
				training_descriptors.push_back(descriptors);
			}
		}
		Kmeans K = Kmeans::build(k, training_descriptors);
		vocabulary = K.getVocabularys();
	}

public:
	Vocabulary(){}

	static Vocabulary build(vector<string> _ImageListFile, int k,
		int minHessian = 400, int width = 255, int height = 255, bool scale = false) {
		return Vocabulary(_ImageListFile, k, minHessian, width, height, scale);
	}

	Mat getVocabulary() {
		return vocabulary;
	}

	DenseSIFT getSIFT() {
		return sift;
	}
};

#endif