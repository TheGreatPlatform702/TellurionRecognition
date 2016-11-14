#pragma once
#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

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

class Kmeans {
private:
	int k;
	Mat vocabularys;
	Kmeans(int _k, Mat _data) {
		k = _k;
		BOWKMeansTrainer bowtrainer(k); // k clusters
		bowtrainer.add(_data);
		fprintf(stderr, "clustering ...\n");
		vocabularys = bowtrainer.cluster();
		fprintf(stderr, "clustering completed.\n");
	}

public:
	static Kmeans build(int k, Mat data) {
		return Kmeans(k, data);
	}
	
	Mat getVocabularys() {
		return vocabularys;
	}
};

#endif