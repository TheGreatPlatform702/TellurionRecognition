#pragma once

#ifndef EARTH_API_H
#define EARTH_API_H

#include "ModelBuilder.h"
#include "../../util/Util.h"

Mat constructImageMat(unsigned char image[], const int &origin_height, const int &origin_width, const int &height, const int &width) {
	Mat img = Mat(1, origin_height * origin_width, CV_8U, image);
	Mat new_img = img.reshape(0, origin_height);
	Mat resized_mat = ImageUtil::ImageReSize(new_img, height, width, false);
	//resize(new_img, img_transformed, Size(height, width), 0, 0, CV_INTER_LINEAR);
	return new_img;
}

Mat getHist(const Mat &img, const string &dict_path, const int &minHessian,const int &width, const int &height) {
	Mat dictionary = Util::readMat(dict_path);
	DenseSIFT sift = DenseSIFT::build(minHessian, height, width, false);
	ConstructHist hist = ConstructHist::build(sift, dictionary);
	hist.computeImageDescriptor(img);
	return hist.getHistDescriptor();
}

int svmTest(const Mat &img, const string &dict_path, const string &model_name, const int &minHessian, const int &width, const int &height) {
	Mat feature = getHist(img, dict_path, minHessian, width, height);
	return SVMModel::testSVM(feature, model_name);
}

int predict(unsigned char image[], const int &origin_height, const int &origin_width,
			const int &minHessian, const int &height, const int &width, const string &model_name, const string &dict_path) {
	Mat img = constructImageMat(image, origin_height, origin_width, height, width);
	int res = svmTest(img, dict_path, model_name, minHessian, width, height);
	return res;
}

#endif // !EARTH_API_H


