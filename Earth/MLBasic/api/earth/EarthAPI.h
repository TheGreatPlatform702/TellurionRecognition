#pragma once

#ifndef EARTH_API_H
#define EARTH_API_H

#include "ModelBuilder.h"
#include "../../util/Util.h"
#include "../../lib/recognition/libsvm/libsvm.h"

Mat constructImageMat(unsigned char image[], const int &origin_height, const int &origin_width, const int &height, const int &width) {
	Mat img = Mat(1, origin_height * origin_width, CV_8U, image);
	Mat new_img = img.reshape(0, origin_height);
	Mat smooth_img = ImageUtil::ImageSmooth(new_img, 3, 3);
	Mat resized_mat = ImageUtil::ImageReSize(smooth_img, height, width, false);
	return resized_mat;
}

Mat getHist(const Mat &img, const string &dict_path, const int &minHessian,const int &width, const int &height) {
	Mat dictionary = Util::readMat(dict_path);
	DenseSIFT sift = DenseSIFT::build(minHessian, width, height, false);
	ConstructHist hist = ConstructHist::build(sift, dictionary);
	hist.computeImageDescriptor(img);
	return hist.getHistDescriptor();
}

svm_node* convert2SvmNode(const Mat &x) {
	int feature_len = x.cols;
	svm_node *sample = new svm_node[feature_len + 1];
	for (int i = 0; i < feature_len; i++) {
		sample[i].index = i + 1;
		sample[i].value = x.rowRange(0, 1).data[i];
	}
	sample[feature_len].index = -1;
	return sample;
}

int classify(const string &model_name, const Mat &x, double &prob) {
	svm_model *model = svm_load_model(model_name.c_str());
	svm_node *sample = convert2SvmNode(x);
	double *probability = new double[model->nr_class];
	int pred = svm_predict_probability(model, sample, probability);
	prob = Util::FindMax(probability, model->nr_class);

	delete model;
	delete[] sample;

	return pred;
}

bool isTellurion(const string &model_name, const Mat &x) {
	svm_model *model = svm_load_model(model_name.c_str());
	svm_node *sample = convert2SvmNode(x);
	int pred = svm_predict(model, sample);

	delete model;
	delete[] sample;

	if (pred == 1) return true;
	else return false;
}

int predict(unsigned char image[], const int &origin_height, const int &origin_width,
			const int &minHessian, const int &height, const int &width, 
			const string &multi_class_model_name, const string &one_class_model_name, const string &dict_path, const double &threashold, double &prob) {
	Mat img = constructImageMat(image, origin_height, origin_width, height, width);
	Mat feature = getHist(img, dict_path, minHessian, width, height);
	int res = -1;
	prob = 0;
	if(isTellurion(one_class_model_name, feature)){
		res = classify(multi_class_model_name, feature, prob);
		if (prob < threashold) res = -1;
	}
	return res;
}


#endif // !EARTH_API_H


