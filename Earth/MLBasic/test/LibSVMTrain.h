#pragma once
#pragma once

#ifndef LIB_SVM_TRAIN_H
#define LIB_SVM_TRAIN_H

#include "../api/earth/ModelBuilder.h"
#include "../api/earth/EarthAPI.h"
#include "../lib/recognition/libsvm/libsvm.h"
#include <time.h>

class LibModelTrain {
public:
	static void train() {
		cout << "***********************************" << endl;
		cout << "LibSvm Training..." << endl;
		cout << "***********************************" << endl;
		//string dataset_path = Util::getRealPath() + "\\data\\SmallTullurion";
		string dataset_path = Util::getRealPath() + "\\data\\iphone";

		string test_image_dir = dataset_path + "\\test_images";
		string image_file_dir = dataset_path + "\\images";
		string csv_file = dataset_path + "\\labels\\total.csv";
		string dictionary_file = dataset_path + "\\model\\dictionary.mat";
		string test_csv_file = dataset_path + "\\labels\\test.csv";
		string train_csv_file = dataset_path + "\\labels\\train.csv";
		string train_X_file = dataset_path + "\\model\\trainX.mat";
		string train_y_file = dataset_path + "\\model\\trainy.mat";
		string test_X_file = dataset_path + "\\model\\testX.mat";
		string test_y_file = dataset_path + "\\model\\testy.mat";

		string multi_class_model_file = dataset_path + "\\model\\multi_class.model";
		string one_class_model_file = dataset_path + "\\model\\one_class.model";

		const int CLASS_NUM = 8;
		int K = 300;
		int dict_height = 200, dict_width = 200;
		int height = 200, width = 200;
		const double threashold = 0.0;

		//Util::getDataAndLabels(image_file_dir, csv_file);
		//return;

		/*Util::getDataAndLabels(image_file_dir, train_csv_file);
		Util::getDataAndLabels(test_image_dir, test_csv_file);
		return;*/

		vector<string> train_file_list = Util::getFileList(train_csv_file);
		ModelBuilder model(dictionary_file, multi_class_model_file);

		if (Util::isFileExist(dictionary_file)) {
			cout << "Dictionary existed!" << endl;
		}
		else {
			cout << "Generating Dictionary.." << endl;
			model.generateDictionary(train_file_list, K, 400, dict_height, dict_width, false);
			cout << "Dictionary generated" << endl;
		}

		// train data
		Mat train_X, train_y;
		if (Util::isFileExist(train_X_file) && Util::isFileExist(train_y_file)) {
			cout << "Train Data existed! Loading.." << endl;
			train_X = Util::readMat(train_X_file);
			train_y = Util::readMat(train_y_file);
		}
		else {
			cout << "Generate Train data" << endl;
			ModelBuilder::loadFeatureAndLabel(train_csv_file, dictionary_file, 400, height, width, false, train_X, train_y);
			Util::writeMat(train_X, train_X_file);
			Util::writeMat(train_y, train_y_file);
		}
		// test data
		Mat test_X, test_y;
		if (Util::isFileExist(test_X_file) && Util::isFileExist(test_y_file)) {
			cout << "Test Data existed! Loading.." << endl;
			test_X = Util::readMat(test_X_file);
			test_y = Util::readMat(test_y_file);
		}
		else {
			cout << "Generate Test data" << endl;
			ModelBuilder::loadFeatureAndLabel(test_csv_file, dictionary_file, 400, height, width, false, test_X, test_y);
			Util::writeMat(test_X, test_X_file);
			Util::writeMat(test_y, test_y_file);
		}

		// training one-class model
		struct svm_model *one_class_model = NULL;
		if (Util::isFileExist(one_class_model_file)) {
			cout << "One-class model existed! Loading.." << endl;
			one_class_model = svm_load_model(one_class_model_file.c_str());
		}
		else {
			cout << "Training one-class model..." << endl;
			struct svm_problem one_class_prob = setSVMProblem(train_X, train_y);
			struct svm_parameter one_class_param = setOneClassSVMParams();
			one_class_model = svm_train(&one_class_prob, &one_class_param);
			svm_save_model(one_class_model_file.c_str(), one_class_model);
		}
		
		// training multi-class model
		struct svm_model *multi_class_model = NULL;
		if (Util::isFileExist(multi_class_model_file)) {
			cout << "Muti-class model existed! Loading.." << endl;
			multi_class_model = svm_load_model(multi_class_model_file.c_str());
		}
		else {
			cout << "Training multi-class model..." << endl;
			struct svm_problem multi_class_prob = setSVMProblem(train_X, train_y);
			struct svm_parameter multi_class_param = setMultiClassSVMParams();
			multi_class_model = svm_train(&multi_class_prob, &multi_class_param);
			svm_save_model(multi_class_model_file.c_str(), multi_class_model);
		}

		// testing one-class model
		vector<int> one_class_train_pred = libsvm_test(one_class_model, train_X);
		for (auto i : one_class_train_pred) {
			cout << i << " ";
		}
		cout << endl;
		double one_class_train_accuracy = computeAccuracy(one_class_train_pred);
		cout << "One-class train accuracy is: " << one_class_train_accuracy << endl;

		vector<int> one_class_test_pred = libsvm_test(one_class_model, test_X);
		for (auto i : one_class_test_pred) {
			cout << i << " ";
		}
		cout << endl;
		double one_class_test_accuracy = computeAccuracy(one_class_test_pred);
		cout << "One-class test accuracy is: " << one_class_test_accuracy << endl;

		// testing multi-class model
		vector<int> train_pred = libsvm_test(multi_class_model, train_X, vector<double>(), CLASS_NUM);
		double train_accuracy = computeAccuracy(train_pred, train_y);
		cout << "Multi-class train accuracy is: " << train_accuracy << endl;

		vector<double> probability;
		vector<int> test_pred = libsvm_test(multi_class_model, test_X, probability, CLASS_NUM);
		double test_accuracy = computeAccuracy(test_pred, test_y, probability, threashold);
		cout << "Multi-class test accuracy is: " << test_accuracy << endl;

		svm_free_model_content(multi_class_model);
		getchar();
	}

	static vector<int> libsvm_test(const svm_model *model, const Mat &X) {
		int test_num = X.rows, feature_len = X.cols;
		struct svm_node *sample = new svm_node[feature_len + 1];
		vector<int> predict_labels;
		for (int i = 0; i < test_num; i++) {
			for (int j = 0; j < feature_len; j++) {
				sample[j].index = j + 1;
				sample[j].value = X.rowRange(i, i + 1).data[j];
			}
			sample[feature_len].index = -1;
			int pred = svm_predict(model, sample);
			predict_labels.push_back(pred);
		}
		delete[] sample;
		return predict_labels;
	}

	static vector<int> libsvm_test(const svm_model *model, const Mat &X, vector<double> &probability, const int CLASS_NUM) {
		int test_num = X.rows, feature_len = X.cols;
		struct svm_node *sample = new svm_node[feature_len + 1];
		double *prob = new double[CLASS_NUM];
		vector<int> predict_labels;
		for (int i = 0; i < test_num; i++) {
			for (int j = 0; j < feature_len; j++) {
				sample[j].index = j + 1;
				sample[j].value = X.rowRange(i, i + 1).data[j];
			}
			sample[feature_len].index = -1;
			int pred = svm_predict_probability(model, sample, prob);
			predict_labels.push_back(pred);
			double p = Util::FindMax(prob, CLASS_NUM);
			probability.push_back(p);
		}
		
		delete[] prob;
		delete[] sample;

		return predict_labels;
	}

	static double computeAccuracy(const vector<int> &pred, const Mat &real_y, const vector<double> &probability, const double threashold) {
		int right = 0, all = 0;
		for (int i = 0; i < pred.size(); i++) {
			if (probability[i] < threashold) continue;
			if (pred[i] == real_y.rowRange(i, i + 1).data[0]) right++;
			all += 1;
		}
		cout << pred.size() << " " << all << endl;
		return (double)right / all;
	}

	static double computeAccuracy(const vector<int> &pred, const Mat &real_y) {
		int right = 0;
		for (int i = 0; i < pred.size(); i++) {
			if (pred[i] == real_y.rowRange(i, i + 1).data[0]) right++;
		}
		return (double)right / pred.size();
	}

	static double computeAccuracy(const vector<int> &pred) {
		int right = 0;
		for (int i = 0; i < pred.size(); i++) {
			if (pred[i] == 1) right += 1;
		}
		return (double)right / pred.size();
	}

	static svm_problem setSVMProblem(const Mat &X, const Mat &y) {
		struct svm_problem prob;
		prob.l = X.rows;
		int feature_len = X.cols;
		prob.y = new double[prob.l];
		svm_node *x_space = new svm_node[(feature_len + 1) * prob.l];
		prob.x = new svm_node *[prob.l];

		for (int i = 0; i < prob.l; i++) {
			prob.y[i] = y.rowRange(i, i + 1).data[0];
			for (int j = 0; j < feature_len; j++) {
				x_space[(feature_len + 1) * i + j].index = j + 1;
				x_space[(feature_len + 1) * i + j].value = X.rowRange(i, i + 1).data[j];
			}
			x_space[(feature_len + 1) * i + feature_len].index = -1;
			prob.x[i] = &x_space[(feature_len + 1) * i];
		}
		return prob;
	}

	static svm_parameter setMultiClassSVMParams() {
		struct svm_parameter param;
		param.svm_type = C_SVC;
		param.kernel_type = LINEAR;
		param.degree = 1;
		param.gamma = 0.5;
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 10;
		param.eps = 1e-4;
		param.p = 0.1;
		param.shrinking = 1;
		//param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;
		return param;
	}

	static svm_parameter setOneClassSVMParams() {
		struct svm_parameter param;
		param.svm_type = ONE_CLASS;
		param.kernel_type = LINEAR;
		param.degree = 2;
		param.gamma = 100;
		param.coef0 = 0;
		param.nu = 0.1;
		param.cache_size = 100;
		param.C = 10;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		//param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;
		return param;
	}
};

#endif // !LIB_SVM_TRAIN_H

