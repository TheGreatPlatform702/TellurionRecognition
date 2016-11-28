#pragma once

#ifndef TEST_API_H
#define TEST_API_H

#include "../api/earth/ModelBuilder.h"
#include "../api/earth/EarthAPI.h"
#include "../lib/recognition/libsvm/libsvm.h"
#include <time.h>

class ModelTrain {
public:
	static void test() {
		cout << "***********************************" << endl;
		cout << "Testing..." << endl;
		cout << "***********************************" << endl;
		string root_path = Util::getRealPath() + "\\data";
		string dataset_path = root_path + "\\SmallTullurion";
		string image_file_dir = dataset_path + "\\images";
		string dictionary_file = dataset_path + "\\model\\dictionary.mat";
		string model_file = dataset_path + "\\model\\svm_model.xml";
		string label_file = dataset_path + "\\labels\\country2label.txt";
		string csv_file = dataset_path + "\\labels\\total.csv";
		string test_X_file = dataset_path + "\\model\\testX.mat";
		string test_y_file = dataset_path + "\\model\\testy.mat";

		//Util::getDataAndLabels(image_file_dir, csv_file);
		//return;
		// test
		ModelBuilder model(dictionary_file, model_file);
		Mat test_X, test_y;
		if (Util::isFileExist(test_X_file) && Util::isFileExist(test_y_file)) {
			cout << "Test Data existed! Loading.." << endl;
			test_X = Util::readMat(test_X_file);
			test_y = Util::readMat(test_y_file);
		}
		else {
			cout << "Generate Test data" << endl;
			ModelBuilder::loadFeatureAndLabel(csv_file, dictionary_file, 400, 180, 180, false, test_X, test_y);
			Util::writeMat(test_X, test_X_file);
			Util::writeMat(test_y, test_y_file);
		}
		vector<int> test_pred = model.predict(test_X);
		cout << "Test accuracy is: " << ModelTrain::computeAccuracy(test_pred, test_y) << endl;

		getchar();

	}
	static void train() {
		cout << "***********************************" << endl;
		cout << "Training..." << endl;
		cout << "***********************************" << endl;
		//string dataset_path = Util::getRealPath() + "\\data\\SmallTullurion";
		 string dataset_path = Util::getRealPath() + "\\data\\iphone";
		
		string test_image_dir = dataset_path + "\\test_images";
		string image_file_dir = dataset_path + "\\images";
		string csv_file = dataset_path + "\\labels\\total.csv";
		string model_file = dataset_path + "\\model\\svm_model.xml";
		string dictionary_file = dataset_path + "\\model\\dictionary.mat";
		string test_csv_file = dataset_path + "\\labels\\test.csv";
		string train_csv_file = dataset_path + "\\labels\\train.csv";
		string train_X_file = dataset_path + "\\model\\trainX.mat";
		string train_y_file = dataset_path + "\\model\\trainy.mat";
		string test_X_file = dataset_path + "\\model\\testX.mat";
		string test_y_file = dataset_path + "\\model\\testy.mat";
		string oneclass_model_file = dataset_path + "\\model\\oneclass_model.xml";

		int K = 300;

		//Util::getDataAndLabels(image_file_dir, csv_file);
		//return;

		/*Util::getDataAndLabels(image_file_dir, train_csv_file);
		Util::getDataAndLabels(test_image_dir, test_csv_file);
		return;*/

		vector<string> train_file_list = Util::getFileList(train_csv_file);
		ModelBuilder model(dictionary_file, model_file);

		if (Util::isFileExist(dictionary_file)) {
			cout << "Dictionary existed!" << endl;
		}
		else {
			cout << "Generating Dictionary.." << endl;
			model.generateDictionary(train_file_list, K, 400, 180, 180, false);
			cout << "Dictionary generated" << endl;
		}


		
		// train
		Mat train_X, train_y;
		if (Util::isFileExist(train_X_file) && Util::isFileExist(train_y_file)) {
			cout << "Train Data existed! Loading.." << endl;
			train_X = Util::readMat(train_X_file);
			train_y = Util::readMat(train_y_file);
		}
		else {
			cout << "Generate Train data" << endl;
			ModelBuilder::loadFeatureAndLabel(train_csv_file, dictionary_file, 400, 180, 180, false, train_X, train_y);
			Util::writeMat(train_X, train_X_file);
			Util::writeMat(train_y, train_y_file);
		}

		CvSVMParams params_multiclass;
		//params.svm_type = CvSVM::EPS_SVR;
		params_multiclass.C = 10;
		params_multiclass.kernel_type = CvSVM::LINEAR;
		//params.kernel_type = CvSVM::RBF;
		//params.kernel_type = CvSVM::POLY;
		params_multiclass.degree = 3;
		params_multiclass.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-4);

		

		if (Util::isFileExist(model_file)) {
			cout << "model existed!" << endl;
		}
		else {
			cout << "Training..." << endl;
			model.train(train_X, train_y, params_multiclass);
			cout << "Training successfully ended" << endl;
		}

		CvSVMParams params_oneclass;
		params_oneclass.C = 10;
		params_oneclass.kernel_type = CvSVM::LINEAR;
		params_oneclass.svm_type = CvSVM::ONE_CLASS;
		params_oneclass.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-4);
		params_oneclass.nu = 0.01;


		ModelBuilder oneclass_model(dictionary_file, oneclass_model_file);
		oneclass_model.train(train_X, params_oneclass);

		vector<int> one_class_result = oneclass_model.predict(train_X);
		int one_count = 0, zero_count = 0;
		for (auto c : one_class_result) {
			if (c == 1) one_count += 1;
			else zero_count += 1;
		}
		cout << one_count << " " << zero_count + one_count << " " << float(one_count) / (zero_count + one_count) << endl;

		// test
		Mat test_X, test_y;
		if (Util::isFileExist(test_X_file) && Util::isFileExist(test_y_file)) {
			cout << "Test Data existed! Loading.." << endl;
			test_X = Util::readMat(test_X_file);
			test_y = Util::readMat(test_y_file);
		}
		else {
			cout << "Generate Test data" << endl;
			ModelBuilder::loadFeatureAndLabel(test_csv_file, dictionary_file, 400, 180, 180, false, test_X, test_y);
			Util::writeMat(test_X, test_X_file);
			Util::writeMat(test_y, test_y_file);
		}
		
		vector<int> train_pred = model.predict(train_X);
		cout << "Train accuracy is: " << ModelTrain::computeAccuracy(train_pred, train_y) << endl;

		vector<int> test_pred = model.predict(test_X);
		cout << "Test accuracy is: " << ModelTrain::computeAccuracy(test_pred, test_y) << endl;

		getchar();
	}
	
	static double computeAccuracy(const vector<int> &pred, const Mat &real_y) {
		int right = 0, all = pred.size();
		for (int i = 0; i < all; i++) {
			if (pred[i] == real_y.rowRange(i, i + 1).data[0]) right++;
		}
		return (double)right / all;
	}

};

#endif // !TEST_API_H

