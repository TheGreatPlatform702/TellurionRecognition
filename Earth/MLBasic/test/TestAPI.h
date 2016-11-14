#pragma once

#ifndef TEST_API_H
#define TEST_API_H

#include "../api/earth/ModelBuilder.h"
#include "../api/earth/EarthAPI.h"
#include <time.h>

class TestAPI {
public:
	static void test() {
		string image_file_dir = Util::getRealPath() + "\\data\\images";
		string csv_file = Util::getRealPath() + "\\data\\labels\\total.csv";
		string model_file = Util::getRealPath() + "\\data\\model\\svm_model.xml";
		string dictionary_file = Util::getRealPath() + "\\data\\model\\dictionary.mat";
		string test_csv_file = Util::getRealPath() + "\\data\\labels\\test.csv";
		string train_csv_file = Util::getRealPath() + "\\data\\labels\\train.csv";
		string train_X_file = Util::getRealPath() + "\\data\\model\\trainX.mat";
		string train_y_file = Util::getRealPath() + "\\data\\model\\trainy.mat";
		string test_X_file = Util::getRealPath() + "\\data\\model\\testX.mat";
		string test_y_file = Util::getRealPath() + "\\data\\model\\testy.mat";

		int K = 10;

		// Util::getDataAndLabels(image_file_dir, csv_file);

		vector<string> train_file_list = Util::getFileList(train_csv_file);
		ModelBuilder model(dictionary_file, model_file);

		if (Util::isFileExist(dictionary_file)) {
			cout << "Dictionary existed!" << endl;
		}
		else {
			cout << "Generating Dictionary.." << endl;
			model.generateDictionary(train_file_list, K, 400, 100, 100, true);
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
			ModelBuilder::loadFeatureAndLabel(train_csv_file, dictionary_file, 400, 100, 100, true, train_X, train_y);
			Util::writeMat(train_X, train_X_file);
			Util::writeMat(train_y, train_y_file);
		}

		CvSVMParams params;
		params.svm_type = CvSVM::C_SVC;
		//params.kernel_type = CvSVM::LINEAR;
		params.kernel_type = CvSVM::RBF;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3);

		if (Util::isFileExist(model_file)) {
			cout << "model existed!" << endl;
		}
		else {
			cout << "Training..." << endl;
			model.train(train_X, train_y, params);
			cout << "Training successfully ended" << endl;
		}

		// test
		Mat test_X, test_y;
		if (Util::isFileExist(test_X_file) && Util::isFileExist(test_y_file)) {
			cout << "Test Data existed! Loading.." << endl;
			test_X = Util::readMat(test_X_file);
			test_y = Util::readMat(test_y_file);
		}
		else {
			cout << "Generate Test data" << endl;
			ModelBuilder::loadFeatureAndLabel(test_csv_file, dictionary_file, 400, 100, 100, true, test_X, test_y);
			Util::writeMat(test_X, test_X_file);
			Util::writeMat(test_y, test_y_file);
		}
		
		vector<int> train_pred = model.predict(train_X);
		cout << "Train accuracy is: " << TestAPI::computeAccuracy(train_pred, train_y) << endl;

		vector<int> test_pred = model.predict(test_X);
		cout << "Test accuracy is: " << TestAPI::computeAccuracy(test_pred, test_y) << endl;

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

