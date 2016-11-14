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
		string train_X_file = Util::getRealPath() + "\\data\\trainX.mat";
		string train_y_file = Util::getRealPath() + "\\data\\trainy.mat";
		string test_X_file = Util::getRealPath() + "\\data\\testX.mat";
		string test_y_file = Util::getRealPath() + "\\data\\testy.mat";

		int K = 10;

		// Util::getDataAndLabels(image_file_dir, csv_file);

		vector<string> train_file_list = Util::getFileList(train_csv_file);
		ModelBuilder model(dictionary_file, model_file);

		cout << "Generating Dictionary.." << endl;
		model.generateDictionary(train_file_list, K, 400, 100, 100, true);
		cout << "Dictionary generated" << endl;

		// train
		Mat train_X, train_y;
		ModelBuilder::loadFeatureAndLabel(train_csv_file, dictionary_file, 400, 100, 100, true, train_X, train_y);

		CvSVMParams params;
		params.svm_type = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, FLT_DIG);

		cout << "Training..." << endl;

		model.train(train_X, train_y, params);

		cout << "Training successfully ended" << endl;

		// test
		Mat test_X, test_y;
		ModelBuilder::loadFeatureAndLabel(test_csv_file, dictionary_file, 400, 100, 100, true, test_X, test_y);

		//Util::writeMat(test_X, test_X_file);
		//Util::writeMat(test_y, test_y_file);
		
		//Mat train_X, train_y;
		//ModelBuilder::loadFeatureAndLabel(train_csv_file, dictionary_file, 100, 100, true, train_X, train_y);

		//Util::writeMat(train_X, train_X_file);
		//Util::writeMat(train_y, train_y_file);

		cout << "Predict.." << endl;
		vector<int> pred = model.predict(test_X);

		int right = 0, all = test_X.rows;
		for (int i = 0; i < all; i++) {
			// int response = SVMModel::testSVM(test_X.rowRange(i, i + 1), model_file);
			
			if (pred[i] == test_y.data[i]) right++;
			// cout << pred[i] << " " << (int)test_y.data[i] << endl;
			// getchar();
		}

		cout << "Accuracy is: " << (float)right / all << endl;
		// vector<int> pred = model.predict(test_X);

		////test
		//int height = 100, width = 100, minHessian = 400;
		//int origin_height = 100, origin_width = 100;
		//unsigned char image[10000];
		//srand((unsigned)time(NULL));
		//for (int i = 0; i < height * width; i++) {
		//	image[i] = rand() % 256;
		//}
		//int label = predict(image, origin_height, origin_width, minHessian, height, width, model_file, dictionary_file);
		//cout << label << endl;
		//cout << "success" << endl; 
		getchar();
	}
		
};

#endif // !TEST_API_H

