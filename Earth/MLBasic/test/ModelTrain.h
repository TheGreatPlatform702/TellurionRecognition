#pragma once

#ifndef TEST_API_H
#define TEST_API_H

#include "../api/earth/ModelBuilder.h"
#include "../api/earth/EarthAPI.h"
#include <time.h>

class ModelTrain {
public:
	static void test() {
		string root_path = Util::getRealPath() + "\\data";
		string dataset_path = root_path + "\\1115";
		string dictionary_file = dataset_path + "\\model\\dictionary.mat";
		string model_file = dataset_path + "\\model\\svm_model.xml";
		string label_file = dataset_path + "\\labels\\country2label.txt";
		map<int, string> countryMap;
		ifstream f(label_file);
		string country_name; int label;
		while (f >> label >> country_name) {
			countryMap[label] = country_name;
		}
		for (int i = 1; i <= 22; i++) {
			String img_path = root_path;
			img_path += "\\test\\";
			stringstream name;
			name << i;
			img_path += name.str();
			img_path += ".jpg";
			Mat img = ImageUtil::load(img_path);
			img = ImageUtil::ImageSmooth(img, 3, 3);
			imshow("origin", img);
			Mat resized_mat = ImageUtil::ImageReSize(img, 100, 100, false);
			//imshow("resize", resized_mat);
			waitKey(0);
			int res = svmTest(resized_mat, dictionary_file, model_file, 400, 100, 100);
			cout << i << " " << countryMap[res] << endl;
		}
		getchar();
	}
	static void train() {
		string dataset_path = Util::getRealPath() + "\\data\\iphone";
		// string dataset_path = Util::getRealPath() + "\\data\\total";
		
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
			model.generateDictionary(train_file_list, K, 400, 240, 240, false);
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
			ModelBuilder::loadFeatureAndLabel(train_csv_file, dictionary_file, 400, 240, 240, false, train_X, train_y);
			Util::writeMat(train_X, train_X_file);
			Util::writeMat(train_y, train_y_file);
		}

		CvSVMParams params;
		//params.svm_type = CvSVM::EPS_SVR;
		params.C = 10;
		params.kernel_type = CvSVM::LINEAR;
		params.kernel_type = CvSVM::RBF;
		params.kernel_type = CvSVM::POLY;
		params.degree = 3;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-4);

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
			ModelBuilder::loadFeatureAndLabel(test_csv_file, dictionary_file, 400, 240, 240, false, test_X, test_y);
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

