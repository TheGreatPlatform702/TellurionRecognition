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
		string csv_file = Util::getRealPath() + "\\data\\labels\\TestUtil.csv";
		string model_file = Util::getRealPath() + "\\svm_model.xml";
		string dictionary_file = Util::getRealPath() + "\\dictionary.mat";
		
		Util::getDataAndLabels(image_file_dir, csv_file);
		vector<string> file_list = Util::getFileList(csv_file);

		ModelBuilder model(dictionary_file, model_file);
		model.generateDictionary(file_list, 10);

		Mat X, y;
		ModelBuilder::loadFeatureAndLabel(csv_file, dictionary_file, 100, 100, false, X, y);
		CvSVMParams params;
		params.svm_type = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, FLT_DIG);

		model.train(X, y, params);
		//test
		int height = 100, width = 100, minHessian = 400;
		int origin_height = 100, origin_width = 100;
		unsigned char image[10000];
		srand((unsigned)time(NULL));
		for (int i = 0; i < height * width; i++) {
			image[i] = rand() % 256;
		}
		int label = predict(image, origin_height, origin_width, minHessian, height, width, model_file, dictionary_file);
		cout << label << endl;
		cout << "success" << endl; 
		getchar();
	}
		
};

#endif // !TEST_API_H

