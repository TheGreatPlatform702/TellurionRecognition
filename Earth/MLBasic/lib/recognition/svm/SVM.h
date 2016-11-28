#ifndef SVM_H
#define SVM_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class SVMModel {
public:
	static bool trainSVM(const Mat &trainDataMat, const Mat &trainDataMatLabel,
		const CvSVMParams &params, const string &modelName) {
		bool flag = false;
		fstream model_file;
		model_file.open(modelName, ios::in);
		try {
			CvSVM svm;
			cout << "SVM training" << endl;
			svm.train(trainDataMat, trainDataMatLabel, Mat(), Mat(), params);
			svm.save(modelName.c_str());
			cout << "SVM training complete" << endl;
			flag = true;
			return flag;
		}
		catch (exception) {
			return flag;
		}
	}

	static int testSVM(const Mat &testDataMat, const string &modelName) {
		CvSVM svm;
		try {
			svm.load(modelName.c_str());
		}
		catch (exception) {
			cout << "loading model error" << endl;
		}
		return (int)svm.predict(testDataMat);
	}

	static vector<int> testAllSampleSvm(const Mat &allSampleDataMat, const String &modelName)
	{
		vector<int> predictLabels;
		CvSVM svm;
		
		try {
			svm.load(modelName.c_str());
			for (int i = 0; i < allSampleDataMat.rows; i++) {
				double response = svm.predict(allSampleDataMat.rowRange(i, i + 1));
				//cout << response << endl;
				predictLabels.push_back((int)response);
			}
		}
		catch (exception) {
			cout << "model loading error" << endl;
		}
		return predictLabels;
	}
};

#endif // !SVM_H

