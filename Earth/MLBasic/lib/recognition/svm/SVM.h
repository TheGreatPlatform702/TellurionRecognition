#ifndef SVM_H
#define SVM_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class SVMModel {
public:
	static bool trainSVM(const Mat &trainDataMat, const Mat &trainDataMatLabel,
		const CvSVMParams &params, const string &modelName) {
		/*����ע�ͣ�
		trainDataMat��ѵ�����ݵ�ֱ��ͼ����
		trainDataMatLabel:ѵ�����ݵı�ǩ
		params:ģ�͵Ĳ���
		modelName:�洢��ģ����
		*/
		bool flag = false;
		fstream model_file;
		model_file.open(modelName, ios::in);
		try {
			CvSVM svm;
			cout << "SVM��������ʼѵ��" << endl;
			svm.train(trainDataMat, trainDataMatLabel, Mat(), Mat(), params);
			svm.save(modelName.c_str());
			cout << "SVM������ѵ�����" << endl;
			flag = true;
			return flag;
		}
		catch (exception) {
			return flag;
		}
	}

	static int testSVM(const Mat &testDataMat, const string &modelName) {
		/*����ע�ͣ�
		testDataMat���������ݵ�ֱ��ͼ����
		modelName:
		����Ԥ�����ı�ǩ
		*/
		CvSVM svm;
		try {
			svm.load(modelName.c_str());
		}
		catch (exception) {
			cout << "ģ�ͼ��ش���" << endl;
		}
		return (int)svm.predict(testDataMat);
	}

	static vector<int> testAllSampleSvm(const Mat &allSampleDataMat, const String &modelName)
	{
		/*����ע�ͣ�
		allSampleDataMat�� ���в������ݵ�ֱ��ͼ����
		modelName:ģ�͵�����
		����Ԥ�����ı�ǩ����
		*/
		vector<int> predictLabels;
		CvSVM svm;
		try {
			svm.load(modelName.c_str());
			for (int i = 0; i < allSampleDataMat.rows; i++) {
				double response = svm.predict(allSampleDataMat.rowRange(i, i + 1));
				predictLabels.push_back((int)response);
			}
		}
		catch (exception) {
			cout << "ģ�ͼ��ش���" << endl;
		}
		return predictLabels;
	}
};

#endif // !SVM_H

