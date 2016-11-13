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
		/*参数注释：
		trainDataMat：训练数据的直方图矩阵
		trainDataMatLabel:训练数据的标签
		params:模型的参数
		modelName:存储的模型名
		*/
		bool flag = false;
		fstream model_file;
		model_file.open(modelName, ios::in);
		if (model_file)
		{
			cout << "模型已经训练完毕" << endl;
			flag = true;
			return flag;
		}
		else
		{
			try {
				CvSVM svm;
				cout << "SVM分类器开始训练" << endl;
				svm.train(trainDataMat, trainDataMatLabel, Mat(), Mat(), params);
				svm.save(modelName.c_str());
				cout << "SVM分类器训练完毕" << endl;
				flag = true;
				return flag;
			}
			catch (exception) {
				return flag;
			}
		}
	}

	static int testSVM(const Mat &testDataMat, const string &modelName) {
		/*参数注释：
		testDataMat：测试数据的直方图矩阵
		modelName:
		返回预测结果的标签
		*/
		CvSVM svm;
		try {
			svm.load(modelName.c_str());
		}
		catch (exception) {
			cout << "模型加载错误" << endl;
		}
		return (int)svm.predict(testDataMat);

	}
};

#endif // !SVM_H

