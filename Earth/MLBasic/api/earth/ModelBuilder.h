#ifndef MODEL_BUILDER_H
#define MODEL_BUILDER_H

#include <iostream>
#include "../../util/Util.h"
#include "../../lib/image/feature/bow/Vocabulary.h"
#include "../../lib/image/feature/bow/ConstructHist.h"
#include "../../lib/image/feature/bow/DenseSIFT.h"

#include "../../lib/recognition/svm/SVM.h"


class ModelBuilder {
public:
	ModelBuilder(){}
	ModelBuilder(const string &dict_name, const string &model_name): dictionary_name(dict_name), model_name(model_name){}
	void generateDictionary(const vector<string> &image_file_list, const int &k = 10,
		int minHessian = 400, int width = 255, int height = 255, bool scale = false) {
		setK(k);
		buildDictionary(image_file_list, minHessian, width, height, scale);
	}
	void train(const Mat &X, const Mat &y, const CvSVMParams &params){
		svmTrain(X, y, params);
	}
	vector<int> predict(const Mat &X) {
		return svmTest(X);
	}
	static void loadFeatureAndLabel(const string &csv_file, const string &dict_path, 
		const int &minHessian,const int &width, const int &height, bool scale, Mat &X, Mat &y) {
		vector< vector<string> > data = Util::ImportDataFromCSV(csv_file);
		Mat dictionary = Util::readMat(dict_path);
		DenseSIFT sift = DenseSIFT::build(minHessian, width, height, scale);
		ConstructHist hist = ConstructHist::build(sift, dictionary);
		for (int i = 0; i < data.size(); i++) {
			cout << data[i][0] << endl;
			Mat image = ImageUtil::load(data[i][0]);
			Mat new_image = ImageUtil::ImageReSize(image, width, height, scale);
			//Mat new_image = ImageUtil::ImageCut(image);
			hist.computeImageDescriptor(new_image);
			Mat x = hist.getHistDescriptor();
			cout << x.rows << " " << x.cols << endl;
			if (x.rows == 0 || x.cols == 0) continue;
			X.push_back(x);
			y.push_back(std::atoi(data[i][1].c_str()));
		}
	}
private:
	string dictionary_name;
	string model_name;
	int k;
	void setK(int k) {
		this->k = k;
	}
	void buildDictionary(const vector<string> &image_file_list,
		int minHessian = 400, int width = 255, int height = 255, bool scale = false) {
		Vocabulary vocab = Vocabulary::build(image_file_list, k, minHessian, width, height,scale);
		Mat dictionary = vocab.getVocabulary();
		Util::writeMat(dictionary, dictionary_name);
	}
	bool svmTrain(const Mat &X, const Mat &y, const CvSVMParams &params) {
		bool whether_succeed = SVMModel::trainSVM(X, y, params, model_name);
		return whether_succeed;
	}
	vector<int> svmTest(const Mat &X) {
		return SVMModel::testAllSampleSvm(X, model_name);
	}
};

#endif