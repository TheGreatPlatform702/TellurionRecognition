#include <iostream>
 
#include "../util/Util.h"
using namespace std;
using namespace cv;

class TestUtil{
public:
	static int test() {
		string image_file_dir = Util::getRealPath() + "\\data\\images";
		string csv_file = Util::getRealPath() + "\\data\\labels\\TestUtil.csv";
		Util::getDataAndLabels(image_file_dir, csv_file);
		vector<vector<string>> data = Util::ImportDataFromCSV(csv_file);
		for ( int i = 0; i < data.size(); i++){
			for(int j = 0; j < data[i].size(); j++) {
				cout << data[i][j] << ", ";
			}
			cout << endl;
		}
		return 0;
	}

	static int testWrite() {
		string image_file = Util::getRealPath() + "\\data\\images\\IMG_20161103_181500.jpg";
		Mat image = ImageUtil::load(image_file);
		Util::writeMat(image, Util::getRealPath() + "\\data\\test.ext");
		return 0;
	}

	static int testRead() {
		cout << "FFF" << endl;
		Mat mat = Util::readMat(Util::getRealPath() + "\\data\\test.ext");
		cout << "SS" << endl;
		cout << mat.rows << " " << mat.cols << endl;
		for (int i = 0; i < 10; ++i)
		{
			cout << int(mat.data[i]) << endl;
		}
		getchar();

		return 0;
	}


};