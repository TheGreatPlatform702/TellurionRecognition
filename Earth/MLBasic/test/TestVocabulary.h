#include <iostream>
 
#include "../util/Util.h"
#include "../../MLBasic/lib/image/feature/bow/Vocabulary.h"

using namespace std;
using namespace cv;

class TestVocabulary{
public:
	static int test() {
		string image_file_dir = Util::getRealPath() + "\\data\\images";
		string csv_file = Util::getRealPath() + "\\data\\labels\\TestUtil.csv";
		Util::getDataAndLabels(image_file_dir, csv_file);
		vector<vector<string>> data = Util::ImportDataFromCSV(csv_file);
		vector<string> _ImageListFile;
		for (int i = 0; i < data.size(); i++)
			_ImageListFile.push_back(data[i][0]);
		Vocabulary vocab = Vocabulary::build(_ImageListFile, 10);
		Mat aaa = vocab.getVocabulary();
		return 0;
	}
};