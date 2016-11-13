#include <iostream>
 
#include "../util/Util.h"
#include "../../MLBasic/lib/image/feature/bow/Vocabulary.h"
#include "../../MLBasic/lib/image/feature/bow/ConstructHist.h"
#include "../../MLBasic/lib/image/feature/bow/DenseSIFT.h"

using namespace std;
using namespace cv;

class TestHist{
public:
	static int test() {
		string image_file_dir = Util::getRealPath() + "\\data\\images";
		string csv_file = Util::getRealPath() + "\\data\\labels\\TestUtil.csv";
		string test_image_file=Util::getRealPath() + "\\data\\images\\IMG_20161103_181500.jpg";
		Util::getDataAndLabels(image_file_dir, csv_file);
		vector<vector<string>> data = Util::ImportDataFromCSV(csv_file);
		vector<string> _ImageListFile;
		for (int i = 0; i < data.size(); i++)
			_ImageListFile.push_back(data[i][0]);
		Vocabulary vocab = Vocabulary::build(_ImageListFile, 10);
		Mat aaa = vocab.getVocabulary();
		DenseSIFT sift = DenseSIFT::build();
		
		Mat image = ImageUtil::load(test_image_file);
		if(image.empty())
		{
			fprintf(stderr, "ImageData.load is error\n");
			return -1;
		}

		ConstructHist hist=ConstructHist::build(sift,aaa);
		hist.computeImageDescriptor(image);
		cv::Mat res=hist.getHistDescriptor();
		
		cout<<"cols:"<<res.cols<<", rows:"<<res.rows<<endl;

		return 0;
	}
};