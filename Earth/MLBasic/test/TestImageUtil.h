#include <iostream>
 
#include "../util/Util.h"
using namespace std;
using namespace cv;

class TestImageUtil{
public:
	static int test() {
		string image_file_path = Util::getRealPath() + "\\data\\images\\IMG_20161103_181419.jpg";
		Mat image = ImageUtil::load(image_file_path);
		if(image.empty())
		{
			fprintf(stderr, "ImageData.load is error\n");
			return -1;
		}
		image = ImageUtil::ImageReSize(image);
		return 0;
	}
};