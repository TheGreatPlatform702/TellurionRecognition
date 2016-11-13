#pragma once
#ifndef TEST_DENSE_SIFT_H
#define TEST_DENSE_SIFT_H

#include <iostream>
 
#include "../../MLBasic/lib/image/feature/bow/DenseSIFT.h"

using namespace std;
using namespace cv;

class TestDenseSIFT{
public:
	static int test() {
		string image_file_path = Util::getRealPath() + "\\data\\images\\IMG_20161103_181419.jpg";
		Mat image = ImageUtil::load(image_file_path);
		if(image.empty())
		{
			fprintf(stderr, "ImageData.load is error\n");
			return -1;
		}
		DenseSIFT extract = DenseSIFT::build();
		extract.extractDescriptors(image);
		Mat descriptors = extract.getDescriptors();
		if(descriptors.empty()) {
			fprintf(stderr, "ImageData.load is error\n");
			return -1;
		}
		fprintf(stderr, "Descriptors size is (%d, %d).\n", descriptors.rows, descriptors.cols);
		return 0;
	}
};

#endif