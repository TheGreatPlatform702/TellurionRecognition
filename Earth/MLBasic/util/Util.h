#pragma once
#ifndef UTIL_H
#define UTIL_H

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
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/photo/photo.hpp"
#include <cv.h>

using namespace std;
using namespace cv;

#ifdef _DEBUG
#pragma comment(lib, "opencv_core2413d")
#pragma comment(lib, "opencv_highgui2413d")
#pragma comment(lib, "opencv_features2d2413d")
#pragma comment(lib, "opencv_ml2413d")
#pragma comment(lib, "opencv_nonfree2413d")
#pragma comment(lib, "opencv_imgproc2413d")
#else
#pragma comment(lib, "opencv_core2413")
#pragma comment(lib, "opencv_highgui2413")
#pragma comment(lib, "opencv_features2d2413")
#pragma comment(lib, "opencv_ml2413")
#pragma comment(lib, "opencv_nonfree2413")
#pragma comment(lib, "opencv_imgproc2413")
#endif

#include <direct.h>  
#include <stdio.h> 
#include <io.h>
#include <iostream>
#include <string>
#include <fstream>
#include <string>

class ImageUtil {
public:
	static cv::Mat ImageReSize(const cv::Mat& image, int weight = 255, int height = 255, bool scale = false)
	{
		cv::Mat unified_image;
		cv::Size size;
		if (scale) {
			size = cv::Size(weight, height);
		} else {
			int s = cv::min(image.rows, image.cols);
			float scale = 1.0*weight / s;
			size = cv::Size(image.cols  * scale, image.rows * scale);
		}
		cv::resize(image, unified_image, size);
		cv::medianBlur(unified_image, unified_image, 3);

		/*cout << "-----------------------------------" << endl;
		cout << weight << " " << height << " " << scale << endl;
		cout << image.rows << " " << image.cols << endl;
		cout << unified_image.rows << " " << unified_image.cols << endl;
		cout << "-----------------------------------" << endl;*/
		//unified_image = removeCharacters(unified_image);
		return unified_image;
	}

	static Mat ImageCut(const Mat &origin) {
		Mat img = ImageUtil::ImageReSize(origin, 400, 400, false);
		int padding = 120;
		int row_start = (int)((float)img.rows / 2 - padding);
		int row_end = (int)((float)img.rows / 2  + padding);
		int col_start = (int)((float)img.cols / 2 - padding);
		int col_end = (int)((float)img.cols / 2 + padding);
		Range row_range(row_start, row_end);
		Range col_range(col_start, col_end);
		return Mat(img, row_range, col_range);
	}

	static Mat ImageSmooth(const Mat &src, const int height, const int width) {
		Mat smooth_image;
		GaussianBlur(src, smooth_image, Size(height, width), 0, 0);
		return smooth_image;
	}
	static cv::Mat load(std::string image_file_path) {
		cv::Mat image;
		if (image_file_path.find(".jpg")) 
			image = cv::imread(image_file_path);
		Mat smooth_image = ImageUtil::ImageSmooth(image, 3, 3);
		return smooth_image;
	}
	static Mat removeCharacters(const Mat &src_img) {
		Mat imgCompont = ImageUtil::GetBlackComponet(src_img);
		return ImageUtil::Inpainting(src_img, imgCompont);
	}
private:
	static Mat GetBlackComponet(const Mat &srcImg)
	{
		Mat dstImg = srcImg.clone();
		Mat_<Vec3b>::iterator it = dstImg.begin<Vec3b>();
		Mat_<Vec3b>::iterator itend = dstImg.end<Vec3b>();
		int R = 90, G = 56, B = 44, EPS = 80;
		for (; it != itend; it++)
		{
			if (abs((*it)[0] - B) <= EPS && abs((*it)[1] - G) <= EPS && abs((*it)[2] - R) <= EPS) {

			}
			else
			{
				(*it)[0] = 0;
				(*it)[1] = 0;
				(*it)[2] = 0;
			}
		}
		return dstImg;
	}

	static Mat Inpainting(const Mat &oriImg, const Mat &maskImg)
	{
		Mat grayMaskImg;
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
		dilate(maskImg, maskImg, element);//膨胀后结果作为修复掩膜
										  //将彩色图转换为单通道灰度图，最后一个参数为通道数
		cvtColor(maskImg, grayMaskImg, CV_BGR2GRAY, 1);
		//修复图像的掩膜必须为8位单通道图像
		Mat inpaintedImage;
		inpaint(oriImg, grayMaskImg, inpaintedImage, 3, INPAINT_TELEA);
		return inpaintedImage;
	}
};

class Util {
public:
	static void ExportToCSVFile(const string& strFileName, const vector<vector<string>>& data)
	{
		if (data.empty()) return;
		ofstream csvIn(strFileName);
		for ( int row_num = 0; row_num < data.size(); ++ row_num) {
			string line = "";
			for ( int col_num = 0; col_num < data[row_num].size(); ++ col_num ) {
				if (col_num > 0) line.append(",");
				line.append(data[row_num][col_num]);
			}
			line.append("\n");
			if(csvIn.is_open()) 
			{
				csvIn << line;
			}
			else
			{
				cout<<"Error in opennig"<<strFileName<<endl;
			}				
		}
		csvIn.close();
	}

	static vector<vector<string>> ImportDataFromCSV(const string& strFileName)
	{
		vector<vector<string>> data;

		try
		{
			ifstream csvOut(strFileName);
			char rowContent[256];
			string strRow;
			while(!csvOut.eof())
			{
				csvOut.getline(rowContent,200);
				if (strlen(rowContent) == 0) continue;
				vector<string> vecRow=splitCSVRow(rowContent);
				if (vecRow.empty()) continue;
				data.push_back(vecRow);
			}
			csvOut.close();
		}
		catch (...)
		{
			return data;
		}

		return data;
	}


	static vector<string> splitCSVRow(const string& row)
	{
		vector<string> result;
		string oneCell="";
		for(int i=0;i<row.length();i++)
		{
			if(row[i]==',')
			{
				result.push_back(oneCell);
				oneCell="";					
			}
			else
			{
				oneCell+=row[i];
			}
		}
		result.push_back(oneCell);
		return result;
	}


	static std::string getRealPath() {
		#define MAX_PATH 100
		char buffer[MAX_PATH];   
		getcwd(buffer, MAX_PATH);
		std::string realPath(buffer);
		return realPath;
	}

	static void getFiles(string path, vector<string> &files, bool recursion = false){
		long hFile = 0;
		struct _finddata_t fileInfo;
		std::string pathName;

		if ((hFile = _findfirst(pathName.assign(path).append("\\*").c_str(), &fileInfo)) == -1) {
			return ;
		}
		do {
			string subPath = pathName.assign(path);
			if (subPath.back()!='\\' && subPath.back()!='/') {
				subPath.append("\\");
			}
			subPath = subPath + fileInfo.name;
			if (fileInfo.attrib&_A_SUBDIR) {
				string name(fileInfo.name);
				if (recursion && (name!=".") && (name!="..")) getFiles(subPath, files, recursion);
				continue;
			}
			files.push_back(subPath);
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
		return;
	}

	static void getDataAndLabels(const string &data_dir, const string &path_label_file) {
		long hFile = 0;
		struct _finddata_t fileInfo;
		std::string pathName;

		if ((hFile = _findfirst(pathName.assign(data_dir).append("\\*").c_str(), &fileInfo)) == -1) {
			return ;
		}

		vector<vector<string>> pathandLabel;
		int flag=1;

		do {
			string subPath = pathName.assign(data_dir);
			vector<string> files;
			if (subPath.back()!='\\' && subPath.back()!='/') {
				subPath.append("\\");
			}
			subPath = subPath + fileInfo.name;
			if (fileInfo.attrib&_A_SUBDIR) {
				string name(fileInfo.name);
				if ((name!=".") && (name!="..")) 
				{
					getFiles(subPath, files,false);
					if(files.size()>0)
					{
						for(int i=0;i<files.size();i++)
						{
							vector<string> temp;
							temp.push_back(files[i]);
							temp.push_back(to_string((long long) flag));
							pathandLabel.push_back(temp);
						}
						flag++;
					}
				}
			}
			
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
		
		ExportToCSVFile(path_label_file,pathandLabel);
	}

	static void writeMat(const Mat &mat, const string &fileName)
	{
		int n = mat.rows;
		int m = mat.cols;
		// Declare what you need
		cv::FileStorage file(fileName, cv::FileStorage::WRITE);
		// Write to file!
		try {
			file << "vocab" << mat;
		}
		catch (Exception e)
		{
			cout << e.what() << endl;
			getchar();
		}
	}
	static Mat readMat(const string &fileName) {

		// Declare what you need
		cv::FileStorage file(fileName, cv::FileStorage::READ);

		Mat mat;
		string mat_name;
		// Write to file!
		try {
			file["vocab"] >> mat;
		}
		catch (Exception e)
		{
			cout << e.what() << endl;
			getchar();
		}
		return mat;

	}
	static vector<string> getFileList(const string &csv_file) {
		vector< vector<string> > data = Util::ImportDataFromCSV(csv_file);
		vector<string> image_file_list;
		for (int i = 0; i < data.size(); i++) {
			image_file_list.push_back(data[i][0]);
		}
		return image_file_list;
	}
	static bool isFileExist(string filePath) {
		fstream handle;
		handle.open(filePath, ios::in);
		bool flag = false;
		if (handle) flag = true;
		handle.close();
		return flag;
	}
};

#endif