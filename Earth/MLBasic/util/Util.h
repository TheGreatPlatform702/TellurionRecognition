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

		return unified_image;
	}
	/*static Mat ImageSmooth(const Mat &src, const int height, const int width) {
		IplImage img_src = src;
		IplImage *img_dst = cvCreateImage(cvGetSize(&img_src), IPL_DEPTH_8U, 3);
		cvSmooth(&img_src, img_dst, CV_MEDIAN, height, width);
		return img_dst;
	}*/
	static cv::Mat load(std::string image_file_path) {
		cv::Mat image;
		if (image_file_path.find(".jpg")) 
			image = cv::imread(image_file_path);
		Mat smooth_image;
		GaussianBlur(image, smooth_image, Size(3, 3), 0, 0);
		return image;
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