#include "ModelPredict.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;
using namespace cv::dnn;

#ifdef WIN32
	#include <io.h>
	#include <windows.h>
	void readFileNameInDir(string strDir, 
						   vector<string>& vFileFullPath,
						   vector<string>& vFileName)
	{
		string file_dir = strDir;
		intptr_t handle;    			// file handle
		struct _finddata_t fileInfo;    // file struct
		handle = _findfirst(strDir.append("/*").c_str(), &fileInfo);    // get file handle first
		while (!_findnext(handle, &fileInfo)){
			string filename = fileInfo.name;
			if (filename != "." && filename != ".."){
				vFileName.push_back(filename);
				vFileFullPath.push_back(file_dir.append("/") + filename);
			}
		}
		_findclose(handle);
	}
#else
	#include <dirent.h>
	void readFileNameInDir(string strDir, 
						   vector<string>& vFileFullPath, 
						   vector<string>& vFileName)
	{
		struct dirent* pDirent;
		DIR* pDir = opendir(strDir.c_str());
		if (pDir == NULL)
			return;

		while ((pDirent = readdir(pDir)) != NULL)
		{
			string strFileName = pDirent->d_name;
			
			if (strFileName != "." && strFileName != ".."){
				string strFileFullPath = strDir + "/" + strFileName;
				vFileName.push_back(strFileName);
				vFileFullPath.push_back(strFileFullPath);
			}
		}
	}
#endif

int main(int argc, char* argv[])
{
	char* model_path = "../../models/mask_rcnn_sim.onnx";
	String img_dir = "../../models/imgs/";

	string save_dir = "../../models/inference";
	vector<string> vec_img_paths, vec_img_names;
	readFileNameInDir(img_dir, vec_img_paths, vec_img_names);

	// construct ModelPredict object and load model
	ModelPredict onnx_mp(true, 0);
	auto sta = onnx_mp.LoadModel(model_path);
	if (sta==false){
		std:cerr << "ERROR: Mode load fail.\n";
		return -1;
	}

	cout << "INFO: All inference images: " << vec_img_paths.size() << endl;
	for (size_t i = 0; i < vec_img_paths.size(); i++){
		cout << "INFO: inference at: " << std::to_string(i) << ", img name is: " << vec_img_names[i]<< endl;
		cv::Mat img = imread(vec_img_paths[i]);

		float score_thresh = 0.6f;
		bool status = onnx_mp.PredictAction(img, score_thresh);

		cv::Mat result_img = onnx_mp.ShowPredictMask(img, score_thresh);

		// // Get minimum bounding boxes
		// std::vector<std::vector<cv::Point2f>> min_bboxes;
		// min_bboxes = onnx_mp.GetMinBoundingBoxes();

		// // Get bounding box angels
		// std::vector<float> bbox_angles = onnx_mp.GetBoundingBoxAngles();
		// std::cout << "INFO: Bounding box inclination angles are: ";
		// for (const auto& element : bbox_angles)
        // 	std::cout << element << " ";
		// std::cout << std::endl;

		// Save images
		cv::String save_path = save_dir + "/" + vec_img_names[i];
		imwrite(save_path, result_img);

		// // Images display
		cv::String win_name = "Inference result of image-[" + std::to_string(i) + "]";
		cv::namedWindow(win_name, cv::WINDOW_NORMAL);
		int initial_width = 800, initial_height = 600;
		cv::resizeWindow(win_name, initial_width, initial_height);
		cv::imshow(win_name, result_img);
		cv::waitKey(0);
	}
	return 0;
}


