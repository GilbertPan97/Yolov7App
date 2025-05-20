#include "ModelPredict.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring> // for strdup
#include <cstdlib> // for free
#include <algorithm> // For std::sort
#include <numeric>   // For std::iota
#include <filesystem>  // C++17 and later

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
	// Function to read file names in the specified directory
	void readFileNameInDir(const string& strDir, 
		vector<string>& vFileFullPath,
		vector<string>& vFileName)
	{
		string file_dir = strDir;  // Store the directory path
		intptr_t handle;    			// File handle
		struct _finddata_t fileInfo;    // Structure to store file information

		// Create the search path by appending "\\*" to the directory
		string searchPath = file_dir + "/*";  // Use backslash for Windows paths

		// Start searching for files in the specified directory
		handle = _findfirst(searchPath.c_str(), &fileInfo);

		// If no files are found, return early
		if (handle == -1) {  
			return;
		}

		// Iterate through the directory files
		while (_findnext(handle, &fileInfo) == 0) {
			string filename = fileInfo.name;  // Get the file name
			if (filename != "." && filename != "..") {  // Skip "." and ".." directories
				vFileName.push_back(filename);  // Store file name
				vFileFullPath.push_back(file_dir + "/" + filename);  // Store full file path
			}
		}

		// Close the file handle after use
		_findclose(handle);
	}
#else
	#include <dirent.h>
	void readFileNameInDir(const string& strDir, 
						vector<string>& vFileFullPath, 
						vector<string>& vFileName)
	{
		struct dirent* pDirent;
		DIR* pDir = opendir(strDir.c_str());
		if (pDir == NULL)
			return;

		// Read directory entries
		while ((pDirent = readdir(pDir)) != NULL)
		{
			string strFileName = pDirent->d_name;
			
			if (strFileName != "." && strFileName != "..")
			{
				string strFileFullPath = strDir + "/" + strFileName;
				vFileName.push_back(strFileName);
				vFileFullPath.push_back(strFileFullPath);
			}
		}

		closedir(pDir);

		// Sort file names and their corresponding full paths
		vector<size_t> indices(vFileName.size());
		iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., n-1

		sort(indices.begin(), indices.end(),
			[&vFileName](size_t i1, size_t i2) {
				return vFileName[i1] < vFileName[i2];
			});

		vector<string> sortedFileName(vFileName.size());
		vector<string> sortedFileFullPath(vFileFullPath.size());

		for (size_t i = 0; i < indices.size(); ++i)
		{
			sortedFileName[i] = vFileName[indices[i]];
			sortedFileFullPath[i] = vFileFullPath[indices[i]];
		}

		vFileName = move(sortedFileName);
		vFileFullPath = move(sortedFileFullPath);
	}
#endif

std::vector<cv::String> load_categories(const cv::String& file_path) {
    std::vector<cv::String> categories;
    std::ifstream file(file_path.c_str()); // Convert cv::String to std::string for file stream

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return categories; // Return empty vector if file cannot be opened
    }

    std::string line;
    while (std::getline(file, line)) {
        // Convert each line to cv::String and add to the vector
        categories.push_back(cv::String(line));
    }

    file.close();
    return categories;
}

int main(int argc, char* argv[])
{
	char* model_path = "../models/yolov7-seg.onnx";
	String img_dir = "../imgs";
	String categories_path = "../models/labels_algae.txt";
	string save_dir = "../runs";
	vector<string> vec_img_paths, vec_img_names;
	readFileNameInDir(img_dir, vec_img_paths, vec_img_names);

	// construct ModelPredict object and load model
	ModelPredict onnx_mp(true, 0, 8);
	auto sta = onnx_mp.LoadModel(model_path, ModelPredict::TaskType::InstanceSeg);
	if (sta==false){
		std:cerr << "ERROR: Mode load fail.\n";
		return -1;
	}

	cout << "INFO: All inference images: " << vec_img_paths.size() << endl;
	for (size_t i = 1; i < vec_img_paths.size(); i++){
		cout << "INFO: inference at: " << std::to_string(i) << ", img name is: " << vec_img_names[i]<< endl;
		cv::Mat img = imread(vec_img_paths[i]);

		float score_thresh = 0.4f;
		onnx_mp.LabelCategories(load_categories(categories_path));
		bool status = onnx_mp.PredictAction(img, score_thresh);
		cv::Mat result_img = onnx_mp.RenderInference(img, 0.0f);

		// Save images
		cv::String save_path = save_dir + "/" + vec_img_names[i];
		imwrite(save_path, result_img);

		// Images display
		cv::String win_name = "Inference result";
		cv::namedWindow(win_name, cv::WINDOW_NORMAL);
		int initial_width = 800, initial_height = 600;
		cv::resizeWindow(win_name, initial_width, initial_height);
		cv::imshow(win_name, result_img);
		cv::waitKey(0);
	}
	return 0;
}


