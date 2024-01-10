#ifndef MODELPREDICT_H
#define MODELPREDICT_H

#include "ModelPredictGlobal.h"

#include <stdio.h>

#include <string>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#define _CRT_SECURE_NO_WARNINGS 

class MP_EXPORT ModelPredict {
private:
	// Ort handle
	Ort::Env env_;
	Ort::SessionOptions session_ops_;
	Ort::Session* session_;

	// model info
	std::vector<char*> input_names_;
	std::vector<char*> output_names_;
	std::vector<Ort::AllocatedStringPtr> input_names_ptr_;
	std::vector<Ort::AllocatedStringPtr> output_names_ptr_;		// avoid names pointer to be released.

	// inference results
	std::vector<std::array<float, 4>> bboxes_;		// element: [x_min, y_min, x_max, y_max]
	std::vector<std::array<float, 8>> minbboxes_;	// element: [x0, y0, x1, y1, x2, y2, x3, y3]
	std::vector<int> labels_;
	std::vector<cv::String> classes_name_;
	std::vector<float> scores_;
	std::vector<cv::Mat> masks_;

	// random colors list for mask show
	std::vector<cv::Scalar> colors_list_;

public:
	/** @brief Initializes the ModelPredict object.
	 * @param with_gpu (bool): Specifies whether to use GPU acceleration (default: false).
	 * @param device_id (int): ID of the GPU/CPU device to be used (default: 0).
	 * @param thread (int): Number of threads to be used for prediction (default: 1).
	 */
	ModelPredict(bool with_gpu = false, int device_id = 0, int thread = 1);

	~ModelPredict();

	/** @brief Loads the model from the specified path.
	 * @param model_path (char*): Path to the onnx model file.
	 * @return (bool) True if the model was loaded successfully, false otherwise.
	 */
	bool LoadModel(char* model_path);

	/** @brief Predicts the action based on the input image.
	 * @param inputImg (cv::Mat&): Original input image for prediction.
	 * @param score_thresh (float): Threshold score for output results (default: 0.7), 
	 * only higher score results will be saved in the model prediction object.
	 * @return (bool) True if the prediction was successful, false otherwise.
	 */
	bool PredictAction(cv::Mat& inputImg, float score_thresh = 0.7);

	/** @brief Returns an image of the predicted result plotted with masks, bounding boxes and scores
	 * @param imputImg (cv::Mat): Original input image, should be the same as PredictAction-inputImg.
	 * @param scoreThreshold (float): Threshold score imply to display the result, 
	 * the masks and bboxes above the score will be displayed on the image
	 * @return (cv::Mat) An image of the predicted result with masks, bboxes and scores.
	*/
	cv::Mat ShowPredictMask(cv::Mat& inputImg, float scoreThreshold = 0.7);

	/** @brief Returns a vector of bounding box coordinates.
	 * @return (std::vector<std::vector<cv::Point2f>>) Vector of bounding box coordinates, 
	 * where each element is a vector containing [(x_min, y_min), (x_max, y_max)].
	 */
	std::vector<std::vector<cv::Point2f>> GetBoundingBoxes();

	/** @brief Returns a vector of minimum bounding box coordinates.
	 * @return (std::vector<std::vector<cv::Point2f>>) Vector of minimum bounding box coordinates, 
	 * where each element is a vector containing [[(x0, y0), (x1, y1), (x2, y2), (x3, y3)]].
	 */
	std::vector<std::vector<cv::Point2f>> GetMinBoundingBoxes();

	/** @brief Returns the angle vector of the minimum bounding box inclination
	 * @return (std::vector<float>) Vector of minimum bounding box inclination angles.
	 */
	std::vector<float> GetBoundingBoxAngles();

	/** @brief Returns a vector of predicted masks.
	 * @return (std::vector<cv::Mat>) Vector of predicted masks.
	 */
	std::vector<cv::Mat> GetPredictMasks();

	/** @brief Returns a vector of predicted labels (id) to imply the class.
	 * @return (std::vector<int>) Vector of predicted labels (id).
	 */
	std::vector<int> GetPredictLabels();

	/** @brief Returns a vector of predicted scores.
	 * @return (std::vector<float>) Vector of predicted scores.
	 */
	std::vector<float> GetPredictScores();

private:
	void softNMSBoxes_filter(float score_threshold = 0.5, float nms_threshold = 0.6);

	void calcMinBoundingBoxes();

	void drawDashRect(cv::Mat& img, cv::Point p1, cv::Point p2, 
					  cv::Scalar& color, int thickness = 1);

	cv::Scalar hsv_to_rgb(std::vector<float> hsv);

	std::vector<cv::Scalar> random_colors(int nbr, bool bright=true);

	Ort::Value create_tensor(const cv::Mat &mat, 
							 const std::vector<int64_t> &tensor_dims,
                             const Ort::MemoryInfo &memory_info_handler,
                             std::vector<float> &tensor_value_handler,
                             std::string data_format);

};

#endif // MODELPREDICT_H