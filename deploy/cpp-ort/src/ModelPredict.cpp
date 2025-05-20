/********************************************************************************
 * @file ModelPredict.cpp
 * @brief ModelPredict.cpp is designed for C++ deploying onnx models. The model inference is 
 * implemented based on onnxruntime. Due to the API incompatibility between different 
 * versions of onnxruntime, currently this version supports running on onnxruntime 1.12.1~1.18.1.
 * @author Pan, Jiabin
 * @date 2023.06.26
 * @company Shanghai Fanuc Robotics Co., Ltd.
 * @license Apache License, Version 2.0
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ********************************************************************************/
#include "ModelPredict.h"

#include <fstream>
#include <iostream>
#include <openssl/evp.h>
#include <openssl/aes.h>
#if WIN32
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>

#include <numeric>  // Required for std::accumulate

using namespace std;
using namespace cv;

ModelPredict::ModelPredict(bool gpu, int device_id, int threads){
	// create onnxruntime running environment
    env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "OnnxModel");
	session_ops_.SetIntraOpNumThreads(threads);	// op thread
    session_ops_.SetGraphOptimizationLevel(
		GraphOptimizationLevel::ORT_ENABLE_ALL);	// Enable all possible optimizations

	// if used GPU, shared lib onnxruntime_providers_cuda will be load
	// it will auto inference on cpu if no gpu or cuda on the computer
#if WITH_GPU == true
    if (!gpu) {
        std::cout << "WARNING: GPU option is not selected. The model will run on the CPU.\n";
    } else {
        try {
            // Attempt to accelerate model inference using GPU with CUDA
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_ops_, device_id));
            std::cout << "INFO: Successfully enabled CUDA on GPU: " << device_id << std::endl;
        } catch (const Ort::Exception& exception) {
            std::cerr << "WARNING: " << exception.what() << std::endl
                      << "WARNING: Failed to enable CUDA. The model will run on the CPU instead.\n";
            // Uncomment the following line to throw an error if CUDA is not detected
            // throw std::runtime_error("ERROR: No CUDA detected. Running the model on the GPU failed.\n");
        }
    }    
#endif

}

ModelPredict::~ModelPredict(){
	cout << "INFO: Destruct model" << endl;
	delete session_;
}

bool ModelPredict::LoadModel(char* model_path, TaskType task, std::string key){
    // creat session to load  model.
	cout << "INFO: Start loading model." << endl;

	if (key != ""){
		// Decrypt the model file
    	std::vector<uint8_t> decrypted_model_data = DecryptModelFile(model_path, key);
		session_ = new Ort::Session(env_, decrypted_model_data.data(), decrypted_model_data.size(), session_ops_);
	} else {
#ifdef WIN32
		int bufSize = MultiByteToWideChar(CP_ACP, 0, model_path, -1, NULL, 0);
		wchar_t* w_model_path = new wchar_t[bufSize];
		MultiByteToWideChar(CP_ACP, 0, model_path, -1, w_model_path, bufSize);
		// Create model session and load model
		try {
			session_ = new Ort::Session(env_, w_model_path, session_ops_);
		}
		catch (const Ort::Exception& exception) {
			std::cerr << "Error: " << exception.what() << std::endl;
			return false;
		}
#else
		try {
			session_ = new Ort::Session(env_, model_path, session_ops_);
		}
		catch (const Ort::Exception& exception) {
			std::cerr << "Error: " << exception.what() << std::endl;
			return false;
		}
#endif
	}

	// print model input layer (node names, types, shape etc.)
    size_t num_input_nodes = session_->GetInputCount();
	Ort::AllocatorWithDefaultOptions ort_alloc_in;
	for (size_t i = 0; i < num_input_nodes; i++){
		input_names_ptr_.push_back(session_->GetInputNameAllocated(i, ort_alloc_in));
		input_names_.push_back(input_names_ptr_[i].get());
		cout << "INFO: Model input name-[" << i << "] is: " 
			<< input_names_[i] << endl;
	}

	size_t num_output_nodes = session_->GetOutputCount();
	Ort::AllocatorWithDefaultOptions ort_alloc_out;
	for (size_t i = 0; i < num_output_nodes; i++){
		output_names_ptr_.push_back(session_->GetOutputNameAllocated(i, ort_alloc_out));
		output_names_.push_back(output_names_ptr_[i].get());
		cout << "INFO: Model output name-[" << i << "] is: "
			<< output_names_[i] << endl;
	}
	
	WarmUpModel();  		// Warm up model with virtual input
	cout << "INFO: Succeed loading model." << endl;

    task_ = task;
    if (task_ == TaskType::ObjectDet)
        cout << "INFO: Current task type: Object Detection.\n";
    else if (task_ == TaskType::InstanceSeg)
        cout << "INFO: Current task type: Instance Segmentation.\n";
    
	return true;
}

bool ModelPredict::LabelCategories(std::vector<cv::String> classes){
	classes_name_ = classes;

	return true;
}

// Computes probabilities for each query, excluding the "no-object" class
std::vector<float> compute_probabilities(const float* pred_logits, size_t num_queries, size_t num_classes) {
	// Every query include: object (num_classes) and no-object
    std::vector<float> probas(num_queries * (num_classes + 1));

    for (size_t i = 0; i < num_queries; ++i) {
        // Pointer to logits for the i-th query
        const float* query_logits = pred_logits + i * (num_classes + 1);

        // Convert logits to probabilities
        std::vector<float> logits(query_logits, query_logits + (num_classes + 1));
        float max_logit = *std::max_element(logits.begin(), logits.end());

        // Compute softmax probabilities
        std::vector<float> query_probas(num_classes + 1);
        float sum_probas = 0.0f;

        for (size_t j = 0; j < num_classes + 1; ++j) {
            query_probas[j] = std::exp(logits[j] - max_logit);
            sum_probas += query_probas[j];
        }

        for (auto& p : query_probas) {
            p /= sum_probas;
        }

        // Store probabilities in output vector, including the last class
        std::copy(query_probas.begin(), query_probas.end(), probas.begin() + i * (num_classes + 1));
    }

    return probas;
}

bool ModelPredict::PredictAction(cv::Mat& inputImg, float score_thresh){ 
	// clear results member
    bboxes_.clear();
	minbboxes_.clear();
	labels_.clear();
	scores_.clear();
	masks_.clear();

    // input_dims {batch_size:1, chanel:3, height: ,width: }
	auto input_dims = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	auto output_dims = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	auto model_in_height = input_dims[2];
	auto model_in_width = input_dims[3];

	cv::Mat inferImg = inputImg.clone();		// inferImg is the model input
    if (model_in_height != inputImg.rows || model_in_width != inputImg.cols){
		Size in_size(model_in_width, model_in_height);		// width(cols), height(rows)
		inferImg = letterbox(inferImg, in_size);

		// register input image shape to restore model output
		img_info_.diff_with_model = true;
		img_info_.in_img_size = cv::Size(inputImg.cols, inputImg.rows);
	} else
		img_info_.diff_with_model = false;

	// construct input image tensor
	std::vector<float> tensor_value;
	std::vector<Ort::Value> input_tensors;
	input_tensors.push_back(create_tensor(inferImg, input_dims, memory_info_, tensor_value, "CHW"));

	// inference run
	double timeStart = (double)getTickCount();
	std::vector<Ort::Value> output_tensors;
	try {
		output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), 
			input_tensors.data(), input_tensors.size(), output_names_.data(), output_names_.size());
	}
	catch (const Ort::Exception& exception) {
		std::cerr << "Error: " << exception.what() << std::endl;
		return false;
	}
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "Inference time consume : " << nTime  << " s."<< endl;
	
	// Allocate outputData from output tensors ptr
	using DataOutputType = std::pair<float*, std::vector<int64_t>>;
	std::vector<DataOutputType> outputData;
	outputData.reserve(1);

	for (auto& elem : output_tensors) {
		outputData.emplace_back(std::make_pair(std::move(elem.GetTensorMutableData<float>()),
			elem.GetTensorTypeAndShapeInfo().GetShape()));
	}

    // Process predictions
    float* pred = outputData[0].first;
	std::vector<int64_t> shape_pred = outputData[0].second;		// Extract shapes
    int nc = shape_pred[2] - 1 - 4;

    float* pred_masks_proto;
    std::vector<int64_t> shape_pred_masks_proto;
    if (task_ == TaskType::InstanceSeg) {
        pred_masks_proto = outputData[4].first;
        shape_pred_masks_proto = outputData[4].second;

        nc -= 32;       // TODO: 32 is the dimension of mask coeff
    }

	auto all_batch_detections = non_max_suppression(pred, shape_pred, nc);      

    // Process detections
    // FIXME: Only one batch supported here (assuming batch size = 1)
    for (const std::vector<std::vector<float>> & det : all_batch_detections) {
        if (!det.empty()) {
			// Iterate over each detection in the batch
			for (const auto& detection : det) {
				// Extract bounding box (xyxy), confidence, and class (assuming detection format: [x1, y1, x2, y2, confidence, class])
				std::array<float, 4> box = {detection[0], detection[1], detection[2], detection[3]};  // Bounding box (xyxy)
				float conf = detection[4];  // Confidence
				int cls = static_cast<int>(detection[5]);  // Class

				// Store the rescaled box, confidence, and class
				bboxes_.push_back(box);     // Rescaled coordinates
				scores_.push_back(conf);    // Confidence score
				labels_.push_back(cls);     // Class ID

                if (!shape_pred_masks_proto.empty()) {
                    std::vector<float> masks_coeff;
                    masks_coeff.insert(masks_coeff.end(), detection.begin() + 5 + 1, detection.end());
                    
                    // Proc mask and resize to original image size
                    cv::Mat instanceMask = computeInstanceMask(pred_masks_proto, shape_pred_masks_proto, masks_coeff, inferImg.size());
                    cv::Mat cleanedMask = applyBoxMaskConstraint(instanceMask, box);

                    masks_.push_back(cleanedMask);
                }       
			}

            // Rescale boxes to original image shape
			rescale_coords(inferImg.size(), bboxes_, inputImg.size());

            // Recover masks to original image shape
            if (!masks_.empty())
                recoverMasksToOriginalSize(masks_, inputImg.size());
            
            // for (size_t im = 0; im < masks_.size(); im++)
            // {   
            //     cv::Mat fitMask;
            //     cv::resize(masks_[im], fitMask, cv::Size(), 0.2, 0.2);
            //     std::string maskWinName = "Mask window - " + std::to_string(im);
            //     cv::namedWindow(maskWinName, cv::WINDOW_NORMAL);
            //     cv::imshow(maskWinName, fitMask);
            //     cv::waitKey(10);
            // }
            
        }
    }

	return true;
}

cv::Mat ModelPredict::RenderInference(cv::Mat& inputImg, float scoreThreshold) {
    assert(bboxes_.size() == labels_.size());

	// Find the maximum value in the labels vector and expand colors list
    auto max_iter = std::max_element(labels_.begin(), labels_.end());
	int max_value;
	if (max_iter != labels_.end())
		max_value = *max_iter;
	else
		max_value = 0;
	GetColorsList(colors_list_, max_value + 1);		// Allocate colors to colors_list_

    cv::Mat img_render = inputImg.clone();
	
	// Filter low socre results (filter seq: n-1->0)
	size_t nbr_result = bboxes_.size();
	for (size_t i = nbr_result; i > 0; i--){
		if (scores_[i-1] < scoreThreshold){
			bboxes_.erase(std::begin(bboxes_)+i-1);
			masks_.erase(std::begin(masks_)+i-1);
			labels_.erase(std::begin(labels_)+i-1);
			scores_.erase(std::begin(scores_)+i-1);
		}else
			continue;
	}
	if (bboxes_.size() == 0)
		return img_render;

	// Define scaling factor based on image size
    double scaleFactor = std::min(img_render.cols / 1920.0, img_render.rows / 1080.0); // example for 1920x1080 resolution

	// Adjust line thickness and font size based on scaling factor
    int lineThickness = static_cast<int>(5 * scaleFactor);
    double fontSize = 1.5 * scaleFactor;
	
	// -----------------------Draw bbox and labels-----------------------//
	for (size_t i = 0; i < bboxes_.size(); ++i) {
		// Get label name
		uint64_t classIdx = labels_[i];
		string class_name;
		if (classes_name_.size()!=0)
			class_name = classes_name_[classIdx];
		else
			class_name = "target";

		// Read color from colors list
		cv::Scalar& curColor = colors_list_[classIdx];

		// Get score and transfer to string
		float score = scores_[i];
		string str_score = to_string(score).substr(0, to_string(score).find(".") + 4);

		// Marker (class name and predict score)
		cv::String marker = class_name + " " + str_score;
		
		// Draw bbox as a solid rectangle
		auto& curBbox = bboxes_[i]; // current bounding box in loop (curBbox: x0, y0, x1, y1)
		cv::rectangle(img_render, cv::Point(curBbox[0], curBbox[1]), cv::Point(curBbox[2], curBbox[3]), curColor, lineThickness);

		// Draw marker (class name and score)
		int baseLine = 0;
		cv::Size labelSize = cv::getTextSize(marker, cv::FONT_HERSHEY_COMPLEX, fontSize, lineThickness * 0.5, &baseLine);
		
		// Define padding for the bounding box around the text
		int padding = 15; // Adjust padding as needed
		cv::Rect textBox(cv::Point(curBbox[0] + 20, curBbox[1] - labelSize.height - padding),
						cv::Size(labelSize.width + 2 * padding, labelSize.height + 2 * padding));

		// Draw filled rectangle background for the text with tan color
    	cv::rectangle(img_render, textBox, cv::Scalar(140, 180, 210), cv::FILLED); // Tan color fill

		// Draw black border around the text background
    	cv::rectangle(img_render, textBox, cv::Scalar(0, 0, 0), lineThickness * 0.5); // Black border with thickness 2

		// Draw text
		cv::Point textOrg(textBox.x + padding, textBox.y + padding + labelSize.height);
		cv::putText(img_render, marker, textOrg, cv::FONT_HERSHEY_SIMPLEX, fontSize, cv::Scalar(0, 0, 0), lineThickness * 0.7);
	}

	// -----------------------Visualize masks-----------------------//
    for (size_t i = 0; i < masks_.size(); ++i) {
		// Get label name
		uint64_t classIdx = labels_[i];
		string class_name;
		if (classes_name_.size()!=0)
			class_name = classes_name_[classIdx];
		else
			class_name = "target";

		// Read color from colors list
		cv::Scalar& curColor = colors_list_[classIdx];

        cv::Mat curMask = masks_[i].clone();

        cv::Mat colored_img = (0.2 * curColor + 0.8 * img_render);	// splash transparent color to image
        colored_img.convertTo(colored_img, CV_8UC3);

        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        curMask.convertTo(curMask, CV_8U);

		// draw mask contour on colored image
        cv::findContours(curMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(colored_img, contours, -1, curColor, 8, cv::LINE_8, hierarchy, 100);
        colored_img.copyTo(img_render, curMask);		// copy colored mask region to result
    }

    return img_render;
}

cv::Mat ModelPredict::letterbox(const cv::Mat& img, cv::Size new_shape, 
                  cv::Scalar color, bool auto_resize, 
                  bool scaleFill, bool scaleup, int stride) {
    // Get current image size
    cv::Size shape = img.size();  // width, height
    
    // Compute scale ratio
    float r = std::min(new_shape.height / float(shape.height), new_shape.width / float(shape.width));
    if (!scaleup) {
        r = std::min(r, 1.0f);
    }
    
    // Compute new image size after scaling
    cv::Size new_unpad(std::round(shape.width * r), std::round(shape.height * r));
    
    // Compute padding
    float dw = new_shape.width - new_unpad.width;
    float dh = new_shape.height - new_unpad.height;
    if (auto_resize) {
        dw = std::fmod(dw, stride);
        dh = std::fmod(dh, stride);
    } else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_unpad = new_shape;
        r = new_shape.width / float(shape.width);
    }
    
    // Distribute padding equally
    dw /= 2.0f;
    dh /= 2.0f;
    
    // Resize the image
    cv::Mat resized_img;
    if (shape != new_unpad) {
        cv::resize(img, resized_img, new_unpad, 0, 0, cv::INTER_LINEAR);
    } else {
        resized_img = img;
    }
    
    // Add padding to the image
    cv::Mat padded_img;
    cv::copyMakeBorder(resized_img, padded_img, int(std::round(dh - 0.1f)), 
                       int(std::round(dh + 0.1f)), int(std::round(dw - 0.1f)), 
                       int(std::round(dw + 0.1f)), cv::BORDER_CONSTANT, color);

    return padded_img;
}

// Function to Compute Intersection over Union (IoU)
float ModelPredict::iou(const std::vector<float>& box1, const std::vector<float>& box2) {
    float x1 = std::max(box1[0], box2[0]);
    float y1 = std::max(box1[1], box2[1]);
    float x2 = std::min(box1[2], box2[2]);
    float y2 = std::min(box1[3], box2[3]);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);

    float intersection = w * h;
    float union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) +
                       (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection;

    return intersection / union_area;
}

// Function to Convert xywh to xyxy
void ModelPredict::xywh2xyxy(std::vector<float>& box) {
    float x_center = box[0];
    float y_center = box[1];
    float width = box[2];
    float height = box[3];

    box[0] = x_center - width / 2;  // x_min
    box[1] = y_center - height / 2; // y_min
    box[2] = x_center + width / 2;  // x_max
    box[3] = y_center + height / 2; // y_max
}

// Function for Non-Maximum Suppression (NMS)
std::vector<std::vector<std::vector<float>>> ModelPredict::non_max_suppression(
    float* pred, const std::vector<int64_t>& shape_pred, const int nc, 
    float conf_thres, float iou_thres, bool multi_label, int max_det) {

    int pred_size = shape_pred[0] * shape_pred[1] * shape_pred[2];  // Total number of predictions
    std::vector<std::vector<std::vector<float>>> output(shape_pred[0]);  // Output for each batch

    // Convert raw pointer to a vector of predictions
    std::vector<std::vector<float>> predictions;
    for (int i = 0; i < shape_pred[1]; ++i) {
        predictions.push_back(std::vector<float>(pred + i * shape_pred[2], pred + (i + 1) * shape_pred[2]));
    }

    // Process each prediction batch
    for (int xi = 0; xi < shape_pred[0]; ++xi) {  // Loop over batch size
        std::vector<std::vector<float>> items_pred;

        // Iterate over all the boxes in the current prediction
        for (int i = 0; i < shape_pred[1]; ++i) {  // Loop over grid size
            float* current_pred = pred + xi * pred_size + i * shape_pred[2];  // Offset by batch and box index

            // Extract prediction (xywh + object confidence)
            float confidence = current_pred[4];  // Object confidence score

            if (confidence < conf_thres) {
                continue;
            }

            std::vector<float> item = {current_pred[0], current_pred[1], current_pred[2], current_pred[3]};  // [x_center, y_center, width, height]
            xywh2xyxy(item);  // Convert to [x_min, y_min, x_max, y_max]

            // Get scores of all calss
            std::vector<float> class_scores;
            for (int j = 5; j < 5 + nc; ++j) {
                class_scores.push_back(current_pred[j]);
            }

            // Get extra coeff: mask_coeff in yolo-seg model
            std::vector<float> extra_coeff;
            for (size_t j = 5 + nc; j < shape_pred[2]; ++j) {
                extra_coeff.push_back(current_pred[j]);
            }

            // Pick the highest class score
            auto max_class_score_iter = std::max_element(class_scores.begin(), class_scores.end());
            float max_class_score = *max_class_score_iter;
            int class_idx = std::distance(class_scores.begin(), max_class_score_iter);

            if (class_idx >= nc) {
                // If class not in selected classes, throw an error.
                throw std::runtime_error("Detected class ID " + std::to_string(class_idx) + " is not in the allowed class list.");  
            }

            item.push_back(max_class_score);  // Add class score to the box
            item.push_back(static_cast<float>(class_idx));  // Add class index to the box
            if (!extra_coeff.empty()) 
                item.insert(item.end(), extra_coeff.begin(), extra_coeff.end());

            items_pred.push_back(item);
        }

        // Sort by confidence score (descending)
        std::sort(items_pred.begin(), items_pred.end(), [](const std::vector<float>& a, const std::vector<float>& b) {
            return a[4] > b[4];
        });

		// Apply NMS to filter boxes: Value keep is the output of one batch
        std::vector<std::vector<float>> keep;
        while (!items_pred.empty()) {
            keep.push_back(items_pred[0]);  // Keep the first (highest confidence) box
            items_pred.erase(items_pred.begin());

            // Calculate IoU with remaining boxes and remove those with high IoU
            items_pred.erase(std::remove_if(items_pred.begin(), items_pred.end(), [&](const std::vector<float>& box) {
                return iou(keep.back(), box) > iou_thres;
            }), items_pred.end());

            if (keep.size() >= max_det)
                break;
        }

        output[xi] = keep;  // Store final selected boxes for this batch
    }

    return output;
}

// Sigmoid function
inline float ModelPredict::sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

// Generate the final instance mask and convert it to cv::Mat
cv::Mat ModelPredict::computeInstanceMask(
    float* masks,                           // Pointer to proto masks: (1, C, H, W)
    const std::vector<int64_t>& shape,      // shape = {1, C, H, W}
    const std::vector<float>& mask_coeff,   // Coefficients for this instance (length = C)
    const cv::Size& infer_size,             
    float threshold,                        // Threshold to binarize mask
    bool applyMorph                         // Whether to apply morphology
) {
    int c = static_cast<int>(shape[1]); // channels (number of bases)
    int h = static_cast<int>(shape[2]);
    int w = static_cast<int>(shape[3]);

    // 1. Linear combination
    cv::Mat mask = cv::Mat::zeros(h, w, CV_32F);
    for (int i = 0; i < c; ++i) {
        const float* proto = masks + i * h * w;
        cv::Mat protoMat(h, w, CV_32F, const_cast<float*>(proto)); // No copy
        mask += protoMat * mask_coeff[i];
    }

    // 2. Sigmoid activation
    cv::Mat activatedMask;
    cv::exp(-mask, activatedMask);
    activatedMask = 1.0f / (1.0f + activatedMask);

    // 3. Resize to original image size
    cv::Mat resizedMask;
    cv::resize(activatedMask, resizedMask, infer_size, 0, 0, cv::INTER_LINEAR);

    // 4. Thresholding to binarize
    cv::Mat binaryMask;
    cv::threshold(resizedMask, binaryMask, threshold, 1.0, cv::THRESH_BINARY);

    // 5. Optional: morphological operations to clean up noise
    if (applyMorph) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_CLOSE, kernel);
    }

    return binaryMask;
}

// Suppress all mask regions outside the box (x1, y1, x2, y2)
cv::Mat ModelPredict::applyBoxMaskConstraint(const cv::Mat& mask, const std::array<float, 4>& box_xyxy) {
    CV_Assert(mask.type() == CV_32F); // Ensure float mask

    // Clone the input mask
    cv::Mat masked = cv::Mat::zeros(mask.size(), CV_32F);

    // Clamp coordinates to image bounds
    int x1 = std::clamp(static_cast<int>(std::floor(box_xyxy[0])), 0, mask.cols - 1);
    int y1 = std::clamp(static_cast<int>(std::floor(box_xyxy[1])), 0, mask.rows - 1);
    int x2 = std::clamp(static_cast<int>(std::ceil(box_xyxy[2])), 0, mask.cols - 1);
    int y2 = std::clamp(static_cast<int>(std::ceil(box_xyxy[3])), 0, mask.rows - 1);

    // Copy only the box region from original mask
    mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(masked(cv::Rect(x1, y1, x2 - x1, y2 - y1)));

    return masked;
}

// Reverse letterbox: recover mask alignment to original image size by removing padding and rescaling
void ModelPredict::recoverMasksToOriginalSize(
    std::vector<cv::Mat>& masks, 
    const cv::Size& oriImgSize, 
    const cv::Size& modelInputSize) 
{
    int model_w = modelInputSize.width;
    int model_h = modelInputSize.height;
    int ori_w = oriImgSize.width;
    int ori_h = oriImgSize.height;

    // Compute scale and padding
    float scale = std::min(static_cast<float>(model_w) / ori_w, static_cast<float>(model_h) / ori_h);
    int new_w = static_cast<int>(round(ori_w * scale));
    int new_h = static_cast<int>(round(ori_h * scale));
    int pad_x = (model_w - new_w) / 2;
    int pad_y = (model_h - new_h) / 2;

    // Define crop ROI
    cv::Rect roi(pad_x, pad_y, new_w, new_h);

    for (auto& mask : masks) {
        CV_Assert(mask.type() == CV_32F);
        CV_Assert(mask.rows == model_h && mask.cols == model_w);

        // Crop and resize
        cv::Mat cropped = mask(roi);
        cv::resize(cropped, mask, oriImgSize, 0, 0, cv::INTER_LINEAR);  // in-place overwrite
    }
}

// Function to Clip Coordinates within bounds of img0_shape
void ModelPredict::clip_coords(std::vector<std::array<float, 4>>& coords, const cv::Size& img0_shape) {
    // Clip each coordinate (xyxy) to ensure it's within the image bounds
    for (auto& box : coords) {
        box[0] = std::max(0.f, std::min(box[0], static_cast<float>(img0_shape.width - 1)));  // x_min
        box[1] = std::max(0.f, std::min(box[1], static_cast<float>(img0_shape.height - 1)));  // y_min
        box[2] = std::max(0.f, std::min(box[2], static_cast<float>(img0_shape.width - 1)));  // x_max
        box[3] = std::max(0.f, std::min(box[3], static_cast<float>(img0_shape.height - 1)));  // y_max
    }
}

// Function to Scale Coordinates from img1 to img0
void ModelPredict::rescale_coords(const cv::Size& img1_shape, std::vector<std::array<float, 4>>& coords,
                                const cv::Size& img0_shape, const std::vector<std::vector<float>>& ratio_pad) {
    float gain, pad_x, pad_y;

    if (ratio_pad.empty()) {  // Calculate from img0_shape
        gain = std::min(static_cast<float>(img1_shape.height) / img0_shape.height, static_cast<float>(img1_shape.width) / img0_shape.width);
        pad_x = (img1_shape.width - img0_shape.width * gain) / 2.0f;
        pad_y = (img1_shape.height - img0_shape.height * gain) / 2.0f;
    } else {  // Use provided ratio_pad
        gain = ratio_pad[0][0];
        pad_x = ratio_pad[1][0];
        pad_y = ratio_pad[1][1];
    }

    // Apply padding adjustments and scaling
    for (auto& box : coords) {
        box[0] -= pad_x;  // x padding
        box[1] -= pad_y;  // y padding
        box[2] -= pad_x;  // x padding
        box[3] -= pad_y;  // y padding

        // Rescale coordinates
        box[0] /= gain;
        box[1] /= gain;
        box[2] /= gain;
        box[3] /= gain;
    }

    // Clip coordinates to be within the bounds of img0_shape
    clip_coords(coords, img0_shape);
}

std::vector<std::vector<cv::Point2f>> ModelPredict::GetBoundingBoxes()
{
	std::vector<std::vector<cv::Point2f>> bboxes;
	for(auto box: bboxes_){		// box: each bbox in array data
		std::vector<cv::Point2f> cv_bbox = {		// cv_bbox: each bbox in cv point data
			cv::Point2f(box[0], box[1]),
			cv::Point2f(box[2], box[3])
		};
		bboxes.push_back(cv_bbox);
	}
	return bboxes;
}

std::vector<std::vector<cv::Point2f>> ModelPredict::GetMinBoundingBoxes()
{
    std::vector<std::vector<cv::Point2f>> min_bboxes; 	// minimum bounding boxes

    // Traverse the contour in the mask
	for (size_t i = 0; i < minbboxes_.size(); i++)
	{
		// Convert mask to binary image to diaplay
        cv::Mat binaryImage;
        cv::threshold(masks_[i], binaryImage, 0.5, 255, cv::THRESH_BINARY);

		// Convert the minbboxes from array data to cv Point data
		std::vector<cv::Point2f> min_bbox;
		for (size_t j = 0; j < minbboxes_[i].size(); j += 2) {
			cv::Point2f point(minbboxes_[i][j], minbboxes_[i][j + 1]);
			min_bbox.push_back(point);
		}

		// Draw bounding box on binary image
		// Convert the rotated rectangle vertices to integer points
		cv::Point verticesInt[4];
		for (int j = 0; j < 4; j++)
			verticesInt[j] = cv::Point(static_cast<int>(min_bbox[j].x), static_cast<int>(min_bbox[j].y));
		cv::polylines(binaryImage, std::vector<cv::Point>{verticesInt, verticesInt + 4}, true, cv::Scalar(255), 2);

		// cv::imshow("Mask in binary format", binaryImage);
		// cv::waitKey(0);

		min_bboxes.push_back(min_bbox);
	}

    return min_bboxes;
}

std::vector<float> ModelPredict::GetBoundingBoxAngles()
{
	// Lambda function to calculate bbox inclination angles
	auto CalcbBoxIncline = [](std::vector<cv::Point2f> box) 
	{
		float angle;
		RotatedRect rect = minAreaRect(box);
    	
		if (rect.size.width > rect.size.height) {
			angle = rect.angle; 	// The angle of the length side (longer side) of the rectangle
		} else
			angle = rect.angle + 90.0f; 	// Add 90 degrees for the angle of the length side

		return angle;
	};

	std::vector<float> minbBoxAngels;
	auto bboxes = GetBoundingBoxes();	// bounding box in cv data, [x_min, y_min, x_max, y_max]
	
	// reshape bboxes to [x0, y0, ..., y3]
	for(std::vector<cv::Point2f> &bbox: bboxes){
		// point: [x_min, y_min] view as [x0, y0]
		cv::Point2f pnt1(bbox[0].x, bbox[1].y);
		cv::Point2f pnt3(bbox[1].x, bbox[0].y);

		bbox.insert(bbox.begin() + 1, pnt1);
		bbox.push_back(pnt3);
	}

	auto minbboxes = GetMinBoundingBoxes();		// bounding box in cv data, [x0, y0, ..., y3]
	for (size_t i = 0; i < minbboxes.size(); i++)
	{
		
		float angle1 = CalcbBoxIncline(bboxes[i]);
		float angle2 = CalcbBoxIncline(minbboxes[i]);

		float angleDiff = angle2 - angle1;

		// limit angeels in range [-90, 90]
		while (angleDiff < -90.0f)
			angleDiff += 180.0f;
		while (angleDiff > 90.0f)
			angleDiff -= 180.0f;
		
		minbBoxAngels.push_back(angleDiff);
	}

	return minbBoxAngels;
}

std::vector<cv::Mat> ModelPredict::GetPredictMasks()
{
	return masks_;
}

std::vector<int> ModelPredict::GetPredictLabels()
{
	return labels_;
}

std::vector<float> ModelPredict::GetPredictScores()
{
	return scores_;
}



cv::Scalar ModelPredict::hsv_to_rgb(std::vector<float> hsv){
	// hsv value convert to rgb (0~255)
	float alpha = 0.5;

	cv::Scalar rgb;
	float h = hsv[0], s = hsv[1], v = hsv[2];
	if(s == 0.0){
		rgb = {v, v, v};
		return rgb;
	}
    int i = int(h*6.0); // assume int() truncates!
	float f = (h*6.0) - i;
    float p = v*(1.0 - s);
    float q = v*(1.0 - s*f);
    float t = v*(1.0 - s*(1.0-f));
	i = i % 6;

	// get rgb (value range: 0~1)
	switch (i)
	{
	case 0:
		rgb[0] = v;
		rgb[1] = t;
		rgb[2] = p;
		break;
	case 1:
		rgb[0] = q;
		rgb[1] = v;
		rgb[2] = p;
		break;
	case 2:
		rgb[0] = p;
		rgb[1] = v;
		rgb[2] = t;
		break;
	case 3:
		rgb[0] = p;
		rgb[1] = q;
		rgb[2] = v;
		break;	
	case 4:
		rgb[0] = t;
		rgb[1] = p;
		rgb[2] = v;
		break;	
	case 5:
		rgb[0] = v;
		rgb[1] = p;
		rgb[2] = q;
		break;
	default:
		break;
	}

	// transfer rgb (value range: 0~255)
	for (size_t i = 0; i < 3; i++){
		rgb[i] = rgb[i] * 255;
	}

	return rgb;
}

std::vector<cv::Scalar> ModelPredict::random_colors(int nbr, bool bright){
	// Generate random colors.
    // To get visually distinct colors, generate them in HSV space then
    // convert to RGB.
	float brightness = 1.0;
	if (!bright)
		brightness = (float) 0.7;
	
	std::vector<cv::Scalar> list_colors;
	for (size_t i = 0; i < nbr; i++){
		std::vector<float> hsv = {i/float(nbr), 1, brightness};
		list_colors.push_back(hsv_to_rgb(hsv));
	}
    return list_colors;
}

void printMatData(const cv::Mat& mat) {
    // Check if the mat is empty
    if (mat.empty()) {
        std::cout << "Empty matrix!" << std::endl;
        return;
    }

    // Ensure that the matrix is of type float32 (CV_32F)
    if (mat.type() != CV_32FC3) {
        std::cerr << "Error: Mat is not of type CV_32FC3" << std::endl;
        return;
    }

    // Print the matrix data (values of each pixel)
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            // Access each pixel (3 channels: R, G, B)
            cv::Vec3f pixel = mat.at<cv::Vec3f>(i, j);  // Vec3f is for 3 float channels

            // Print the normalized values (R, G, B)
            std::cout << "Pixel (" << i << ", " << j << "): ";
            std::cout << "R: " << pixel[0] << ", G: " << pixel[1] << ", B: " << pixel[2] << std::endl;
        }
    }
}

// BUG: tensor_value_handler should be std::vector<std::vector<float>> when inference batch size > 1 
Ort::Value ModelPredict::create_tensor(const cv::Mat &mat,
                         const std::vector<int64_t> &tensor_dims,
                         const Ort::MemoryInfo &memory_info_handler,
                         std::vector<float> &tensor_value_handler,
                         const std::string &data_format) {

    const unsigned int rows = mat.rows;
    const unsigned int cols = mat.cols;
    const unsigned int channels = mat.channels();

    cv::Mat mat_ref;
    if (mat.type() != CV_32FC(channels)) 
        mat.convertTo(mat_ref, CV_32FC(channels));
    else 
        mat_ref = mat; // reference only. zero-time cost. support 1/2/3/... channels

    // Convert BGR to RGB
    cv::Mat rgb_mat;
    cv::cvtColor(mat_ref, rgb_mat, cv::COLOR_BGR2RGB);

    // Normalize to [0, 1]
    cv::Mat norm_mat_ref;
    rgb_mat.convertTo(norm_mat_ref, CV_32FC3, 1.0 / 255.0);

    if (tensor_dims.size() != 4) 
        throw std::runtime_error("dims mismatch.");
    if (tensor_dims.at(0) != 1) 
        throw std::runtime_error("batch != 1");

    // Determine target dimensions
    const unsigned int target_channel = tensor_dims.at(1);
    const unsigned int target_height = tensor_dims.at(2);
    const unsigned int target_width = tensor_dims.at(3);
    const unsigned int target_tensor_size = target_channel * target_height * target_width;

    if (target_channel != channels) 
        throw std::runtime_error("channel mismatch!");

    tensor_value_handler.resize(target_tensor_size);

    // Resize image to target dimensions
    cv::Mat resize_mat_ref;
    if (target_height != rows || target_width != cols)
        cv::resize(norm_mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
    else 
        resize_mat_ref = norm_mat_ref; // reference only. zero-time cost.

	// printMatData(resize_mat_ref);

    if (data_format == "CHW") {
        std::vector<cv::Mat> mat_channels;
        cv::split(resize_mat_ref, mat_channels); // mat_channels: R->G->B

        // CXHXW transform
        for (unsigned int i = 0; i < channels; ++i)
            std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
                        mat_channels.at(i).data, 
                        target_height * target_width * sizeof(float));
    } else {
        // HXWXC
        std::memcpy(tensor_value_handler.data(), resize_mat_ref.data, 
                    target_tensor_size * sizeof(float));
    }

    return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                                           target_tensor_size, tensor_dims.data(),
                                           tensor_dims.size());
}

// Function to convert ONNX Runtime output tensors to std::vector<float>
std::vector<float> ModelPredict::extract_tensor_data(const Ort::Value& tensor) {
    // Get the tensor shape
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }

    // Get the tensor data
    const float* tensor_data = tensor.GetTensorData<float>();
    std::vector<float> data(tensor_data, tensor_data + num_elements);

    return data;
}

void ModelPredict::GetColorsList(std::vector<cv::Scalar>& colors_list, size_t num_classes) {
	// Define the colors list for visualization (converted from RGB to BGR)
	colors_list = {
		cv::Scalar(741, 447, 0),   // RGB [0.000, 0.447, 0.741] converted to BGR
		cv::Scalar(98, 325, 850),   // RGB [0.850, 0.325, 0.098] converted to BGR
		cv::Scalar(125, 694, 929),  // RGB [0.929, 0.694, 0.125] converted to BGR
		cv::Scalar(556, 184, 494),  // RGB [0.494, 0.184, 0.556] converted to BGR
		cv::Scalar(188, 674, 466),  // RGB [0.466, 0.674, 0.188] converted to BGR
		cv::Scalar(933, 745, 301),  // RGB [0.301, 0.745, 0.933] converted to BGR
		cv::Scalar(1, 972, 941)     // RGB [0.941, 0.972, 1.000] converted to BGR
	};

	if (colors_list.size() < num_classes)
	{
		std::vector<cv::Scalar> expanded_colors_list;
    	expanded_colors_list.reserve(num_classes);
		size_t colors_list_size = colors_list.size();
		for (size_t i = 0; i < num_classes; ++i) {
			expanded_colors_list.push_back(colors_list[i % colors_list_size]);
		}
		colors_list = expanded_colors_list;
	}
}

// BUG: dummy_data should be std::vector<std::vector<float>> when inference batch size > 1 
void ModelPredict::WarmUpModel() {
    // Check if input_names_ and output_names_ are not empty
    if (input_names_.empty()) {
        throw std::runtime_error("Input names are not set.");
    }
    if (output_names_.empty()) {
        throw std::runtime_error("Output names are not set.");
    }

    // Prepare a vector to hold the dummy inputs
    std::vector<Ort::Value> dummy_inputs;

    // Iterate over input names to create dummy inputs
	std::vector<float> dummy_data;		// dummy_date claim shouble out of loop 
	for (size_t i = 0; i < input_names_.size(); i++)
	{
		// Get the shape of the input tensor
        auto input_info = session_->GetInputTypeInfo(i);
        auto input_shape = input_info.GetTensorTypeAndShapeInfo().GetShape();
        
        // Calculate the number of elements required
        size_t num_elements = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
        
        // Create a dummy data vector (filled with zeros)
        dummy_data.assign(num_elements, 0.3f);

        // Create an Ort::Value for the dummy input
        Ort::Value dummy_input = Ort::Value::CreateTensor<float>(
            memory_info_, dummy_data.data(), dummy_data.size(), input_shape.data(), input_shape.size()
        );

        dummy_inputs.push_back(std::move(dummy_input));
	}

    // Perform a dummy inference to warm up the model
    std::vector<Ort::Value> dummy_outputs;
    try {
        dummy_outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), dummy_inputs.data(), 
                                     dummy_inputs.size(), output_names_.data(), output_names_.size());
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Model warm-up failed: ") + e.what());
    }
}

// Function to decrypt the model file
std::vector<uint8_t> ModelPredict::DecryptModelFile(const char* encrypted_model_path, const std::string& key_hex) {
    // Ensure the key length is 64 hex characters (32 bytes)
    if (key_hex.size() != 64) {
        throw std::runtime_error("The encryption key must be 64 hex characters (32 bytes) for AES-256.");
    }

    // Convert the hex key to a byte array
    std::vector<uint8_t> key(32);
    for (size_t i = 0; i < key_hex.length(); i += 2) {
        key[i / 2] = static_cast<uint8_t>(std::stoi(key_hex.substr(i, 2), nullptr, 16));
    }

    // Open the encrypted model file
    std::ifstream infile(encrypted_model_path, std::ios::binary);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open the encrypted model file.");
    }

    // Read the initialization vector (IV) from the file (first 16 bytes)
    unsigned char iv[AES_BLOCK_SIZE];
    infile.read(reinterpret_cast<char*>(iv), AES_BLOCK_SIZE);

    // Read the rest of the file (encrypted data)
    std::vector<uint8_t> encrypted_data((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();

    // Create a context for decryption
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create decryption context.");
    }

    // Initialize the decryption operation with AES-256-CFB mode
    if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cfb(), nullptr, key.data(), iv)) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("Failed to initialize the decryption operation.");
    }

    // Prepare output buffer
    std::vector<uint8_t> decrypted_data(encrypted_data.size());
    int len = 0;

    // Perform the decryption
    if (1 != EVP_DecryptUpdate(ctx, decrypted_data.data(), &len, encrypted_data.data(), encrypted_data.size())) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("Decryption failed.");
    }
    int decrypted_len = len;

    // Finalize decryption
    if (1 != EVP_DecryptFinal_ex(ctx, decrypted_data.data() + len, &len)) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("Decryption finalization failed.");
    }
    decrypted_len += len;

    // Clean up
    EVP_CIPHER_CTX_free(ctx);

    // Resize the output to the actual decrypted size
    decrypted_data.resize(decrypted_len);

    return decrypted_data;
}
