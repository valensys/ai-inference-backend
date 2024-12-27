#pragma once
#include <chrono>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <vector>
#include "globalvars.h"

namespace utilities {
    float* blobFromImage(cv::Mat& img);
    void visualize(cv::Mat& img, float *Boxes, float *ClassIndexs, int *BboxNum);
    nlohmann::json getTrackedDataInJson(std::unique_ptr<float[]>& Boxes, std::unique_ptr<float[]>& ClassIndexs, 
                                std::unique_ptr<float[]>& Scores, std::unique_ptr<int[]>& BboxNum);
    void fixBoxesForBytetrack(std::unique_ptr<float[]>& Boxes, std::unique_ptr<float[]>& ClassIndexs, 
                                std::unique_ptr<float[]>& Scores, std::unique_ptr<int[]>& BboxNum);
    void nms(std::vector<std::vector<float>>& boxes, const float iou_threshold=0.45);
}