#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <onnxruntime_cxx_api.h>


class OnnxRuntime {
public:
    OnnxRuntime(char* modelPath);
    void Infer(
        int aWidth,
        int aHeight,
        int aChannel,
        unsigned char* aBytes,
        std::unique_ptr<float[]>& Boxes, 
        std::unique_ptr<float[]>& ClassIndexs, 
        std::unique_ptr<float[]>& Scores, 
        std::unique_ptr<int[]>& BboxNum);

private:
    float* blobFromImage(cv::Mat& img);
    float letterbox(
        const cv::Mat& image,
        cv::Mat& out_image,
        const cv::Size& new_shape,
        int stride,
        const cv::Scalar& color,
        bool fixed_shape,
        bool scale_up);

    const Ort::Env onnxEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
    Ort::Session session = Ort::Session(nullptr);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo(nullptr);
    std::vector<const char *> input_node_names;
    std::vector<const char *> output_node_names;
};
