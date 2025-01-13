#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

class AiLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // if (severity != Severity::kINFO) {
        //     std::cout << msg << std::endl;
        // }
    }
};

enum class ModelType
{
    POSE,
    HEADPHONES,
    SMARTPHONE,
    SMOKING,
    NONE
};

struct HumanSkeleton
{
    cv::Rect2f rect;
    float conf;
    std::map<std::string, cv::Point2f> joints;
    std::vector<float> forIou()
    {
        return {rect.x, rect.y, rect.x + rect.width, rect.y + rect.height};
    }
};

class Yolo {
public:
    Yolo(const std::string model_path, ModelType model_type);
    void init(const std::string model_path, ModelType model_type);
    std::vector<HumanSkeleton> infer(cv::Mat& image);
    float letterbox(
        const cv::Mat& image,
        cv::Mat& out_image,
        const cv::Size& new_shape,
        int stride,
        const cv::Scalar& color,
        bool fixed_shape,
        bool scale_up);
    void nms(std::vector<HumanSkeleton>& humanRects, const float iou_threshold=0.45);
    float iou(const std::vector<float>& boxA, const std::vector<float>& boxB);
    float* blobFromImage(const cv::Mat& dpImage);
    void draw_objects(cv::Mat& img, float* Boxes, int* ClassIndexs, int* BboxNum);
    float* blobFromImageHCW(cv::Mat& image);
    ~Yolo();

private:
    ModelType model_type;
    std::string model_path;
    std::string input_tensor_name = "images";
    std::string output_tensor_name = "output0";
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    uint64_t in_size;
    uint64_t out_size;
    void* input_buffer;
    void* output_buffer;
    AiLogger gLogger;

    std::map<std::string, cv::Point2f> getSkeleton(float* det_output, int index, float scale, float x_offset, float y_offset);
};
