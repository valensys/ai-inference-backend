#include "TensorRT_Yolo.h"
#include "NvInferPlugin.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

Yolo::Yolo(const std::string model_path, ModelType model_type)
{
    init(model_path, model_type);
}

float Yolo::letterbox(
    const cv::Mat& image,
    cv::Mat& out_image,
    const cv::Size& new_shape = cv::Size(640, 640),
    int stride = 32,
    const cv::Scalar& color = cv::Scalar(114, 114, 114),
    bool fixed_shape = false,
    bool scale_up = true) 
{
    cv::Size shape = image.size();
    float r = std::min((float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);

    if (!scale_up) {
        r = std::min(r, 1.0f);
    }

    int newUnpad[2]{
        (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r) };

    cv::Mat tmp;
    if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
        cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
    }

    else {
        tmp = image.clone();
    }

    float dw = new_shape.width - newUnpad[0];
    float dh = new_shape.height - newUnpad[1];

    if (!fixed_shape) {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }

    dw /= 2.0f;
    dh /= 2.0f;

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return 1.0f / r;
}



void Yolo::init(const std::string model_path, ModelType model_type)
{
    std::cout << "\nModel path: " << model_path << std::endl;
    std::cout << "Model type: " << static_cast<int>(model_type) << std::endl;
    ifstream ifile(model_path, ios::in | ios::binary);
    if (!ifile)
    {
        cout << "read serialized file failed\n";
        std::abort();
    }
    this->model_type = model_type;
    this->model_path = model_path;

    ifile.seekg(0, ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, ios::beg);
    vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    cout << "model size: " << mdsize << endl;

    runtime = nvinfer1::createInferRuntime(gLogger);
    initLibNvInferPlugins(&gLogger, "");
    engine = runtime->deserializeCudaEngine((void *)&buf[0], mdsize);
    context = engine->createExecutionContext();
    if (!context)
    {
        std::cout << "create execution context failed\n";
        std::abort();
    }

    std::map<std::string, vector<int64_t>> in_out_sizes;
    std::vector<nvinfer1::DataType> in_out_types;

    std::cout << "Tensors names and shapes:" << std::endl;
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        string tensor_name = engine->getIOTensorName(i);
        in_out_types.push_back(engine->getTensorDataType(tensor_name.c_str()));
        auto tensorShape = engine->getTensorShape(tensor_name.c_str());
        for (int j = 0; j < tensorShape.nbDims; ++j)
        {
            in_out_sizes[tensor_name].push_back(tensorShape.d[j]);
        }
    }

    for (const auto &keyval : in_out_sizes)
    {
        std::cout << "Tensor name: " << keyval.first << std::endl;
        std::cout << "Tensor shape: ";
        for (const auto &val : keyval.second)
            std::cout << val << " ";
        std::cout << std::endl;
    }

    std::cout << "Tensor types: ";
    for (int i = 0; i < in_out_types.size(); ++i)
    {
        std::cout << static_cast<int>(in_out_types[i]) << " ";
    }
    std::cout << std::endl;

    uint64_t input_size = 0;

    auto input_sizes = in_out_sizes[input_tensor_name];
    auto input_dims = nvinfer1::Dims4{input_sizes[0], input_sizes[1], input_sizes[2], input_sizes[3]};
    context->setInputShape(input_tensor_name.c_str(), input_dims);
    input_size = input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * input_dims.d[3];

    nvinfer1::DataType output_type = in_out_types[1];
    auto output_dims = context->getTensorShape(output_tensor_name.c_str());

    uint64_t output_size = 1;
    for (int j = 0; j < output_dims.nbDims; j++)
    {
        output_size *= output_dims.d[j];
    }

    std::cout << "Input size = " << input_size << ", Output size = " << output_size << std::endl;

    cudaError_t state;

    state = cudaMalloc(&input_buffer, input_size * sizeof(float));
    if (state)
    {
        cout << "allocate memory failed\n";
        std::abort();
    }

    state = cudaMalloc(&output_buffer, output_size * sizeof(float));
    if (state)
    {
        cout << "allocate memory failed\n";
        std::abort();
    }

    state = cudaStreamCreate(&stream);
    if (state)
    {
        cout << "create stream failed\n";
        std::abort();
    }

    in_size = input_size;
    out_size = output_size;

    context->setTensorAddress(input_tensor_name.c_str(), input_buffer);
    context->setTensorAddress(output_tensor_name.c_str(), output_buffer);
}

float *Yolo::blobFromImageHCW(cv::Mat &image)
{
    float *blob = new float[image.total() * 3];
    int channels = 3;
    int img_h = image.rows;
    int img_w = image.cols;
    uchar color;
    
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < img_h; h++)
        {
            uchar* pixel = image.ptr<uchar>(h);
            for (size_t w = 0; w < img_w; w++)
            {
                color = *(pixel + w * channels + c);
                blob[c * img_w * img_h + h * img_w + w] = color / 255.f;
            }
        }
    }
    return blob;
}

float Yolo::iou(const std::vector<float>& boxA, const std::vector<float>& boxB)
{
    // The format of box is [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    const float eps = 1e-6;
    float iou = 0.f;
    float areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    float areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    float x1 = std::max(boxA[0], boxB[0]);
    float y1 = std::max(boxA[1], boxB[1]);
    float x2 = std::min(boxA[2], boxB[2]);
    float y2 = std::min(boxA[3], boxB[3]);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float inter = w * h;
    iou = inter / (areaA + areaB - inter + eps);
    return iou;
}

void Yolo::nms(std::vector<HumanSkeleton>& humanRects, const float iou_threshold)
{
    // The format of boxes is [[top_left_x, top_left_y, bottom_right_x, bottom_right_y, score, class_id], ...]
    // Sorting "score + class_id" is to ensure that the boxes with the same class_id are grouped together and sorted by score
    std::sort(humanRects.begin(), humanRects.end(), [](const HumanSkeleton& boxA, const HumanSkeleton& boxB) { return boxA.conf > boxB.conf;});
    for (size_t i = 0; i < humanRects.size(); ++i)
    {
        if (humanRects[i].conf == 0.f)
        {
            continue;
        }
        for (size_t j = i + 1; j < humanRects.size(); ++j)
        {
            if (iou(humanRects[i].forIou(), humanRects[j].forIou()) > iou_threshold)
            {
                humanRects[j].conf = 0.f;
            }
        }
    }
    std::erase_if(humanRects, [](const HumanSkeleton& box) { return box.conf == 0.f; });
}

float *Yolo::blobFromImage(const cv::Mat &image)
{
    cv::Size requiredSize = image.size();
    float *blob = new float[requiredSize.area()];

    for (int y = 0; y < requiredSize.height; ++y)
    {
        const uchar *pixel_dp = image.ptr<uchar>(y);
        for (int x = 0; x < requiredSize.width; ++x)
        {
            float result_dp = pixel_dp[x];

            uint ywx = y * requiredSize.width + x;
            blob[0 * requiredSize.area() + ywx] = result_dp / 255.f;
        }
    }

    return blob;
}

std::map<std::string, cv::Point2f> Yolo::getSkeleton(float *det_output, int i, float scale, float x_offset, float y_offset)
{
    std::map<std::string, cv::Point2f> joints;

    float nose_x = det_output[(8400 * 5) + i] * scale - x_offset;
    float nose_y = det_output[(8400 * 6) + i] * scale - y_offset;
    cv::Point2f nose = cv::Point2f(nose_x, nose_y);
    joints["nose"] = nose;

    float left_eye_x = det_output[(8400 * 8) + i] * scale - x_offset;
    float left_eye_y = det_output[(8400 * 9) + i] * scale - y_offset;
    cv::Point2f left_eye = cv::Point2f(left_eye_x, left_eye_y);
    joints["left_eye"] = left_eye;

    float right_eye_x = det_output[(8400 * 11) + i] * scale - x_offset;
    float right_eye_y = det_output[(8400 * 12) + i] * scale - y_offset;
    cv::Point2f right_eye = cv::Point2f(right_eye_x, right_eye_y);
    joints["right_eye"] = right_eye;

    float left_ear_x = det_output[(8400 * 14) + i] * scale - x_offset;
    float left_ear_y = det_output[(8400 * 15) + i] * scale - y_offset;
    cv::Point2f left_ear = cv::Point2f(left_ear_x, left_ear_y);
    joints["left_ear"] = left_ear;

    float right_ear_x = det_output[(8400 * 17) + i] * scale - x_offset;
    float right_ear_y = det_output[(8400 * 18) + i] * scale - y_offset;
    cv::Point2f right_ear = cv::Point2f(right_ear_x, right_ear_y);
    joints["right_ear"] = right_ear;

    float left_shoulder_x = det_output[(8400 * 20) + i] * scale - x_offset;
    float left_shoulder_y = det_output[(8400 * 21) + i] * scale - y_offset;
    cv::Point2f left_shoulder = cv::Point2f(left_shoulder_x, left_shoulder_y);
    joints["left_shoulder"] = left_shoulder;

    float right_shoulder_x = det_output[(8400 * 23) + i] * scale - x_offset;
    float right_shoulder_y = det_output[(8400 * 24) + i] * scale - y_offset;
    cv::Point2f right_shoulder = cv::Point2f(right_shoulder_x, right_shoulder_y);
    joints["right_shoulder"] = right_shoulder;

    float left_elbow_x = det_output[(8400 * 26) + i] * scale - x_offset;
    float left_elbow_y = det_output[(8400 * 27) + i] * scale - y_offset;
    cv::Point2f left_elbow = cv::Point2f(left_elbow_x, left_elbow_y);
    joints["left_elbow"] = left_elbow;

    float right_elbow_x = det_output[(8400 * 29) + i] * scale - x_offset;
    float right_elbow_y = det_output[(8400 * 30) + i] * scale - y_offset;
    cv::Point2f right_elbow = cv::Point2f(right_elbow_x, right_elbow_y);
    joints["right_elbow"] = right_elbow;

    float left_wrist_x = det_output[(8400 * 32) + i] * scale - x_offset;
    float left_wrist_y = det_output[(8400 * 33) + i] * scale - y_offset;
    cv::Point2f left_wrist = cv::Point2f(left_wrist_x, left_wrist_y);
    joints["left_wrist"] = left_wrist;

    float right_wrist_x = det_output[(8400 * 35) + i] * scale - x_offset;
    float right_wrist_y = det_output[(8400 * 36) + i] * scale - y_offset;
    cv::Point2f right_wrist = cv::Point2f(right_wrist_x, right_wrist_y);
    joints["right_wrist"] = right_wrist;

    float left_hip_x = det_output[(8400 * 38) + i] * scale - x_offset;
    float left_hip_y = det_output[(8400 * 39) + i] * scale - y_offset;
    cv::Point2f left_hip = cv::Point2f(left_hip_x, left_hip_y);
    joints["left_hip"] = left_hip;

    float right_hip_x = det_output[(8400 * 41) + i] * scale - x_offset;
    float right_hip_y = det_output[(8400 * 42) + i] * scale - y_offset;
    cv::Point2f right_hip = cv::Point2f(right_hip_x, right_hip_y);
    joints["right_hip"] = right_hip;

    float left_knee_x = det_output[(8400 * 44) + i] * scale - x_offset;
    float left_knee_y = det_output[(8400 * 45) + i] * scale - y_offset;
    cv::Point2f left_knee = cv::Point2f(left_knee_x, left_knee_y);
    joints["left_knee"] = left_knee;

    float right_knee_x = det_output[(8400 * 47) + i] * scale - x_offset;
    float right_knee_y = det_output[(8400 * 48) + i] * scale - y_offset;
    cv::Point2f right_knee = cv::Point2f(right_knee_x, right_knee_y);
    joints["right_knee"] = right_knee;

    float left_ankle_x = det_output[(8400 * 50) + i] * scale - x_offset;
    float left_ankle_y = det_output[(8400 * 51) + i] * scale - y_offset;
    cv::Point2f left_ankle = cv::Point2f(left_ankle_x, left_ankle_y);
    joints["left_ankle"] = left_ankle;

    float right_ankle_x = det_output[(8400 * 53) + i] * scale - x_offset;
    float right_ankle_y = det_output[(8400 * 54) + i] * scale - y_offset;
    cv::Point2f right_ankle = cv::Point2f(right_ankle_x, right_ankle_y);
    joints["right_ankle"] = right_ankle;

    return joints;
}

std::vector<HumanSkeleton> Yolo::infer(cv::Mat &image)
{
    static float *det_output = new float[out_size];

    cv::Mat pr_img;
    float scale = letterbox(image, pr_img, {640, 640}, 32, { 114, 114, 114 }, true);
    
    cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB);
    float *blob = blobFromImageHCW(pr_img);

    cudaError_t state = cudaMemcpyAsync(input_buffer, blob, in_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (state)
    {
        std::cout << "transmit to device failed Error = " << (int)state << std::endl;
        std::abort();
    }

    context->enqueueV3(stream);

    state = cudaMemcpyAsync(det_output, output_buffer, out_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (state)
    {
        std::cout << "transmit to host failed Error = " << (int)state << std::endl;
        std::abort();
    }

    int img_w = image.cols;
    int img_h = image.rows;
    int x_offset = (640 * scale - img_w) / 2;
    int y_offset = (640 * scale - img_h) / 2;

    int n_dets = 0;

    std::vector<HumanSkeleton> rects;

    for (int i = 0; i < 8400; i++) // dimCoords = 8400
    {
        float x = det_output[i] * scale - x_offset;
        float y = det_output[(8400 * 1) + i] * scale - y_offset;
        float w = det_output[(8400 * 2) + i];
        float h = det_output[(8400 * 3) + i];
        float conf = det_output[(8400 * 4) + i];

        float l = x - (0.5 * w) * scale;
        float t = y - (0.5 * h) * scale;
        float wid = w * scale;
        float hei = h * scale;

        if (conf < 0.5)
        {
            continue;
        }

        HumanSkeleton r;

        r.joints = getSkeleton(det_output, i, scale, x_offset, y_offset);

        r.rect = cv::Rect2f(l, t, wid, hei);
        r.conf = conf;
        rects.push_back(r);
    }

    nms(rects);
    delete[] blob;
    return rects;
}

Yolo::~Yolo()
{
    cout << "\nDESTRUCTOR OF " << this->model_path << endl;
    cudaStreamSynchronize(stream);
    cudaFree(input_buffer);
    cudaFree(output_buffer);
    cudaStreamDestroy(stream);
    context->~IExecutionContext();
    engine->~ICudaEngine();
    runtime->~IRuntime();
}