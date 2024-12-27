#include "OnnxRuntime.h"

OnnxRuntime::OnnxRuntime(char *modelPath)
{
    auto providers = Ort::GetAvailableProviders();
    for (auto provider : providers)
    {
        std::cout << "Available providers:" << std::endl;
        std::cout << provider << std::endl;
    }

    Ort::SessionOptions sessionOptions;

    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetIntraOpNumThreads(1);

    // Optimization will take time and memory during startup
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    // CUDA options. If used.
    try
    {
        // Model path is const wchar_t*
        session = Ort::Session(onnxEnv, modelPath, sessionOptions);
        size_t num_input_nodes = 0;
        size_t num_output_nodes = 0;
        ONNXTensorElementDataType type;                         // Used to print input info
        Ort::TypeInfo *type_info;

        num_input_nodes = session.GetInputCount();
        num_output_nodes = session.GetOutputCount();
        Ort::AllocatorWithDefaultOptions allocator;

        input_node_names.push_back("images");
        output_node_names.push_back("output0");
        input_node_dims = {1, 3, 640, 640};

        std::vector<const char *> in_names, out_names;
        std::vector<int64_t> in_dims, out_dims;


        for (int i = 0; i < num_input_nodes; i++)
        {
            char *tempstring = new char[strlen(session.GetInputNameAllocated(i, allocator).get()) + 1];
            strcpy(tempstring, session.GetInputNameAllocated(i, allocator).get());
            in_names.push_back(tempstring);
            type_info = new Ort::TypeInfo(session.GetInputTypeInfo(i));
            auto tensor_info = type_info->GetTensorTypeAndShapeInfo();
            type = tensor_info.GetElementType();
            in_dims = tensor_info.GetShape();

            // print input shapes/dims

            printf("Input %d : name=%s\n", i, in_names.back());
            printf("Input %d : num_dims=%zu\n", i, in_dims.size());
            for (int j = 0; j < in_dims.size(); j++)
                printf("Input %d : dim %d=%jd\n", i, j, in_dims[j]);
            printf("Input %d : type=%d\n", i, type);

            delete (type_info);
            delete[] tempstring;
        }

        for (int i = 0; i < num_output_nodes; i++)
        {
            char *tempstring = new char[strlen(session.GetOutputNameAllocated(i, allocator).get()) + 1];
            strcpy(tempstring, session.GetOutputNameAllocated(i, allocator).get());
            out_names.push_back(tempstring);
            type_info = new Ort::TypeInfo(session.GetOutputTypeInfo(i));
            auto tensor_info = type_info->GetTensorTypeAndShapeInfo();
            type = tensor_info.GetElementType();
            out_dims = tensor_info.GetShape();

            // print input shapes/dims

            printf("Output %d : name=%s\n", i, out_names.back());
            printf("Output %d : num_dims=%zu\n", i, out_dims.size());
            for (int j = 0; j < out_dims.size(); j++)
                printf("Output %d : dim %d=%jd\n", i, j, out_dims[j]);
            printf("Output %d : type=%d\n", i, type);

            delete (type_info);
            delete[] tempstring;
        }
    }
    catch (Ort::Exception oe)
    {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
    }

    try
    {
        memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }
    catch (Ort::Exception oe)
    {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
    }
}

float OnnxRuntime::letterbox(
    const cv::Mat &image,
    cv::Mat &out_image,
    const cv::Size &new_shape = cv::Size(640, 640),
    int stride = 32,
    const cv::Scalar &color = cv::Scalar(114, 114, 114),
    bool fixed_shape = false,
    bool scale_up = true)
{
    cv::Size shape = image.size();
    float r = std::min(
        (float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);
    if (!scale_up)
    {
        r = std::min(r, 1.0f);
    }

    int newUnpad[2]{
        (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

    cv::Mat tmp;
    if (shape.width != newUnpad[0] || shape.height != newUnpad[1])
    {
        cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
    }
    else
    {
        tmp = image.clone();
    }

    float dw = new_shape.width - newUnpad[0];
    float dh = new_shape.height - newUnpad[1];

    if (!fixed_shape)
    {
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

void OnnxRuntime::Infer(int aWidth, int aHeight, int aChannel, unsigned char* aBytes, 
                        std::unique_ptr<float[]>& Boxes, std::unique_ptr<float[]>& ClassIndexs, 
                        std::unique_ptr<float[]>& Scores, std::unique_ptr<int[]>& BboxNum)
{
    cv::Mat img(aHeight, aWidth, CV_MAKETYPE(CV_8U, aChannel), aBytes);
    cv::Mat pr_img;
    float scale_x = aWidth / 640;
    float scale_y = aHeight / 640;
    float scale = letterbox(img, pr_img, {640, 640}, 32, {114, 114, 114}, true);
    cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB);
    float *blob = blobFromImage(pr_img);

    std::vector<Ort::Value> inputTensor, outputTensor; // Onnxruntime allowed input

    size_t input_tensor_size = 1 * 3 * 640 * 640;
    try
    {
        inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob, input_tensor_size, input_node_dims.data(), input_node_dims.size()));
    }
    catch (Ort::Exception oe)
    {
        std::cout << __LINE__ << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
    }

    try
    {
        outputTensor = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), inputTensor.data(), inputTensor.size(), output_node_names.data(), 1);
    }
    catch (Ort::Exception oe)
    {
        std::cout << __LINE__ << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
    }
    // Pushing the results
    if (outputTensor.size() > 0)
    {
        float *arr = outputTensor.front().GetTensorMutableData<float>();

        int img_w = img.cols;
        int img_h = img.rows;
        int x_offset = (640 * scale - img_w) / 2;
        int y_offset = (640 * scale - img_h) / 2;

        int n_dets = 0;

        for (int i = 0; i < 8400; i++) // dimCoords = 8400
        {
            float x = arr[i] * scale - x_offset; // data -> pointer to raw output
            float y = arr[(8400 * 1) + i] * scale - y_offset;
            float w = arr[(8400 * 2) + i];
            float h = arr[(8400 * 3) + i];

            float l = x - (0.5 * w) * scale;
            float t = y - (0.5 * h) * scale;
            float wid = w * scale;
            float hei = h * scale;

            std::vector<float> cls_confidences;
            for (int j = 0; j < 8; j++)
            {
                cls_confidences.push_back(arr[(8400 * (4 + j)) + i]);
            }

            float maxClsIdx = std::max_element(cls_confidences.begin(), cls_confidences.end()) - cls_confidences.begin();
            float maxClsConfidence = cls_confidences[maxClsIdx];

            if (maxClsConfidence < 0.5)
            {
                continue;
            }

            Boxes[n_dets * 4] = l;
            Boxes[n_dets * 4 + 1] = t;
            Boxes[n_dets * 4 + 2] = wid;
            Boxes[n_dets * 4 + 3] = hei;
            ClassIndexs[n_dets] = maxClsIdx;
            Scores[n_dets] = maxClsConfidence;
            n_dets++;
        }

        BboxNum[0] = n_dets;
    }
    inputTensor.clear();
    delete blob;
}

float *OnnxRuntime::blobFromImage(cv::Mat &img)
{
    float *blob = new float[img.total() * 3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < img_h; ++h)
        {
            for (int w = 0; w < img_w; ++w)
            {
                blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c] / 255.0;
            }
        }
    }
    return blob;
}