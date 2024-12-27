#pragma once
#include "BYTETracker.h"
#include <vector>
#include <opencv2/opencv.hpp>

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

class OutputsDecoder
{
public:
    static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
    static inline float intersection_area(const Object& a, const Object& b);
    static void qsort_descent_inplace(vector<Object>& faceobjects, int left, int right);
    static void qsort_descent_inplace(vector<Object>& objects);
    static void nms_sorted_bboxes(const vector<Object>& faceobjects, vector<int>& picked);
    static void generate_yolox_proposals(vector<GridAndStride> grid_strides, float* feat_blob, vector<Object>& objects);
    static void decode_outputs(std::unique_ptr<float[]>& Boxes, std::unique_ptr<float[]>& ClassIndexs, 
                                    std::unique_ptr<int[]>& BboxNum, vector<Object> &objects, float scale);
    static constexpr double nmsThreshold = 0.7;
    static constexpr double bboxConfThreshold = 0.1;
    static vector<GridAndStride> grid_strides;
};