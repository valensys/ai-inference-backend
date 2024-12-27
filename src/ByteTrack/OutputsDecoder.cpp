#include "OutputsDecoder.h"

vector<GridAndStride> OutputsDecoder::grid_strides;

void OutputsDecoder::generate_grids_and_stride(const int target_w, const int target_h, vector<int>& strides, vector<GridAndStride>& grid_s)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

float OutputsDecoder::intersection_area(const Object& a, const Object& b)
{
    Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void OutputsDecoder::qsort_descent_inplace(vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void OutputsDecoder::qsort_descent_inplace(vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void OutputsDecoder::nms_sorted_bboxes(const vector<Object>& faceobjects, vector<int>& picked)
{
    picked.clear();

    const int n = faceobjects.size();

    vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > OutputsDecoder::nmsThreshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void OutputsDecoder::generate_yolox_proposals(vector<GridAndStride> grid_strides, float* feat_blob, vector<Object>& objects)
{
    const int num_class = 1;

    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (num_class + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > OutputsDecoder::bboxConfThreshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

void OutputsDecoder::decode_outputs(std::unique_ptr<float[]>& Boxes, std::unique_ptr<float[]>& ClassIndexs, 
                                    std::unique_ptr<int[]>& BboxNum, vector<Object> &objects, float scale)
{
    vector<Object> proposals;
    for (int j = 0; j < BboxNum[0]; j++)
    {
        cv::Rect2f rect(Boxes[j * 4], Boxes[j * 4 + 1], Boxes[j * 4 + 2], Boxes[j * 4 + 3]);

        Object obj;
        obj.prob = 0.6;
        obj.rect = rect;
        obj.label = static_cast<int>(ClassIndexs[j]);
        proposals.push_back(obj);
    }

    qsort_descent_inplace(proposals);
    vector<int> picked;
    nms_sorted_bboxes(proposals, picked);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        // float x0 = (objects[i].rect.x) / scale;
        // float y0 = (objects[i].rect.y) / scale;
        // float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        // float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // float x0 = objects[i].rect.x;
        // float y0 = objects[i].rect.y;
        // float x1 = objects[i].rect.x + objects[i].rect.width;
        // float y1 = objects[i].rect.y + objects[i].rect.height;

        // objects[i].rect.x = x0;
        // objects[i].rect.y = y0;
        // objects[i].rect.width = x1 - x0;
        // objects[i].rect.height = y1 - y0;
    }
}
