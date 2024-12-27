#include "utilities.h"

using json = nlohmann::json;

float iou(const std::vector<float>& boxA, const std::vector<float>& boxB)
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

void utilities::nms(std::vector<std::vector<float>>& boxes, const float iou_threshold)
{
    // The format of boxes is [[top_left_x, top_left_y, bottom_right_x, bottom_right_y, score, class_id], ...]
    // Sorting "score + class_id" is to ensure that the boxes with the same class_id are grouped together and sorted by score
    std::sort(boxes.begin(), boxes.end(), [](const std::vector<float>& boxA, const std::vector<float>& boxB) { return boxA[4] + boxA[5] > boxB[4] + boxB[5];});
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (boxes[i][4] == 0.f)
        {
            continue;
        }
        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (boxes[i][5] != boxes[j][5])
            {
                break;
            }
            if (iou(boxes[i], boxes[j]) > iou_threshold)
            {
                boxes[j][4] = 0.f;
            }
        }
    }
    std::erase_if(boxes, [](const std::vector<float>& box) { return box[4] == 0.f; });
}

float* utilities::blobFromImage(cv::Mat& img) {
  float* blob = new float[img.total() * 3];
  int channels = 3;
  int img_h = img.rows;
  int img_w = img.cols;
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < img_h; h++) {
      for (int w = 0; w < img_w; w++) {
        blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c] / 255.0;
      }
    }
  }
  return blob;
}

void utilities::visualize(cv::Mat &img, float *Boxes, float *ClassIndexs, int *BboxNum)
{
  for (int j = 0; j < BboxNum[0]; ++j)
  {
    cv::Rect rect(Boxes[j * 4], Boxes[j * 4 + 1], Boxes[j * 4 + 2], Boxes[j * 4 + 3]);

    cv::rectangle(img, rect, cv::Scalar(0x27, 0xC1, 0x36), 3);
    cv::putText(
        img,
        std::to_string(static_cast<int>(ClassIndexs[j])),
        cv::Point(rect.x, rect.y - 1),
        cv::FONT_HERSHEY_PLAIN,
        2,
        cv::Scalar(0xFF, 0xFF, 0xFF),
        2);
  }
}

void utilities::fixBoxesForBytetrack(std::unique_ptr<float[]>& Boxes, std::unique_ptr<float[]>& ClassIndexs, 
                                std::unique_ptr<float[]>& Scores, std::unique_ptr<int[]>& BboxNum)
{
  int class_indexs_size = 1000;

  std::map<std::string, std::vector<std::string>> classesAndRects;
  std::map<std::string, std::vector<std::vector<float>>> classesAndFloatsRects;

  const std::vector<std::string> classesNamesInOrder = {"bus", "car", "concrete_mixer", "human", "microbus",
                                                        "special_vehicle", "truck", "truck_crane"};

  for (int j = 0; j < BboxNum[0]; ++j)
  {
    classesAndFloatsRects[classesNamesInOrder.at(ClassIndexs[j])].push_back(std::vector<float>{Boxes[j * 4],
                                                                                               Boxes[j * 4 + 1],
                                                                                               Boxes[j * 4] + Boxes[j * 4 + 2],
                                                                                               Boxes[j * 4 + 1] + Boxes[j * 4 + 3],
                                                                                               Scores[j],
                                                                                               ClassIndexs[j]});
  }

  for (auto &element : classesAndFloatsRects)
  {
    utilities::nms(element.second);
  }

  int items_count = 0;

  for (const auto &element : classesAndFloatsRects)
  {
    for (const auto &rect : element.second)
    {
      ClassIndexs[items_count] = static_cast<float>(globalvars::objectNameToIndex.at(element.first));
      Boxes[items_count * 4] = rect.at(0);
      Boxes[items_count * 4 + 1] = rect.at(1);
      Boxes[items_count * 4 + 2] = rect.at(2) - rect.at(0);
      Boxes[items_count * 4 + 3] = rect.at(3) - rect.at(1);

      items_count++;
    }
  }

  BboxNum[0] = items_count;

  for (int last_item = items_count; last_item < class_indexs_size; last_item++)
  {
    ClassIndexs[last_item] = 0;
    Boxes[last_item * 4] = 0;
    Boxes[last_item * 4 + 1] = 0;
    Boxes[last_item * 4 + 2] = 0;
    Boxes[last_item * 4 + 3] = 0;
  }
}

json utilities::getTrackedDataInJson(std::unique_ptr<float[]>& Boxes, std::unique_ptr<float[]>& ClassIndexs, 
                                      std::unique_ptr<float[]>& Scores, std::unique_ptr<int[]>& BboxNum)
{
  json jayson;
  std::map<std::string, std::vector<std::string>> classesAndRects;
  std::map<std::string, std::vector<std::vector<float>>> classesAndFloatsRects;

  const std::vector<std::string> classesNamesInOrder = {"bus", "car", "concrete_mixer", "human", "microbus", 
                                                        "special_vehicle", "truck", "truck_crane"};

  for (int j = 0; j < BboxNum[0]; ++j)
  {
    classesAndFloatsRects[classesNamesInOrder.at(ClassIndexs[j])].push_back(std::vector<float> {Boxes[j * 4], 
                                                                                                Boxes[j * 4 + 1], 
                                                                                                Boxes[j * 4] + Boxes[j * 4 + 2], 
                                                                                                Boxes[j * 4 + 1] + Boxes[j * 4 + 3],
                                                                                                Scores[j],
                                                                                                ClassIndexs[j]});
  }


  for (auto &element : classesAndFloatsRects)
  {
    utilities::nms(element.second);

    for (auto &box : element.second)
    {
      box.pop_back();
      box.pop_back();
    }

    for (auto &rect : element.second)
    {
      rect.at(2) = rect.at(2) - rect.at(0);
      rect.at(3) = rect.at(3) - rect.at(1);
    }
  }

  for (const auto &element : classesAndFloatsRects)
  {
    jayson[element.first] = element.second;
  }

  return jayson;
}