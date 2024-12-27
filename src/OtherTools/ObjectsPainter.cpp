#include "ObjectsPainter.h"

void PaintObjects(cv::Mat& frame, BYTETracker& tracker, std::map<int, vector<Point>>& tracks, 
                    const std::vector<STrack>& output_stracks, const std::map<int, int>& allFramesClassesCount, 
                    const std::map<int, int>& allFramesTrackAndNumber, const nlohmann::json& jayson)
{   
    for (size_t i = 0; i < output_stracks.size(); i++)
    {
        vector<float> tlwh = output_stracks[i].tlwh;
        cv::Rect rect = cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);

        Scalar color = globalvars::colorsOfObjects.at(static_cast<globalvars::ObjectsIndexes>(output_stracks[i].class_id));

        putText(frame, format("%d", allFramesTrackAndNumber.at(output_stracks[i].track_id)), Point(tlwh[0], tlwh[1] - 5),
                FONT_HERSHEY_PLAIN, 5, Scalar(0, 0, 255), 3, LINE_AA);

        rectangle(frame, rect, color, 6);
    }
}

void PaintObjects2(cv::Mat& frame, BYTETracker& tracker, std::map<int, vector<Point>>& tracks, 
                    const std::vector<STrack>& output_stracks, const std::map<int, int>& allFramesClassesCount, 
                    const std::map<int, int>& allFramesTrackAndNumber, const nlohmann::json& jayson)
{   
    for (size_t i = 0; i < output_stracks.size(); i++)
    {
        vector<float> tlwh = output_stracks[i].tlwh;
        cv::Rect rect = cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);

        Scalar color = Scalar(0, 255, 0);

        putText(frame, globalvars::classesNamesInOrder.at(output_stracks[i].class_id), Point(tlwh[0], tlwh[1] - 5),
                FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2, LINE_AA);

        rectangle(frame, rect, color, 6);
    }
}