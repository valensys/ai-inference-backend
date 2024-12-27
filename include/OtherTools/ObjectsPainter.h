#pragma once

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <map>
#include <vector>
#include <nlohmann/json.hpp>
#include "STrack.h"
#include "globalvars.h"
#include "BYTETracker.h"

void PaintObjects(cv::Mat& frame, BYTETracker& tracker, std::map<int, vector<Point>>& tracks, 
                    const std::vector<STrack>& output_stracks, const std::map<int, int>& allFramesClassesCount, 
                    const std::map<int, int>& allFramesTrackAndNumber, const nlohmann::json& jayson);

void PaintObjects2(cv::Mat& frame, BYTETracker& tracker, std::map<int, vector<Point>>& tracks, 
                    const std::vector<STrack>& output_stracks, const std::map<int, int>& allFramesClassesCount, 
                    const std::map<int, int>& allFramesTrackAndNumber, const nlohmann::json& jayson);