#pragma once
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <map>

namespace globalvars
{
    enum class ObjectsIndexes
    {
        BUS = 0,
        CAR = 1,
        CONCRETE_MIXER = 2,
        HUMAN = 3,
        MICROBUS = 4,
        SPECIAL_VEHICLE = 5,
        TRUCK = 6,
        TRUCK_CRANE = 7
    };

    const cv::Scalar color_red{0, 0, 255};
    const cv::Scalar color_orange{0, 165, 255};
    const cv::Scalar color_yellow{0, 255, 255};
    const cv::Scalar color_green{0, 128, 0};
    const cv::Scalar color_blue{255, 0, 0};
    const cv::Scalar color_cyan{255, 255, 0};
    const cv::Scalar color_violet{238, 130, 238};
    const cv::Scalar color_lime{54, 243, 137};

    const std::map<std::string, ObjectsIndexes> objectNameToIndex
    {
        {"bus", ObjectsIndexes::BUS},
        {"car", ObjectsIndexes::CAR},
        {"concrete_mixer", ObjectsIndexes::CONCRETE_MIXER},
        {"human", ObjectsIndexes::HUMAN},
        {"microbus", ObjectsIndexes::MICROBUS},
        {"special_vehicle", ObjectsIndexes::SPECIAL_VEHICLE},
        {"truck", ObjectsIndexes::TRUCK},
        {"truck_crane", ObjectsIndexes::TRUCK_CRANE}
    };

    const std::map<ObjectsIndexes, cv::Scalar> colorsOfObjects
    {
        {ObjectsIndexes::BUS, color_red},
        {ObjectsIndexes::CAR, color_orange},
        {ObjectsIndexes::CONCRETE_MIXER, color_yellow},
        {ObjectsIndexes::HUMAN, color_green},
        {ObjectsIndexes::MICROBUS, color_blue},
        {ObjectsIndexes::SPECIAL_VEHICLE, color_cyan},
        {ObjectsIndexes::TRUCK, color_violet},
        {ObjectsIndexes::TRUCK_CRANE, color_lime}
    };

    const std::vector<std::string> classesNamesInOrder = {"bus", "car", "concrete_mixer", "human", "microbus",
                                                        "special_vehicle", "truck", "truck_crane"};
}
