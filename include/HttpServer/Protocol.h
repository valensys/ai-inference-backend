#pragma once
#include <string>
#include <map>

enum class VideoStatus
{
	QUEUED,
	PROCESSING,
	READY,
	NOT_FOUND,
	FAILED,
	TOO_BIG
};

enum class BufferRequestType
{
    START_DEMO,
    VIDEO_UPLOADING,
    STREAM_URI,
    START_PROCESSING,
    STOP_PROCESSING,
    IMAGE_UPLOADING,
    GET_REPORT,
    NONE
};

enum class BufferResponceType
{
    VIDEO_UPLOADING_OK,
    PROCESSING_STOP_OK,
    SEND_IMAGE,
    SEND_JSON,
    NONE
};

const std::map<BufferRequestType, std::string> requestTypeMap
{
    {BufferRequestType::START_DEMO, "<DATA TYPE: DEMO>"},
    {BufferRequestType::VIDEO_UPLOADING, "<DATA TYPE: VIDEO>"},
    {BufferRequestType::STREAM_URI, "<DATA TYPE: STREAM>"},
    {BufferRequestType::IMAGE_UPLOADING, "<DATA TYPE: IMAGE>"},
    {BufferRequestType::STOP_PROCESSING, "<DATA TYPE: STOP>"},
    {BufferRequestType::START_PROCESSING, "<DATA TYPE: START>"},
    {BufferRequestType::GET_REPORT, "<DATA TYPE: REPORT>"}
};

const std::map<BufferResponceType, std::string> responceTypeMap
{
    {BufferResponceType::VIDEO_UPLOADING_OK, "<DATA TYPE: VIDEO OK>"},
    {BufferResponceType::PROCESSING_STOP_OK, "<DATA TYPE: STOP OK>"},
    {BufferResponceType::SEND_IMAGE, "<DATA TYPE: IMAGE OK>"},
    {BufferResponceType::SEND_JSON, "<DATA TYPE: JSON OK>"}
};