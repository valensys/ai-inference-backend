#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <nadjieb/mjpeg_streamer.hpp>
#include <filesystem>

#include "WebSocketServer.h"
#include "Poco/Net/NetException.h"
#include "Poco/Util/HelpFormatter.h"
#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPClientSession.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPMessage.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/ServerSocket.h"

#include "STrack.h"
#include "utilities.h"
#include "OutputsDecoder.h"
#include "OnnxRuntime.h"
#include "ObjectsPainter.h"
#include "ConfigReader.h"

using Poco::ThreadPool;
using Poco::Timestamp;
using Poco::Net::HTTPRequest;
using Poco::Net::HTTPRequestHandler;
using Poco::Net::HTTPRequestHandlerFactory;
using Poco::Net::HTTPResponse;
using Poco::Net::HTTPServer;
using Poco::Net::WebSocketException;
using Poco::Net::HTTPServerParams;
using Poco::Net::HTTPServerRequest;
using Poco::Net::HTTPServerResponse;
using Poco::Net::HTTPClientSession;
using Poco::Net::HTTPMessage;
using Poco::Net::ServerSocket;
using Poco::Net::WebSocket;
using Poco::Util::Application;
using Poco::Util::HelpFormatter;
using Poco::Util::Option;
using Poco::Util::OptionSet;
using Poco::Util::ServerApplication;
using MJPEGStreamer = nadjieb::MJPEGStreamer;

string logsPath = "../logs";
string jwtToken;
string receivedVideoName;
Configuration configuration;

volatile BufferRequestType lastRequest = BufferRequestType::NONE;

int bufferSize = 104857600 * 3;
char *buffer = new char[bufferSize];
int flags;

std::vector<std::string> videosForProcessing;
std::map<std::string, VideoStatus> queueForProcessing;

void sendDataToDb(std::string videoName, VideoStatus videoStatus)
{
	std::string requiredData;
	std::string gisBackHost = configuration.gisBackHost;
	std::string gisBackPort = configuration.gisBackPort;
	std::string gisDbPath = configuration.gisDbPath;
	std::string gisBackHostAndPort = gisBackHost + ":" + gisBackPort;
	std::string gisDbFullPath = gisBackHostAndPort + configuration.gisBackPort;

	std::cout << "gisDbFullPath: " << gisDbFullPath << std::endl;

	nlohmann::json reqData;
	reqData["file_name"] = videoName;
	reqData["file_status"] = static_cast<int>(videoStatus);

	std::cout << "DbReqData: " << reqData << std::endl;

	if (jwtToken.size() > 0)
	{
		HTTPClientSession session(gisBackHost, atoi(gisBackPort.c_str()));
		HTTPRequest request(HTTPRequest::HTTP_POST, gisDbPath, HTTPMessage::HTTP_1_1);

		std::stringstream ss;
		ss << reqData;
		request.setContentType("application/json");
		request.setContentLength(ss.str().size());
		request.add("Authorization", "Token " + jwtToken);
		std::ostream &myOStream = session.sendRequest(request);
		myOStream << ss.str();

		HTTPResponse response;
		session.receiveResponse(response);

		if (response.getStatus() != HTTPResponse::HTTP_OK)
		{
			std::cout << "ERROR: cannot send data to db. Responce status is: " << response.getStatus() << std::endl;
		}
		else
		{
			std::cout << "Data sent to db successfully. Responce status is: " << response.getStatus() << std::endl;
		}
	}
	else
	{
		std::cout << "ERROR: jwt token is empty. Can't send data to db" << std::endl;
	}
}

nlohmann::json queueToJson() 
{ 
	nlohmann::json js;
	for (const auto &video : queueForProcessing)
	{
		js[video.first] = video.second;
	}
	return js;
}

void jsonToQyeue(nlohmann::json js)
{
	for (const auto &item : js.items())
	{
		queueForProcessing[item.key()] = (VideoStatus)item.value();
	}

	for (const auto &video : queueForProcessing)
	{
		std::cout << "Deserialized queue: " << std::endl;
		std::cout << video.first << " " << static_cast<int>(video.second) << std::endl;
	}
}

void serializeQueue()
{
	ofstream file;
	file.open(logsPath + "/queue.txt");
	if (file.is_open() == true)
	{
		nlohmann::json jsonQueue = queueToJson();
		file << jsonQueue.dump(4);
		file.close();
	}
	else
	{
		std::cout << "Failed to open file for serialization" << std::endl;
	}
}

void deserializeQueue()
{
	ifstream file;
	file.open(logsPath + "/queue.txt");
	if (file.is_open() == true)
	{
		nlohmann::json jsonQueue = nlohmann::json::parse(file);
		std::cout << "Deserialized queue: " << jsonQueue << std::endl;
		jsonToQyeue(jsonQueue);
		file.close();
	}
	else
	{
		std::cout << "Failed to open file for deserialization" << std::endl;
	}
}

void PageRequestHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response)
{
	if (request.getMethod() == "POST")
	{
		std::cout << "POST request" << std::endl;
		std::cout << "Content Length: " << request.getContentLength64() << std::endl;
		std::cout << "Content: " << request.getContentType() << std::endl;
		std::cout << "Request address: " << request.clientAddress().toString() << std::endl;

		nlohmann::json jsonObj = nlohmann::json::parse(request.stream());
		std::cout << "JSON: " << jsonObj << std::endl;

		response.setContentType("application/json");
		response.setStatus(Poco::Net::HTTPResponse::HTTPStatus::HTTP_OK);

		if (jsonObj.empty())
		{
			std::cout << "JSON is empty" << std::endl;
			nlohmann::json responseJson;
			responseJson["error"] = "No JSON data received";
			std::ostream& ostr = response.send();
			ostr << responseJson;
		}

		if (jsonObj.contains("video_path"))
		{
			std::cout << "Video path: " << jsonObj["video_path"] << std::endl;
			std::string videoPath = jsonObj["video_path"];
			if (videoPath[0] == '.')
			{
				videoPath.erase(0,1);
			}
			std::string trueVideoPath = configuration.storageDirLocal + "/" + videoPath;

			std::string videoPathx = trueVideoPath.substr(0, trueVideoPath.find_last_of('/'));
			std::string videoName = trueVideoPath.substr(trueVideoPath.find_last_of('/') + 1);
			std::string processedVideoNameNoExtension = videoName.substr(0, videoName.find_last_of('.'));
			std::string processedVideoName = "processed_" + processedVideoNameNoExtension + ".mp4";
			std::string totalVideoName = videoPathx + "/" + processedVideoName;
			std::string jsonProcessedVideoName = videoPathx + "/" + "processed_" + processedVideoNameNoExtension + ".json";

			if (queueForProcessing.find(trueVideoPath) == queueForProcessing.end())
			{
				queueForProcessing[trueVideoPath] = VideoStatus::QUEUED;
				sendDataToDb(videoName, VideoStatus::QUEUED);
			}

			if (jsonObj.contains("reprocessing"))
			{
				std::cout << "Reprocessing: " << jsonObj["reprocessing"] << std::endl;
				queueForProcessing[trueVideoPath] = VideoStatus::QUEUED;
				sendDataToDb(videoName, VideoStatus::QUEUED);
			}

			std::cout << "Current queue is: " << std::endl;
			for (const auto &queue : queueForProcessing)
			{
				std::cout << queue.first << " " << static_cast<int>(queue.second) << std::endl;
			}

			nlohmann::json responseJson;
			responseJson["video_path"] = trueVideoPath;
			responseJson["status"] = queueForProcessing[trueVideoPath];
			std::ostream& ostr = response.send();
			ostr << responseJson;
		}
		else if(jsonObj.contains("required"))
		{
			if (jsonObj["required"] == "history")
			{
				nlohmann::json responseJson;
				responseJson["history"] = nlohmann::json(queueForProcessing);
				std::ostream& ostr = response.send();
				ostr << responseJson;
			}
		}
	}
}

BufferRequestType WebSocketRequestHandler::annalyzeRequest(const char *buffer, int nBytesReceived)
{
	BufferRequestType currentRequest = BufferRequestType::NONE;

	if (nBytesReceived > 0)
	{
		for (const auto &keyval : requestTypeMap)
		{
			std::string currentHeader(buffer, buffer + keyval.second.size());
			if (currentHeader == keyval.second)
			{
				std::cout << "Current header is: " << currentHeader << std::endl;
				currentRequest = keyval.first;
				return currentRequest;
			}
		}

		return currentRequest;
	}
	else
	{
		std::cout << "Wrong data package" << std::endl;
		return BufferRequestType::NONE;
	}
}

void WebSocketRequestHandler::demoStreamProcessor(WebSocket& ws)
{
	Application &app = Application::instance();

	cv::VideoCapture vc = cv::VideoCapture("./media/demo.mpeg");

	MJPEGStreamer streamer;
    streamer.start(5200, 4);

	cv::Mat frame;

	while (vc.read(frame))
	{
		std::vector<uchar> buff_bgr;
		cv::imencode(".jpg", frame, buff_bgr, {cv::IMWRITE_JPEG_QUALITY, 60});
        streamer.publish("/demo_stream", std::string(buff_bgr.begin(), buff_bgr.end()));
	}

	app.logger().information("Video demo is over");
}

void WebSocketRequestHandler::videoProcessor(std::string filename, WebSocket &ws)
{
	Application &app = Application::instance();

	cv::VideoCapture vc = cv::VideoCapture(filename);

	nlohmann::json js;
	js["frame_width"] = vc.get(cv::CAP_PROP_FRAME_WIDTH);
	js["frame_height"] = vc.get(cv::CAP_PROP_FRAME_HEIGHT);

	std::string jsString = to_string(js);
	ws.sendFrame(jsString.data(), jsString.size(), WebSocket::SendFlags::FRAME_TEXT);

	cv::Mat frame;

	auto Boxes = std::make_unique<float[]>(4000);
	auto BboxNum = std::make_unique<int[]>(1);
	auto ClassIndexs = std::make_unique<float[]>(1000);
	auto Scores = std::make_unique<float[]>(1000);

	OnnxRuntime onnxRuntime = OnnxRuntime(const_cast<char *>(configuration.modelPath.c_str()));

	BYTETracker tracker(30, 30);
    std::map<int, vector<Point>> tracks;

	std::map<int, int> allFramesClassesCount;
	std::map<int, int> allFramesTrackAndNumber;

	while (vc.read(frame))
	{
		if (lastRequest == BufferRequestType::STOP_PROCESSING)
		{
			lastRequest = BufferRequestType::NONE;
			break;
		}

		onnxRuntime.Infer(frame.cols, frame.rows, frame.channels(), frame.data, Boxes, ClassIndexs, Scores, BboxNum);
		utilities::fixBoxesForBytetrack(Boxes, ClassIndexs, Scores, BboxNum);

		vector<Object> objects;
		OutputsDecoder::decode_outputs(Boxes, ClassIndexs, BboxNum, objects, 1);
		vector<STrack> output_stracks = tracker.update(objects);

		for (size_t i = 0; i < output_stracks.size(); i++)
		{
			if (allFramesTrackAndNumber.find(output_stracks.at(i).track_id) == allFramesTrackAndNumber.end()) 
			{
				allFramesClassesCount[output_stracks.at(i).class_id]++;
				allFramesTrackAndNumber[output_stracks.at(i).track_id] = allFramesClassesCount.at(output_stracks.at(i).class_id);
			}
		}

		nlohmann::json jayson = utilities::getTrackedDataInJson(Boxes, ClassIndexs, Scores, BboxNum);

		PaintObjects2(frame, tracker, tracks, output_stracks, allFramesClassesCount, allFramesTrackAndNumber, jayson);

		std::vector<uchar> buff_bgr;
		cv::imencode(".jpg", frame, buff_bgr, {cv::IMWRITE_JPEG_QUALITY, 60});

		std::string imageAsString = std::string(buff_bgr.begin(), buff_bgr.end());
		int bytesSent = ws.sendFrame(imageAsString.c_str(), imageAsString.size(), WebSocket::SendFlags::FRAME_BINARY);

		std::string jsonString = to_string(jayson);
		int bytesSentJayson = ws.sendFrame(jsonString.data(), jsonString.size(), WebSocket::SendFlags::FRAME_TEXT);

		app.logger().information("Bytes sent = %d", bytesSent + bytesSentJayson);
	}

	app.logger().information("Video processing is over");
}

void WebSocketRequestHandler::processPackage(WebSocket& ws)
{
	Application &app = Application::instance();
	try
	{
		int n = 0;
		do
		{
			if (lastRequest == BufferRequestType::NONE)
			{
				try
				{
					n = ws.receiveFrame(buffer, bufferSize, flags);
				}
				catch(const Poco::TimeoutException& e)
				{
					std::cout << "TimeoutException occured" << std::endl;
				}
				
				app.logger().information(Poco::format("Package received (length=%d, flags=0x%x).", n, unsigned(flags)));
				lastRequest = annalyzeRequest(buffer, n);
			}

		} while (n > 0 && (flags & WebSocket::FRAME_OP_BITMASK) != WebSocket::FRAME_OP_CLOSE);
		
		delete[] buffer;
		app.logger().information("WebSocket connection closed.");
	}
	catch (WebSocketException &exc)
	{
		app.logger().log(exc);
	}
}

void WebSocketRequestHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response)
{
	Application &app = Application::instance();
	try
	{
		WebSocket ws(request, response);
		app.logger().information("WebSocket connection established.");

		std::thread(&WebSocketRequestHandler::processPackage, this, std::ref(ws)).detach();

		int n;
		while (true)
		{
			if (lastRequest == BufferRequestType::VIDEO_UPLOADING)
			{
				std::cout << "VIDEO UPLOADING" << std::endl;

				n = ws.receiveFrame(buffer, bufferSize, flags);
				app.logger().information(Poco::format("Video received (length=%d, flags=0x%x).", n, unsigned(flags)));

				FILE *stream = fopen("../media/uploaded_video.mp4", "w");
				receivedVideoName = "../media/uploaded_video.mp4";
				fwrite(buffer, sizeof(char), n, stream);
				fclose(stream);

				lastRequest = BufferRequestType::NONE;
				videoProcessor(receivedVideoName, ws);
			}

			if (lastRequest == BufferRequestType::STREAM_URI)
			{
				std::cout << "STREAM PROCESSING" << std::endl;

				n = ws.receiveFrame(buffer, bufferSize, flags);
				app.logger().information(Poco::format("Stream URI received (length=%d, flags=0x%x).", n, unsigned(flags)));

				std::string streamUri = std::string(buffer, buffer + n);
				app.logger().information(Poco::format("Stream URI = %s", streamUri));

				lastRequest = BufferRequestType::NONE;
				videoProcessor(streamUri, ws);
			}

			if (lastRequest == BufferRequestType::START_DEMO)
			{
				std::cout << "START DEMO" << std::endl;
				lastRequest = BufferRequestType::NONE;
				videoProcessor(configuration.demoPath, ws);
			}
		}
		app.logger().information("WebSocket connection closed.");
	}
	catch (WebSocketException &exc)
	{
		app.logger().log(exc);
		switch (exc.code())
		{
		case WebSocket::WS_ERR_HANDSHAKE_UNSUPPORTED_VERSION:
			response.set("Sec-WebSocket-Version", WebSocket::WEBSOCKET_VERSION);
		case WebSocket::WS_ERR_NO_HANDSHAKE:
		case WebSocket::WS_ERR_HANDSHAKE_NO_VERSION:
		case WebSocket::WS_ERR_HANDSHAKE_NO_KEY:
			response.setStatusAndReason(HTTPResponse::HTTP_BAD_REQUEST);
			response.setContentLength(0);
			response.send();
			break;
		}
	}
	catch (Poco::TimeoutException &exc)
	{
		app.logger().log(exc);
		std::cout << "Timeout Exception" << std::endl;
	}
}

HTTPRequestHandler *RequestHandlerFactory::createRequestHandler(const HTTPServerRequest &request)
{
	Application &app = Application::instance();
	app.logger().information("Request from " + request.clientAddress().toString() + ": " + request.getMethod() + " " + request.getURI() + " " + request.getVersion());



	for (HTTPServerRequest::ConstIterator it = request.begin(); it != request.end(); ++it)
	{
		app.logger().information(it->first + ": " + it->second);
	}

	if (request.find("Upgrade") != request.end() && Poco::icompare(request["Upgrade"], "websocket") == 0)
		return new WebSocketRequestHandler;
	else
		return new PageRequestHandler;
}

void WebSocketServer::initialize(Application &self)
{
	loadConfiguration(); // load default configuration files, if present
	ServerApplication::initialize(self);
}

void WebSocketServer::uninitialize()
{
	ServerApplication::uninitialize();
}

void WebSocketServer::defineOptions(OptionSet &options)
{
	ServerApplication::defineOptions(options);

	options.addOption(
		Option("help", "h", "display help information on command line arguments")
			.required(false)
			.repeatable(false));
}

void WebSocketServer::handleOption(const std::string &name, const std::string &value)
{
	ServerApplication::handleOption(name, value);

	if (name == "help")
		_helpRequested = true;
}

void WebSocketServer::displayHelp()
{
	HelpFormatter helpFormatter(options());
	helpFormatter.setCommand(commandName());
	helpFormatter.setUsage("OPTIONS");
	helpFormatter.setHeader("A sample HTTP server supporting the WebSocket protocol.");
	helpFormatter.format(std::cout);
}

std::string WebSocketServer::getJWTToken(const Configuration& configuration)
{
	std::string requiredData;
	
	std::string gisBackHost = configuration.gisBackHost;
	std::string gisBackPort = configuration.gisBackPort;
	std::string gisAuthPath = configuration.gisAuthPath;
	std::string gisBackHostAndPort = gisBackHost + ":" + gisBackPort;
	std::string gisAuthFullPath = gisBackHostAndPort + configuration.gisAuthPath;

	std::cout << "gisAuthFullPath: " << gisAuthFullPath << std::endl;

	nlohmann::json reqData;
	reqData["user_name"] = configuration.gisBackUsername;
	reqData["password"] = configuration.gisBackPassword;
	reqData["id_module"] = 50;

	std::cout << "jwtReqData: " << reqData << std::endl;

	try
	{
		HTTPClientSession session(gisBackHost, atoi(gisBackPort.c_str()));
		HTTPRequest request(HTTPRequest::HTTP_POST, gisAuthPath, HTTPMessage::HTTP_1_1);

		std::stringstream ss;
		ss << reqData;
		request.setKeepAlive(true);
		request.setContentType("application/json");
		request.setContentLength(ss.str().size());

		std::ostream &myOStream = session.sendRequest(request);
		myOStream << ss.str();

		HTTPResponse response;
		std::istream &rs = session.receiveResponse(response);

		if (response.getStatus() == HTTPResponse::HTTP_UNAUTHORIZED)
		{
			std::cout << "ERROR: gis back status login/pw is wrong - " << response.getStatus() << std::endl;
			return string();
		}
		else
		{
			std::stringstream ss2;
			ss2 << rs.rdbuf();
			nlohmann::json responseJson = nlohmann::json::parse(ss2);

			std::cout << "Responce: \n"
					  << responseJson << std::endl;

			requiredData = responseJson["token"];

			std::cout << "JWT token: \n"
					  << requiredData << std::endl;

			return requiredData;
		}
	}
	catch (const Poco::Exception &e)
	{
		std::cerr << "Can't establish connection: " << configuration.gisBackHost << ":" << configuration.gisBackPort << " "<< e.what() << '\n';
		return string();
	}
}

std::string WebSocketServer::readConfig(std::string configPath)
{
	ConfigReader cfgReader = ConfigReader(configPath);
	configuration = cfgReader.readConfig();

	return configuration.host + ":" + configuration.port;
}

void eunEternalProcessing()
{
	std::cout << "Processing thread is running." << std::endl;

	while (true)
	{	
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		for (auto &video : queueForProcessing)
		{	
			if (video.second == VideoStatus::QUEUED)
			{
				std::string videoPath = video.first.substr(0, video.first.find_last_of('/'));
				std::string videoName = video.first.substr(video.first.find_last_of('/') + 1);
				std::string processedVideoNameNoExtension = videoName.substr(0, videoName.find_last_of('.'));
				std::string processedVideoName = "processed_" + processedVideoNameNoExtension + ".mp4";
				std::string totalVideoName = videoPath + "/" + processedVideoName;

				std::string jsonProcessedVideoName = videoPath + "/" + "processed_" + processedVideoNameNoExtension + ".json";

				if (std::filesystem::exists(totalVideoName))
				{
					if (std::filesystem::exists(jsonProcessedVideoName))
					{
						std::cout << "Video already processed. File exists." << std::endl;
						video.second = VideoStatus::READY;
						sendDataToDb(video.first, VideoStatus::READY);
					}
					else
					{
						std::cout << "Video is not processed yet. Json file not found, but video is." << std::endl;
						std::cout << "Check if video is in progress..." << std::endl;

						auto time_pt1 = std::chrono::system_clock::now();
						auto time_pt2 = time_pt1;
						int sec_to_scan = 10;
						uint64_t file_size = std::filesystem::file_size(totalVideoName);
						bool is_in_progress = false;

						while (std::chrono::duration_cast<std::chrono::seconds>(time_pt2 - time_pt1).count() < sec_to_scan)
						{
							uint64_t current_file_size = std::filesystem::file_size(totalVideoName);
							std::cout << "Checking.. got file size: " << current_file_size << std::endl;
							if (current_file_size > file_size)
							{
								std::cout << "Video size changed. Video is in progress." << std::endl;
								is_in_progress = true;
								break;
							}
							std::this_thread::sleep_for(std::chrono::milliseconds(1000));
							time_pt2 = std::chrono::system_clock::now();
						}

						if (is_in_progress)
						{
							video.second = VideoStatus::PROCESSING;
							sendDataToDb(videoName, VideoStatus::PROCESSING);
							continue;
						}
					}
				}

				if (std::filesystem::exists(video.first))
				{
					auto videoSize = std::filesystem::file_size(video.first);
					if (videoSize > configuration.maxVideoSize)
					{
						std::cout << "Video size = " << videoSize << " is too big." << std::endl;
						std::cout << "Max video size = " << configuration.maxVideoSize << std::endl;
					}
				}

				cv::VideoCapture vc = cv::VideoCapture(video.first);

				if (!vc.isOpened())
				{
					video.second = VideoStatus::NOT_FOUND;
					sendDataToDb(video.first, VideoStatus::NOT_FOUND);
					std::cout << video.first << " Video can't be opened." << std::endl;
					continue;
				}

				video.second = VideoStatus::PROCESSING;
				sendDataToDb(video.first, VideoStatus::PROCESSING);
				std::cout << "Processing: " << video.first << std::endl;
				double fps = vc.get(cv::CAP_PROP_FPS);

				VideoWriter vw = VideoWriter(totalVideoName, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps,
											 cv::Size(vc.get(cv::CAP_PROP_FRAME_WIDTH), vc.get(cv::CAP_PROP_FRAME_HEIGHT)),
											 true);

				auto Boxes = std::make_unique<float[]>(4000);
				auto BboxNum = std::make_unique<int[]>(1);
				auto ClassIndexs = std::make_unique<float[]>(1000);
				auto Scores = std::make_unique<float[]>(1000);

				OnnxRuntime onnxRuntime = OnnxRuntime(const_cast<char *>(configuration.modelPath.c_str()));

				BYTETracker tracker(30, 30);
				std::map<int, vector<Point>> tracks;

				std::map<int, int> allFramesClassesCount;
				std::map<int, int> allFramesTrackAndNumber;

				int framenum = 0;
				cv::Mat frame;

				std::vector<nlohmann::json> allJsons;
				while (vc.read(frame))
				{
					if (video.second == VideoStatus::QUEUED)
					{
						break;
					}


					framenum++;
					if (framenum % 10 == 0)
					{
						std::cout << video.first << " Processing frame: " << framenum << std::endl;
					}
					onnxRuntime.Infer(frame.cols, frame.rows, frame.channels(), frame.data, Boxes, ClassIndexs, Scores, BboxNum);
					utilities::fixBoxesForBytetrack(Boxes, ClassIndexs, Scores, BboxNum);

					vector<Object> objects;
					OutputsDecoder::decode_outputs(Boxes, ClassIndexs, BboxNum, objects, 1);
					vector<STrack> output_stracks = tracker.update(objects);

					for (size_t i = 0; i < output_stracks.size(); i++)
					{
						if (allFramesTrackAndNumber.find(output_stracks.at(i).track_id) == allFramesTrackAndNumber.end())
						{
							allFramesClassesCount[output_stracks.at(i).class_id]++;
							allFramesTrackAndNumber[output_stracks.at(i).track_id] = allFramesClassesCount.at(output_stracks.at(i).class_id);
						}
					}

					nlohmann::json jayson = utilities::getTrackedDataInJson(Boxes, ClassIndexs, Scores, BboxNum);
					PaintObjects2(frame, tracker, tracks, output_stracks, allFramesClassesCount, allFramesTrackAndNumber, jayson);

					nlohmann::json currentTracking;
					for (auto& clsname : globalvars::classesNamesInOrder)
					{
						currentTracking[clsname] = jayson[clsname].size();
					}

					nlohmann::json trackingStat;
					for (auto &clsname : globalvars::classesNamesInOrder)
					{
						trackingStat[clsname] = allFramesClassesCount[static_cast<int>(globalvars::objectNameToIndex.at(clsname))];
					}

					nlohmann::json currentJson;
					currentJson["npp"] = framenum;
					currentJson["count_objects"] = BboxNum[0];
					currentJson["start_time"] = 1.0 / fps * framenum;
					currentJson["end_time"] = 1.0 / fps * framenum;
					currentJson["objects"] = currentTracking;
					currentJson["tracking_stat"] = trackingStat;
					
					allJsons.push_back(currentJson);

					vw.write(frame);
				}

				vw.~VideoWriter();

				std::ofstream out(jsonProcessedVideoName);
				if (out.is_open())
				{
					out << std::setw(4) << allJsons << std::endl;
					out.close();
					std::cout << video.first << " Json log is saved" << std::endl;
				}
				else
				{
					std::cout << "Json log can't be saved: " << jsonProcessedVideoName << std::endl;
				}

				if (video.second == VideoStatus::PROCESSING)
				{
					video.second = VideoStatus::READY;
					sendDataToDb(video.first, VideoStatus::READY);
					std::cout << video.first << " Video processing is over" << std::endl;
				}
				if (video.second == VideoStatus::QUEUED)
				{
					std::cout << video.first << " Video processing is intentionally interrupted" << std::endl;
				}
			}
		}
	}
}

int WebSocketServer::main(const std::vector<std::string> &args)
{
	if (_helpRequested)
	{
		displayHelp();
	}
	else
	{
		std::cout << "WebSocketServer init begins..." << std::endl;

		std::string hostAndPort = readConfig("../configs/main.cfg");
		jwtToken = getJWTToken(configuration);
		ServerSocket svs(Poco::Net::SocketAddress(hostAndPort.c_str()));
		HTTPServer srv(new RequestHandlerFactory, svs, new HTTPServerParams);

		std::cout << "WebSocketSever init ends..." << std::endl;

		thread t(eunEternalProcessing);
		t.detach();
		srv.start();

		waitForTerminationRequest();
		srv.stop();
	}
	return Application::EXIT_OK;
}