#include "handlers/PostRequestHandler.h"

#include <Poco/Net/HTTPServerResponse.h>
#include <Poco/Net/HTTPServerRequest.h>
#include <Poco/StreamCopier.h>

#include <sstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>

namespace handlers
{
	std::string camSource;
	bool needTracking;
	cv::Size currentCapSize;
	std::vector<cv::Point> restrictedAreaPts;

	using json = nlohmann::json;

	void PostRequestHandler::handleRequest(
		Poco::Net::HTTPServerRequest &request,
		Poco::Net::HTTPServerResponse &response)
	{
		std::string recv_string;
		Poco::StreamCopier::copyToString(request.stream(), recv_string);
		std::cout << recv_string << std::endl;

		json jsonObj;
		try
		{
			std::stringstream(recv_string) >> jsonObj;
		}
		catch (const std::exception &e)
		{
			std::cerr << e.what() << '\n';
			return;
		}

		std::map<std::string, std::string> key_val_list;
		for (const auto &[key, val] : jsonObj.items())
		{
			key_val_list[key] = val;
		}

		for (const auto &[key, val] : key_val_list)
		{
			std::cout << key << ", " << val << std::endl;
			if (key == "SWITCH_CAM")
			{
				camSource = val;
			}
			if (key == "TRACK_ENABLE")
			{
				needTracking = atoi(val.c_str());
			}
			if (key == "SHAPE_TYPE")
			{
				if (val == "POLYGON")
				{
					restrictedAreaPts.clear();
					std::string x_data = key_val_list["x"];
					std::string y_data = key_val_list["y"];
					std::cout << "x_data = " << x_data << std::endl;
					std::cout << "y_data = " << y_data << std::endl;

					std::string delimiter = ",";

					std::vector<int> x_data_int, y_data_int;

					size_t pos = 0;
					std::string token;
					while ((pos = x_data.find(delimiter)) != std::string::npos)
					{
						token = x_data.substr(0, pos);
						std::cout << token << std::endl;
						x_data_int.push_back(atoi(token.c_str()));
						x_data.erase(0, pos + delimiter.length());
					}
					std::cout << x_data << std::endl;
					x_data_int.push_back(atoi(x_data.c_str()));

					size_t pos2 = 0;
					std::string token2;
					while ((pos2 = y_data.find(delimiter)) != std::string::npos)
					{
						token2 = y_data.substr(0, pos2);
						std::cout << token2 << std::endl;
						y_data_int.push_back(atoi(token2.c_str()));
						y_data.erase(0, pos2 + delimiter.length());
					}
					std::cout << y_data << std::endl;
					y_data_int.push_back(atoi(y_data.c_str()));

					float multiply_x = static_cast<float>(currentCapSize.width) / 1428;
					float multiply_y = static_cast<float>(currentCapSize.height) / 768;

					std::cout << "Ydatasize = " << y_data_int.size() << std::endl;
					for (int i = 0; i < x_data_int.size(); i++)
					{
						restrictedAreaPts.push_back(cv::Point(x_data_int.at(i) * multiply_x, y_data_int.at(i) * multiply_y));
						std::cout << restrictedAreaPts.size() << std::endl;
					}
				}
			}
		}

		response.set("Access-Control-Allow-Origin", "*");
		response.setStatus(Poco::Net::HTTPServerResponse::HTTP_OK);
		response.send().flush();
	}

} // namespace handlers
