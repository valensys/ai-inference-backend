#pragma once

#include <nlohmann/json.hpp>
#include <Poco/Net/HTTPRequestHandler.h>

namespace handlers
{

class PostRequestHandler: public Poco::Net::HTTPRequestHandler
{
private:
	void handleRequest(
		Poco::Net::HTTPServerRequest& request,
		Poco::Net::HTTPServerResponse& response) override;
};

} // namespace handlers
