#pragma once
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/WebSocket.h"
#include "Poco/Util/ServerApplication.h"
#include "Poco/Util/Option.h"
#include "Poco/Util/OptionSet.h"
#include "Protocol.h"
#include "ConfigReader.h"


class PageRequestHandler: public Poco::Net::HTTPRequestHandler
{
public:
	void handleRequest(Poco::Net::HTTPServerRequest& request, Poco::Net::HTTPServerResponse& response);
};


class WebSocketRequestHandler: public Poco::Net::HTTPRequestHandler
{
public:
	void handleRequest(Poco::Net::HTTPServerRequest& request, Poco::Net::HTTPServerResponse& response);
	void processPackage(Poco::Net::WebSocket& ws);
	BufferRequestType annalyzeRequest(const char* buffer, int nBytesReceived);
	void videoProcessor(std::string filename, Poco::Net::WebSocket& ws);
	void demoStreamProcessor(Poco::Net::WebSocket& ws);
};


class RequestHandlerFactory: public Poco::Net::HTTPRequestHandlerFactory
{
public:
	Poco::Net::HTTPRequestHandler* createRequestHandler(const Poco::Net::HTTPServerRequest& request);
};

class WebSocketServer: public Poco::Util::ServerApplication
{
public:
	WebSocketServer(): _helpRequested(false)
	{
	}

	~WebSocketServer()
	{
	}

protected:
	std::string readConfig(std::string configPath);
	std::string getJWTToken(const Configuration& configuration);
	void initialize(Poco::Util::Application& self);
	void uninitialize();
	void defineOptions(Poco::Util::OptionSet& options);
	void handleOption(const std::string& name, const std::string& value);
	void displayHelp();
	int main(const std::vector<std::string>& args);

private:
	bool _helpRequested;
};
