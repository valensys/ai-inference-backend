#include "handlers/Factory.h"

#include "handlers/PostRequestHandler.h"
#include "handlers/Index.h"

#include <Poco/Net/HTTPServerRequest.h>
#include <Poco/StreamCopier.h>

#include <iostream>
#include <ostream>
#include <string>
#include <sstream>

namespace handlers
{

Poco::Net::HTTPRequestHandler* Factory::createRequestHandler(
	const Poco::Net::HTTPServerRequest& request)
{
	std::cout << std::endl;
	
	if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_GET)
	{
		std::cout << "GET request" << std::endl;
	}

	if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_POST)
	{
		std::cout << "POST request" << std::endl;
		std::cout << "Content Length: " << request.getContentLength64() << std::endl;
		std::cout << "Content: " << request.getContentType() << std::endl;

		return new PostRequestHandler();
	}

	std::cout << "URI = " << request.getURI() << std::endl;

	if (request.getURI() == "/")
	{
		return new Index();
	}

	return nullptr;
}

} // namespace handlers
