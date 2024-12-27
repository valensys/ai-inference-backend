#include "handlers/Index.h"

#include <Poco/Net/HTTPServerResponse.h>
#include <Poco/Net/HTTPServerRequest.h>
#include <Poco/StreamCopier.h>
#include <iostream>

namespace handlers
{
    void Index::handleRequest(
        Poco::Net::HTTPServerRequest &request,
        Poco::Net::HTTPServerResponse &response)
    {
        response.set("Access-Control-Allow-Origin", "*");
        response.setStatus(Poco::Net::HTTPResponse::HTTPStatus::HTTP_OK);
        response.send() << "{\"server\": \"http://localhost:8200\"}";
        response.send().flush();
    }
}