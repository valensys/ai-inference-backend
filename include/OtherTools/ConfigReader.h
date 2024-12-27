#pragma once
#include <iostream>
#include <fstream>

struct Configuration
{
    std::string gisBackHost;
    std::string gisBackPort;
    std::string gisBackUsername;
    std::string gisBackPassword;
    std::string gisAuthPath;
    std::string gisDbPath;
    std::string port;
    std::string host;
    std::string storageDirLocal;
    std::string modelPath;
    std::string demoPath;
    uint64_t maxVideoSize = 0;
};

class ConfigReader {
public:
    ConfigReader();
    ConfigReader(std::string configFilePath);
    ~ConfigReader();

    Configuration readConfig();
    Configuration readConfig(std::string configFilePath);

private:
    std::string configFilePath;
    std::ifstream configFile;

    void checkConfig(const Configuration& config);
};