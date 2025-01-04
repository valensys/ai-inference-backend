#include "ConfigReader.h"
#include <string>
#include <algorithm>

ConfigReader::ConfigReader() {};

ConfigReader::ConfigReader(std::string configFilePath) 
{
    this->configFilePath = configFilePath;
    this->configFile.open(configFilePath);
    if (!this->configFile.is_open()) 
    {
        std::cerr << "Config file can't be opened - " << configFilePath << std::endl;
        throw std::invalid_argument("Main config file can't be opened");
    }
};

ConfigReader::~ConfigReader() 
{
    if (this->configFile.is_open())
    {
        this->configFile.close();
    }
};

Configuration ConfigReader::readConfig()
{
    Configuration config = this->readConfig(this->configFilePath);
    return config;
}

Configuration ConfigReader::readConfig(std::string configFilePath)
{
    Configuration config;

    if (!this->configFile.is_open())
    {
        this->configFilePath = configFilePath;
        this->configFile.open(configFilePath);
        if (!this->configFile.is_open())
        {
            std::cerr << "Config file can't be opened - " << configFilePath << std::endl;
            throw std::invalid_argument("Main config file can't be opened");
        }
        else
        {
            std::cout << "Config file opened succesfully." << std::endl;
        }
    }

    std::string str;
    std::string delimiter = "=";

    while (getline(this->configFile, str))
    {
        str.erase(remove(str.begin(), str.end(), ' '), str.end());
        size_t pos = 0;
        pos = str.find(delimiter);

        if (pos != std::string::npos)
        {
            std::string key = str.substr(0, pos);
            std::string value = str.substr(pos + delimiter.length(), str.length());
            std::cout << "Key = " << key << " Value = " << value << std::endl;

            if (key == "gisBackUsername")
            {
                config.gisBackUsername = value;
            }
            if (key == "gisBackPassword")
            {
                config.gisBackPassword = value;
            }
            if (key == "gisAuthPath")
            {
                config.gisAuthPath = value;
            }
            if (key == "gisDbPath")
            {
                config.gisDbPath = value;
            }
            if (key == "gisBackHost")
            {
                config.gisBackHost = value;
            }
            if (key == "gisBackPort")
            {
                config.gisBackPort = value;
            }
            if (key == "storageDir")
            {
                config.storageDirLocal = value;
            }
            if (key == "host")
            {
                config.host = value;
            }
            if (key == "port")
            {
                config.port = value;
            }
            if (key == "modelPath")
            {
                config.modelPath = value;
            }
            if (key == "demoPath")
            {
                config.demoPath = value;
            }
            if (key == "videoSizeLimitMb")
            {
                config.maxVideoSize = std::stoi(value) * 1024 * 1024;
            }
        }
        else
        {
            std::cerr << " Wrong param " << str << " in config file - " << configFilePath << std::endl;
            throw std::invalid_argument("Wrong parameter in config file");
        }
    }

    this->checkConfig(config);

    return config;
}

void ConfigReader::checkConfig(const Configuration &config)
{
    if (config.gisBackUsername.size() == 0)
    {
        std::cout << "Error: gisBackUsername is empty" << std::endl;
    }
    if (config.gisBackPassword.size() == 0)
    {
        std::cout << "Error: gisBackPassword is empty" << std::endl;
    }
    if (config.gisBackHost.size() == 0)
    {
        std::cout << "Error: gisBackHost is empty" << std::endl;
    }
    if (config.gisBackPort.size() == 0)
    {
        std::cout << "Error: gisBackPort is empty" << std::endl;
    }
    if (config.host.size() == 0)
    {
        std::cout << "Error: host is empty" << std::endl;
    }
    if (config.port.size() == 0)
    {
        std::cout << "Error: port is empty" << std::endl;
    }
    if (config.storageDirLocal.size() == 0)
    {
        std::cout << "Error: storageDir is empty" << std::endl;
    }
    if (config.modelPath.size() == 0)
    {
        std::cout << "Error: modelPath is empty" << std::endl;
    }
    if (config.gisAuthPath.size() == 0)
    {
        std::cout << "Error: gisAuthPath is empty" << std::endl;
    }
    if (config.gisDbPath.size() == 0)
    {
        std::cout << "Error: gisDbPath is empty" << std::endl;
    }
}