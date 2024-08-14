#include <logging.h>

#include <iostream>
#include <string>
#include <ctime>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

void Logger::init(const std::string& filename = "")
{
    if (!filename.empty())
    {
        std::filesystem::path logPath(filename);
        std::filesystem::create_directories(logPath.parent_path());
        getInstance().m_fileStream.open(filename, std::ios::out | std::ios::trunc);

        if (!getInstance().m_fileStream)
            std::cerr << "Failed to open log file: " << filename << std::endl;
    }
}

Logger::~Logger()
{
    if (m_fileStream.is_open())
        m_fileStream.close();
}

void Logger::log(LogLevel level, const std::string& message)
{
    std::lock_guard<std::mutex> lock(getInstance().m_mtx);
    std::ostringstream logStream;
    logStream << "[" << getInstance().currentDateTime() << "] "
        << getInstance().logLevelToString(level) << ": "
        << message << std::endl;

    std::string logMessage = logStream.str();

    if (getInstance().m_fileStream.is_open())
    {
        getInstance().m_fileStream << logMessage;
        getInstance().m_fileStream.flush();
    }
    else
        std::cout << logMessage;
}

std::string Logger::logLevelToString(LogLevel level)
{
    switch (level)
    {
    case LogLevel::DEBUG:       return "DEBUG";
    case LogLevel::INFO:        return "INFO";
    case LogLevel::WARN:        return "WARN";
    case LogLevel::ERROR:       return "ERROR";
    case LogLevel::CRITICAL:    return "CRITICAL";
    default:                    return "UNKNOWN";
    }
}

std::string Logger::currentDateTime()
{
    std::time_t now = std::time(nullptr);
    std::tm* localtm = std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(localtm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

Logger& Logger::getInstance()
{
    static Logger instance;
    return instance;
}