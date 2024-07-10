#pragma once

#include <fstream>
#include <mutex>
#include <string>
#include <sstream>

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    CRITICAL
};

class Logger {
public:
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    ~Logger();

    static void init(const std::string& filename);

    static void log(LogLevel level, const std::string& message);

private:
    Logger() = default;

    static Logger& getInstance();

    std::string logLevelToString(LogLevel level);

    std::string currentDateTime();

private:
    std::ofstream m_fileStream;
    std::string m_logFile;
    std::mutex m_mtx;
};