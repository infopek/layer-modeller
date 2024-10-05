#pragma once
#include <iostream>
#include <string>
#include <curl/curl.h>
#include <vector>

struct MemoryStruct {
    std::vector<char> memory;
};

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s);
static size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp);

void make_request(const std::string& url);
bool downloadZipFile(const std::string& url, MemoryStruct& zipData);
int doSomething(); 