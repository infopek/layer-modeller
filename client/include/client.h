#pragma once
#include <iostream>
#include <string>
#include <curl/curl.h>

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s);
void make_request(const std::string& url);
int doSomething(); 