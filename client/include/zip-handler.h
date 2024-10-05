#pragma once

#include <vector>

#include <string>
#include <iostream>
#include <zip.h> 
#include <zlib.h>
#include <fstream>

bool extractZipFromMemory(const std::vector<char>& zipData, const std::string& outputDir);