#include "zip-handler.h"

bool extractZipFromMemory(const std::vector<char>& zipData, const std::string& outputDir) {
    zip_source_t* src = zip_source_buffer_create(zipData.data(), zipData.size(), 0, nullptr);
    if (src == nullptr) {
        std::cerr << "Failed to create zip source from memory buffer." << std::endl;
        return false;
    }
    zip_error_t error;
    zip_t* zip = zip_open_from_source(src, 0, &error);
    if (zip == nullptr) {
        zip_source_free(src);
        std::cerr << "Failed to open ZIP archive from memory." << std::endl;
        return false;
    }
    zip_int64_t numFiles = zip_get_num_entries(zip, 0);
    for (zip_int64_t i = 0; i < numFiles; i++) {
        const char* fileName = zip_get_name(zip, i, 0);
        if (fileName == nullptr) {
            std::cerr << "Failed to get file name from ZIP archive." << std::endl;
            continue;
        }
        zip_file_t* file = zip_fopen_index(zip, i, 0);
        if (file == nullptr) {
            std::cerr << "Failed to open file inside ZIP archive: " << fileName << std::endl;
            continue;
        }

        zip_stat_t stat;
        zip_stat_index(zip, i, 0, &stat);
        std::vector<char> fileData(stat.size);
        zip_fread(file, fileData.data(), stat.size);

        std::string outputPath = outputDir + "/" + fileName;
        std::ofstream outFile(outputPath, std::ios::binary);
        outFile.write(fileData.data(), stat.size);
        outFile.close();

        std::cout << "Extracted file: " << fileName << " to " << outputPath << std::endl;

        zip_fclose(file);
    }
    zip_close(zip);
    return true;
}