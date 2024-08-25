#include <client.h>

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    s->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

void make_request(const std::string& url) {
    CURL* curl;
    CURLcode res;
    std::string response_string;

    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        
        res = curl_easy_perform(curl);

        if(res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "Response from server: " << response_string << std::endl;
        }

        curl_easy_cleanup(curl);
    }
}

int doSomething() {
    std::string url = "http://localhost:13000/raster/pecs";
    make_request(url);

    return 0;
}
