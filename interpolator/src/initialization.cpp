#include "initialization.h"

void readObservationDataFromJson(std::vector<DataPoint>* data, std::string path, std::string formation,int* maxX, int* maxY) {
    std::ifstream f(path);
    json dataJson = json::parse(f);
    int minX = -1, minY = -1;
    double minVal = -1;

    for (const auto& entry : dataJson) {

        if (entry["reteg$lito$nev"] == formation) {
            DataPoint point;

            point.x = (int)entry["eovX"];
            point.y = (int)entry["eovY"];
            point.value = entry["reteg$mig"];
            if (minX == -1 || minX > point.x)
                minX = point.x;
            if (minY == -1 || minY > point.y)
                minY = point.y;
            if (minVal == -1 || minVal > point.value)
                minVal = point.value;
            if ((*maxX) < point.x)
                (*maxX) = point.x;
            if ((*maxY) < point.y)
                (*maxY) = point.y;
            data->push_back(point);
        }
    }
    (*maxX) = (*maxX) - minX;
    (*maxY) = (*maxY) - minY;

    for (int i = 0; i < data->size(); i++) {
        data->at(i).x -= minX;
        data->at(i).y -= minY;
        data->at(i).value -= minVal;
        //std::cout << "x: " << data[i].x << "y: " << data[i].y << "val: " << data[i].value << std::endl;
    }
}