#include <models/point.h>

#include <gdal_priv.h>
#include <models/lithologyData.h>
#include <stdexcept>

class GeoTiffHandler
{
public:
    GeoTiffHandler(const std::string& filepath);
    ~GeoTiffHandler();

    BoundingRectangle getBoundingRectangle();

private:
    GDALDataset* m_dataset;
};
