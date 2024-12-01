#include "plotting.h"

using std::filesystem::current_path;

void gnuPlotValidity(const LithologyData &lithodata, const WorkingArea &area, std::vector<Point> data, std::string properties)
{
    std::string dirPath = getDir(lithodata.stratumName);
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();

    // Determine min and max values for color range from interpolatedData
    for (const auto &point : data)
    {
        min_val = std::min(min_val, point.z);
        max_val = std::max(max_val, point.z);
    }
    std::ofstream dataFile(dirPath + "/matrix_observed_points_validation" + properties + ".txt");
    for (const auto &point : data)
    {
        double scaledX = (point.x - area.boundingRect.minX);
        double scaledY = (point.y - area.boundingRect.minY);
        std::string z = std::format("{:.1f}", std::round(point.z));
        dataFile << scaledX << " " << scaledY << " " << z << "\n";
    }
    dataFile.close();
    std::string gnuplotScript = R"(set title ')" + lithodata.stratumName + " cross-validation"+  R"('
        set encoding utf8
        set term png
        set output ')" + lithodata.stratumName + " cross-validation" + R"(.png'
        set terminal png size 600,600
        set key off
        set xrange [0:)" + std::to_string(area.xAxisPoints*area.xScale) + R"(]
        set yrange [0:)" + std::to_string(area.yAxisPoints*area.yScale) + R"(]
        set cbrange [)" + std::to_string(min_val) +
                                ":" + std::to_string(max_val) + R"(]
        set colorbox
        set size square
        set palette defined (1 'green', 2 'yellow', 3 'red')
        plot 'matrix_observed_points_validation)" +
                                properties + R"(.txt' using 1:2:3 with points palette pointtype 7 pointsize 1.5 notitle,\
        'matrix_observed_points_validation)" +
                                properties + R"(.txt' using 1:2:3 with labels offset 0,1)";
    std::ofstream scriptFile(dirPath + "/matrix_observed_points_validation" + properties + ".plt");
    scriptFile << gnuplotScript;
    scriptFile.close();
}
void gnuPlotArea(std::vector<Point> data, std::string stratumName, const WorkingArea &area, std::string dataTypeStr, std::string properties)
{
    std::string dirPath = getDir(stratumName);
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();

    // Determine min and max values for color range from interpolatedData
    for (const auto &point : data)
    {
        min_val = std::min(min_val, point.z);
        max_val = std::max(max_val, point.z);
    }
    std::ofstream dataFile(dirPath + "/matrix_" + dataTypeStr + properties + ".txt");
    for (const auto &point : data)
    {
        double scaledX = (point.x - area.boundingRect.minX);
        double scaledY = (point.y - area.boundingRect.minY);
        dataFile << scaledX << " " << scaledY << " " << point.z << "\n";
    }
    dataFile.close();
    std::string gnuplotScript = R"(set title ')" + stratumName + " " + dataTypeStr + R"('
        set encoding utf8
        set term png
        set output ')" + stratumName + " " + dataTypeStr + R"(.png'
        set terminal png size 600,600
        set key off
        set xrange [0:)" + std::to_string(area.xAxisPoints*area.xScale) + R"(]
        set yrange [0:)" + std::to_string(area.yAxisPoints*area.yScale) + R"(]
        set pm3d map
        set palette defined (0 "#151F39", 1 "white")
        set size square
        set auto fix
        set cbrange [)" + std::to_string(min_val) +
                                ":" + std::to_string(max_val) + R"(]
        plot 'matrix_)" + dataTypeStr +
                                properties + R"(.txt' using 1:2:3 with image, 'matrix_observed_points.txt' using 1:2:3 with labels point offset character 0,character 1 tc rgb "black" notitle
    )";
    std::ofstream scriptFile(dirPath + "/matrix_" + dataTypeStr + properties + ".plt");
    scriptFile << gnuplotScript;
    scriptFile.close();
}
void gnuPlotKriging(const LithologyData &lithodata, const WorkingArea &area, std::string properties)
{
    std::string dirPath = getDir(lithodata.stratumName);
    std::ofstream pointsFile(dirPath + "/matrix_observed_points.txt");
    for (const auto &point : lithodata.points)
    {
        double scaledX = (point.x - area.boundingRect.minX);
        double scaledY = (point.y - area.boundingRect.minY);
        pointsFile << scaledX << " " << scaledY << " " << point.z << "\n";
    }
    pointsFile.close();
    gnuPlotArea(lithodata.interpolatedData, lithodata.stratumName, area, "interpolation", properties);
    gnuPlotArea(lithodata.certaintyMatrix, lithodata.stratumName, area, "certainty", properties);
}
void gnuPlotVariogram(LithologyData &lithoData)
{
    std::string formation = lithoData.stratumName;
    std::string dirPath = getDir(formation);
    std::vector<double> variograms = lithoData.variogram.empirical.values;
    std::vector<double> distances = lithoData.variogram.empirical.distances;
    std::string theoretical_data = dirPath + "/theoretical_data.txt";
    std::string empirical_data = dirPath + "/empirical_data.txt";
    std::string gnuplotScript = R"(
    set title')" + formation + R"('
    set xlabel 'Distance'
    set ylabel 'Variogram'
    plot ')" + theoretical_data +
                                R"(' with line, ')" + empirical_data + R"(' with line
    )";
    std::ofstream scriptFile(dirPath + "/variogram.plt");
    std::ofstream empiricalFile(empirical_data);
    std::ofstream theoreticalFile(theoretical_data);
    auto param = lithoData.variogram.theoretical;
    auto nugget = param.nugget;
    auto sill = param.sill;
    auto range = param.range;
    for (size_t i = 0; i < variograms.size(); i++)
    {

        auto h = distances[i];
        empiricalFile << distances[i] << " " << variograms[i] << std::endl;
        theoreticalFile << distances[i] << " " << nugget + sill * (1.0 - exp(-((h * h) / (range * range)))) << std::endl;
    }
    empiricalFile.close();
    theoreticalFile.close();
    scriptFile << gnuplotScript;
    scriptFile.close();
}
std::string getDir(const std::string &formation)
{

    std::error_code err;
    std::string dirPath = "/interpolations/" + formation;
    createDirectoryRecursive(dirPath, err);
    return std::filesystem::current_path().string() + dirPath;
}
void writeMatrixCoordinates(const Eigen::MatrixXd &matrix, const std::string &formation)
{
    // Replace spaces in the formation string with underscores

    std::string dirPath = getDir(formation);
    std::string filePath = dirPath + "/interpolated_points.txt";
    std::ofstream outFile(filePath);
    if (!outFile.is_open())
    {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }
    for (int i = 0; i < matrix.rows(); ++i)
        for (int j = 0; j < matrix.cols(); ++j)
            outFile << i << " " << j << " " << matrix(i, j) << "\n";
    outFile.close();
}
void gnuPlotTestValidation(const std::map<std::string, CalculationRunTime>& dataMap, const std::string& plotTitle) {
    // Set file paths
    std::string dirPath = getDir("TEST");
    std::string dataFile = dirPath + "/combined_data_" + plotTitle + ".txt";
    std::string gnuplotScript = dirPath + "/combined_plot_" + plotTitle + ".plt";

    // Open data file for writing
    std::ofstream dataFileOut(dataFile);
    if (!dataFileOut) {
        std::cerr << "Error opening data file!" << std::endl;
        return;
    }

    for (const auto& entry : dataMap) {
        const auto& key = entry.first; 
        const auto& calcTime = entry.second;
        dataFileOut << key << " " 
                    << calcTime.variogram << " "
                    << calcTime.covMatrix << " "
                    << calcTime.kriging << "\n";
    }
    dataFileOut.close();

    std::ofstream gnuplotScriptOut(gnuplotScript);
    if (!gnuplotScriptOut) {
        std::cerr << "Error opening gnuplot script file!" << std::endl;
        return;
    }
    gnuplotScriptOut << R"(set title ')" + plotTitle + R"('
set xlabel 'Number of known points'
set ylabel 'Time (ms)'

# Plot the data with different colors for each parameter
plot ')" + dataFile + "' using 1:2 with lines title 'Variogram' linecolor rgb 'red', \\\n"
                   << "'"
                   + dataFile + "' using 1:3 with lines title 'CovMatrix' linecolor rgb 'blue', \\\n"
                   << "'"
                   + dataFile + "' using 1:4 with lines title 'Kriging' linecolor rgb 'green'\n";

    gnuplotScriptOut.close();
}
void gnuPlotValidationMatrix(const std::map<std::pair<std::string, std::string>, double> &mx)
{
    std::string dirPath = getDir("TEST");
    std::ofstream pointsFile(dirPath + "/matrix_validation.txt");
    for (const auto &point : mx)
    {
        pointsFile << point.first.first << " " << point.first.second << " " << point.second << "\n";
    }
    std::string gnuplotScript = R"(set title 'Validation'
        set pm3d map
        set palette defined (0 "#151F39", 1 "white")
        set size square
        set auto fix
        set cbrange [0:100]
        plot 'matrix_validation.txt' using 1:2:3 with image
    )";
    std::ofstream scriptFile(dirPath + "/matrix_validation.plt");
    scriptFile << gnuplotScript;
    pointsFile.close();
}


bool createDirectoryRecursive(std::string const &dirName, std::error_code &err)
{
    err.clear();

    if (!std::filesystem::create_directories(current_path().string() + dirName, err))
    {
        if (std::filesystem::exists(dirName))
        {
            // The folder already exists:
            err.clear();
            return true;
        }
        return false;
    }
    return true;
}