#include "plotting.h"

using std::filesystem::current_path;

void gnuPlotMatrix(std::string name, const Eigen::MatrixXd& matrix, std::string formation, std::vector<DataPoint>* data, int maxX, int maxY) {
    std::string min_val = std::to_string(matrix.minCoeff());
    std::string max_val = std::to_string(matrix.maxCoeff());
    std::string dirPath = getDir(formation);
    std::ofstream dataFile(dirPath + "/matrix_"+name+"_data.txt");
    dataFile << matrix;
    dataFile.close();
    double mxSize = static_cast<int>(matrix.rows());
    std::ofstream pointsFile(dirPath + "/matrix_observed_points.txt");
    for (const auto& point : (*data)) {
        pointsFile << ((double)point.x / maxX) * mxSize << " " << ((double)point.y / maxY) * mxSize << " " << point.value << "\n";
    }
    pointsFile.close();

    std::string gnuplotScript = R"(set title ')" + formation + R"('
        set xlabel 'Column'
        set ylabel 'Row'
        set pm3d map
        set palette defined (0 "#151F39", 1 "white")
        set offsets graph 0.5, 0, 0, 0.5
        set size square
        set size ratio -1
        set auto fix
        set cbrange [)" + min_val + ":" + max_val + R"(]
        plot 'matrix_)"+name + R"(_data.txt' matrix with image, 'matrix_observed_points.txt' u 1:2:3 with labels point offset character 0,character 1 tc rgb "black" notitle
    )";

    std::ofstream scriptFile(dirPath + "/matrix_" + name + ".plt");
    scriptFile << gnuplotScript;
    scriptFile.close();
}
void gnuPlotVariogram(std::string formation, EmpiricalVariogram* vari, TheoreticalParam param) {
    std::string dirPath = getDir(formation);
    std::vector<double> variograms = vari->values;
    std::vector<double> distances = vari->distances;
    std::string theoretical_data = dirPath + "/theoretical_data.txt";
    std::string empirical_data = dirPath + "/empirical_data.txt";
    std::string gnuplotScript = R"(
    set title')" + formation + R"('
    set xlabel 'Distance'
    set ylabel 'Variogram'
    plot ')" + theoretical_data + R"(' with line, ')" + empirical_data + R"(' with line
    )";
    std::ofstream scriptFile(dirPath+"/variogram.plt");
    std::ofstream empiricalFile(empirical_data);
    std::ofstream theoreticalFile(theoretical_data);
    auto nugget = param.nugget;
    auto sill = param.sill;
    auto range = param.range;
    for (size_t i = 0; i < variograms.size(); i++) {

        auto h = distances[i];
        empiricalFile << distances[i] << " " << variograms[i] << std::endl;
        theoreticalFile << distances[i] << " " <<   nugget + sill * (1.0 - exp(-((h * h) / (range * range))))  << std::endl;
    }
    empiricalFile.close();
    theoreticalFile.close();
    scriptFile << gnuplotScript;
    scriptFile.close();
}
std::string getDir(const std::string& formation) {

    std::error_code err;
    std::string dirPath = "/interpolations/" + formation;
    createDirectoryRecursive(dirPath, err);
    return std::filesystem::current_path().string() + dirPath;
}
void writeMatrixCoordinates(const Eigen::MatrixXd& matrix, const std::string& formation) {
    // Replace spaces in the formation string with underscores

    std::string dirPath = getDir(formation);
    std::string filePath = dirPath + "/interpolated_points.txt";
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }
    for (int i = 0; i < matrix.rows(); ++i)
        for (int j = 0; j < matrix.cols(); ++j) 
            outFile << i << " " << j << " " << matrix(i, j) << "\n";
    outFile.close();
}

bool createDirectoryRecursive(std::string const& dirName, std::error_code& err)
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