#include "plotting.h"

using std::filesystem::current_path;

void gnuPlotMatrix(const Eigen::MatrixXd& matrix, std::string formation, std::vector<DataPoint>* data, int maxX, int maxY) {
    std::string min_val = std::to_string(matrix.minCoeff());
    std::string max_val = std::to_string(matrix.maxCoeff());
    std::error_code err;
    std::replace(formation.begin(), formation.end(), ' ', '_');
    CreateDirectoryRecursive("/interpolations/" + formation, err);
    std::string dirPath = current_path().string() + "/interpolations/" + formation;
    std::ofstream dataFile(dirPath + "/matrix_data.txt");
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
        set palette defined (0 "red", 1 "blue")
        set offsets graph 0.5, 0, 0, 0.5
        set size square
        set size ratio -1
        set auto fix
        set cbrange [)" + min_val + ":" + max_val + R"(]
        plot 'matrix_data.txt' matrix with image, 'matrix_observed_points.txt' u 1:2:3 with labels point offset character 0,character 1 tc rgb "black" notitle
    )";

    std::ofstream scriptFile(dirPath + "/matrix.plt");
    scriptFile << gnuplotScript;
    scriptFile.close();
}
void gnuPlotVariogram(std::string formation, EmpiricalVariogram* vari, TheoreticalParam param) {
    std::replace(formation.begin(), formation.end(), ' ', '_');
    std::error_code err;
    CreateDirectoryRecursive("/interpolations/" + formation, err);
    std::string dirPath = current_path().string() + "/interpolations/" + formation;
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
    for (size_t i = 0; i < variograms.size(); i++) {
        empiricalFile << distances[i] << " " << variograms[i] << std::endl;
        theoreticalFile << distances[i] << " " << gaussianFunction(param.nugget, param.sill, param.range, distances[i]) << std::endl;
    }
    empiricalFile.close();
    theoreticalFile.close();
    scriptFile << gnuplotScript;
    scriptFile.close();
}
bool CreateDirectoryRecursive(std::string const& dirName, std::error_code& err)
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