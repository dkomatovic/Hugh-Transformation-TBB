#include <iostream>
#include <filesystem>
#include "ImageIO.h"
#include "LineVisualization.h"
#include "Report.h"
#include <tbb/tbb.h>
#include "ParallelTransformation.h"
#include <chrono>
#include <string>

namespace fs = std::filesystem;

struct ImageJob {
    fs::path imgPath;
    Image original;
    Image gray;
    Image edges;
    std::vector<std::vector<int>> accumulator;
    std::vector<Line> lines;
    Image result;
    Report report;
    std::chrono::steady_clock::time_point totalStart;
    std::chrono::steady_clock::time_point totalEnd;
};

int ProcessAccumulatorAndGetThreshold(const std::vector<std::vector<int>>& accumulator, Report& report) {
    int maxVal = 0;
    for (const auto& row : accumulator)
        for (int v : row)
            if (v > maxVal) maxVal = v;

    std::vector<int> hist(maxVal + 1, 0);
    for (const auto& row : accumulator)
        for (int v : row)
            hist[v]++;

    report.maxAccumulatorVote = maxVal;
    report.accumulatorHistogram = hist;

    return static_cast<int>(0.70 * maxVal);
}

#include "oneapi/tbb/flow_graph.h"
using namespace oneapi::tbb::flow;

void processImagesWithFlowGraph(std::string inputFolder, std::string outputFolder) {
    tbb::flow::graph g;

    function_node<fs::path, ImageJob> loadNode(g, 1, [](fs::path p) -> ImageJob {
        ImageJob job;
        job.totalStart = std::chrono::steady_clock::now();
        job.imgPath = p;
        job.original = LoadImageFromFile(p);
        job.report.imgTitle = p.stem().string();
        job.report.processingMode = "Parallel";
        return job;
        });

    function_node<ImageJob, ImageJob> grayNode(g, unlimited, [](ImageJob job) -> ImageJob {
        auto start = std::chrono::steady_clock::now();
        job.gray = Grayscale(job.original);
        auto end = std::chrono::steady_clock::now();
        job.report.grayscaleDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return job;
        });

    function_node<ImageJob, ImageJob> edgeNode(g, unlimited, [](ImageJob job) -> ImageJob {
        auto start = std::chrono::steady_clock::now();
        job.edges = SobelEdgeDetection(job.gray, 100);
        auto end = std::chrono::steady_clock::now();
        job.report.edgeDetectionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return job;
        });

    function_node<ImageJob, ImageJob> houghNode(g, unlimited, [](ImageJob job) -> ImageJob {
        auto start = std::chrono::steady_clock::now();
        job.accumulator = HoughTransform(job.edges);
        auto end = std::chrono::steady_clock::now();
        job.report.houghTransformationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return job;
        });

    function_node<ImageJob, ImageJob> lineNode(g, unlimited, [](ImageJob job) -> ImageJob {
        auto start = std::chrono::steady_clock::now();
        int threshold = ProcessAccumulatorAndGetThreshold(job.accumulator, job.report);
        auto concurrent_lines = DetectLines(job.accumulator, threshold);
        job.lines.assign(concurrent_lines.begin(), concurrent_lines.end());
        auto end = std::chrono::steady_clock::now();
        job.report.accumulatorThreshold = threshold;
        job.report.lineDetectionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        job.report.detectedLinesNum = job.lines.size();
        return job;
        });

    function_node<ImageJob, ImageJob> drawNode(g, unlimited, [](ImageJob job) -> ImageJob {
        job.result = DrawLines(job.original, job.lines, 170, 0, 0);
        return job;
        });

    function_node<ImageJob, ImageJob> saveNode(g, 1, [&](ImageJob job) -> ImageJob {
        std::string resultFolder = outputFolder + "/" + job.imgPath.stem().string();
        SaveImageToFile(job.gray, resultFolder + "/gray_" + job.imgPath.stem().string());
        SaveImageToFile(job.edges, resultFolder + "/edge_" + job.imgPath.stem().string());
        SaveImageToFile(job.result, resultFolder + "/final_" + job.imgPath.stem().string());
        job.totalEnd = std::chrono::steady_clock::now();
        job.report.totalProcessingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(job.totalEnd - job.totalStart).count();
        job.report.Save(resultFolder + "/report.txt");
        job.report.SaveHistogram(resultFolder + "/histogram.png");

        return job;
        });

    make_edge(loadNode, grayNode);
    make_edge(grayNode, edgeNode);
    make_edge(edgeNode, houghNode);
    make_edge(houghNode, lineNode);
    make_edge(lineNode, drawNode);
    make_edge(drawNode, saveNode);

    std::vector<fs::path> allPaths = FindInputImages(inputFolder);

    for (const auto& path : allPaths) {
        loadNode.try_put(path);
    }

    g.wait_for_all();

}

void processImage(fs::path imgPath, std::string outputFolder) {
    Report report;
    report.processingMode = "Parallel";
    auto totalStart = std::chrono::steady_clock::now();

    // loading image
    Image img = LoadImageFromFile(imgPath);
    report.img = img;
    report.imgTitle = imgPath.stem().string();

    // grayscale phase
    auto start = std::chrono::steady_clock::now();
    Image grayImg = Grayscale(img);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    report.grayscaleDuration = duration;

    // edge detection phase
    start = std::chrono::steady_clock::now();
    Image edgesImg = SobelEdgeDetection(grayImg, 100);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    report.edgeDetectionDuration = duration;

    // hough transformation phase
    start = std::chrono::steady_clock::now();
    auto accumulator = HoughTransform(edgesImg);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    report.houghTransformationDuration = duration;

    // line detecting phase
    int threshold = ProcessAccumulatorAndGetThreshold(accumulator, report);
    report.accumulatorThreshold = threshold;

    start = std::chrono::steady_clock::now();
    auto concurrent_lines = DetectLines(accumulator, threshold);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    report.lineDetectionDuration = duration;
    report.detectedLinesNum = concurrent_lines.size();

    std::vector<Line> lines; 
    lines.assign(concurrent_lines.begin(), concurrent_lines.end());
    // visualizing lines
    Image imgWithLines = DrawLines(img, lines, 170, 0, 0);

    // Saving results to output folder
    std::string resultFolder = outputFolder + "/" + imgPath.stem().string();
    SaveImageToFile(grayImg, resultFolder + "/gray_" + imgPath.stem().string());
    SaveImageToFile(edgesImg, resultFolder + "/edge_" + imgPath.stem().string());
    SaveImageToFile(imgWithLines, resultFolder + "/final_" + imgPath.stem().string());
    auto totalEnd = std::chrono::steady_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();
    report.totalProcessingDuration = totalDuration;

    report.Print();
    report.Save(resultFolder + "/report.txt");
    report.SaveHistogram(resultFolder + "/histogram.png");
}

int main()
{
    std::cout << "HOUGH TRANSFORMATION (PARALLEL)\n";

    std::string inputFolder = "../Input";
    std::string outputFolder = "../OutputParallel";

    processImagesWithFlowGraph(inputFolder, outputFolder);

    /*std::vector<fs::path> imgPaths = FindInputImages(inputFolder);

    if (imgPaths.size() == 0) {
        std::cout << "No images found in the Input folder\n";
        return 0;
    }

    try {
        std::cout << "Choose the index of the image you would like to process: \n";
        for (size_t i = 0; i < imgPaths.size(); i++)
            std::cout << "[" << i << "] " << imgPaths[i].stem().string() << "\n";

        int index;
        fs::path chosenImgPath;
        while (true) {
            try {
                std::cout << "Index: ";
                std::cin >> index;

                if (std::cin.fail() || index < 0 || index >= static_cast<int>(imgPaths.size())) {
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << "Invalid index. Please enter a number between 0 and " << imgPaths.size() - 1 << std::endl;
                }
                else {
                    chosenImgPath = imgPaths[index];
                    break;
                }
            }
            catch (std::exception e) {
                std::cout << "Invalid index. Try again\n";
            }
        }

        processImage(chosenImgPath, outputFolder);
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }*/
}