#pragma once

#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> data;
};

struct Line {
    double rho;
    double theta;
};