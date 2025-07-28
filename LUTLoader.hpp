#pragma once
#include <vector>
#include <string>

struct FilmLUT {
    std::vector<float> logE;
    std::vector<float> curveR;
    std::vector<float> curveG;
    std::vector<float> curveB;
};

bool loadFilmLUT(const std::string& stockFolder, FilmLUT& out); 