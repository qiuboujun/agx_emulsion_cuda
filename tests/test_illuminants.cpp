#include "spectral_helpers.hpp"
#include <fstream>
#include <cmath>
#include <iostream>

static spec::dvec load(const std::string& path){
    std::ifstream f(path); double v; spec::dvec arr; while(f>>v) arr.push_back(v); return arr;}

int main(){
    auto wl = load("tests/illum_wl.txt");
    auto bb_ref = load("tests/illum_bb.txt");
    auto d65_ref = load("tests/illum_d65.txt");
    spec::dvec bb_calc, d65_calc;
    spec::blackbody_spd(6500.0, wl, bb_calc);
    spec::daylight_spd(6500.0, wl, d65_calc);
    auto check=[&](const spec::dvec& a,const spec::dvec& b,const char* name){
        double maxErr=0.0; for(size_t i=0;i<a.size();++i) maxErr = std::max(maxErr,std::abs(a[i]-b[i]));
        std::cout<<name<<" maxErr="<<maxErr<<"\n";
        return maxErr<1e-6;
    };
    bool ok = check(bb_calc,bb_ref,"blackbody") && check(d65_calc,d65_ref,"daylight");
    return ok?0:1;
} 