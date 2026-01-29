#ifndef TOOLS_H
#define TOOLS_H

#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

// Constants
constexpr double BOHR_TO_ANGS = 0.529177210903;
constexpr double ANGS_TO_BOHR = 1.0 / BOHR_TO_ANGS;
constexpr double PI = M_PI;

// Math Helpers
double double_factorial(int n);
double factorial(int n);

// String Helpers
std::string to_upper(const std::string& str);

#endif // TOOLS_H
