#include "tools.h"

double double_factorial(int n) {
    if (n <= 1) return 1.0;
    double res = 1.0;
    while (n > 1) {
        res *= n;
        n -= 2;
    }
    return res;
}

double factorial(int n) {
    if (n <= 1) return 1.0;
    double res = 1.0;
    for (int i = 2; i <= n; ++i) {
        res *= i;
    }
    return res;
}

std::string to_upper(const std::string& str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}
