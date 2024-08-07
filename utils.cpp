#include "./utils.hpp"

using namespace neat;

int randint(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

double max(double a, double b) {
    if(a>b) return a;
    return b;
}
double min(double a, double b) {
    if(a<b) return a;
    return b;
}
