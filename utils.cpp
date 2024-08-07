#include "./utils.hpp"

int neat::randint(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

double neat::max(double a, double b) {
    if(a>b) return a;
    return b;
}
double neat::min(double a, double b) {
    if(a<b) return a;
    return b;
}
