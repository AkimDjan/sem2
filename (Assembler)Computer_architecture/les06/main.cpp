#include <iostream>

extern "C"
void multstore(long long a, long long b, long long* res);

int main() {
    long long res{};
    multstore(2,3,&res);
    std::cout<< "2*3=" << res <<"\n";

}

extern "C"
long long mult (long long a, long long b) {return a*b;}