#include <iostream>
#include <cstring>

void PrintFloatBinary(float f) {
    unsigned int binary;
    std::memcpy(&binary, &f, sizeof(f));
    std::cout << std::hex << binary << "\n";
}

int main() {
    PrintFloatBinary(9.0f);
    return 0;
}