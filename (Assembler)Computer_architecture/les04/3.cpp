#include <iostream>
#include <cstring>

void PrintFloatBinary(float f) {
    unsigned int binary;
    std::memcpy(&binary, &f, sizeof(f));
    
    for (int i=31; i>=0;--i) {
        std::cout << ((binary >> i)& 1);
        if ((i == 31) || (i== 31- 8)) {
            std::cout<< ' ';
        }
    }
    std::cout << std::endl;
    std::cout << std::hex << binary << "\n";
}

int main() {
    PrintFloatBinary(9.0f);
}