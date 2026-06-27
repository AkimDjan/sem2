#include <stdint.h>

void IsZero (long* dest, long a, long b) {
    if (a<0) 
        *dest = 1;
    else    
        *dest = 0;
}