#include <bits/stdc++.h>

using namespace std;

int main() {
    long double c;
    cin >> c;
    long double l = 0;
    long double  r = 1e12;
    while (r - l > 1e-12) {
        long double mid = (r + l) / 2.0;
        if (mid * mid * mid * mid + mid - c > 1e-12) {r = mid;} else {l = mid;}
    } 
    cout << fixed << setprecision(10) << l * l;
    return 0;
}