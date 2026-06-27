#include <iostream>
#include <vector>
using namespace std;


int main() {
    int n; cin >> n;
    int * f = new int[n+1];
    f[0] = 0; f[1] = 1;
    for (size_t i = 2; i <= n; i++) {
        f[i] = f[i-1] + f[i-2];
    }
    cout << f[n] << "\n";
    delete[] f;
    return 0;
}