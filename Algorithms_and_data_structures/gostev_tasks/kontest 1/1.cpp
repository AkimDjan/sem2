#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    int n, q;
    std::cin >> n >> q;
    std::vector<int> a;
    for (size_t i = 0; i < n; i++) {
        int b;
        std::cin >> b;
        a.push_back(b);
    }
    sort(a.begin(), a.end());
    for (size_t i = 0; i < q; i++) {
        int x;
        std::cin >> x;
        int l = 0;
        int r = n - 1;
        while (r - l > 1) {
            if ( a[(r + l) / 2] > x ) {
                r = (r + l) / 2;
            } else {
                l = (r + l) / 2;
            }
        }
        if (a[r] == x || a[l] == x) {
            std::cout << "YES" << "\n";
        } else {
            std::cout << "NO" << "\n";
        }
    }
    return 0;
}