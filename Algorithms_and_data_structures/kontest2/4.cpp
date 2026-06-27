#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int n; cin >> n;
    vector<int> v;
    for (size_t i = 0; i < n; i++) {
        int a; cin >> a;
        v.push_back(a);
    }
    vector<int> ans(n,0);
    for (size_t j = 0; j < n; j++) {
        for (size_t k = j +1 ; k < n; k++) {
            if (v[j]<v[k]) {
                ans[j]++;
            }
        }
    }
    for (size_t i = 0; i < n; i++) {
        cout << ans[i] << " ";
    }
    cout << "\n";
}