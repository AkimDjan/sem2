#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
using namespace std;

int main() {
    int n; cin >> n;
    vector<int> v1; vector<int> v2;
    for (size_t i = 0; i < n; i++) {
        int a; cin >> a;
        v1.push_back(a);
    }
    for (size_t i = 0; i < n; i++) {
        int a; cin >> a;
        v2.push_back(a);
    }
    vector<int> diff(n);
    for (int i = 0; i < n; i++) {
        diff[i] = v1[i] - v2[i];
    }
    vector<int> prefix_sum(n+1, 0);
    for (int i = 0; i < n; ++i) {
        prefix_sum[i + 1] = prefix_sum[i] + diff[i];
    }
    unordered_map<int, int> sum_index;
    int max_len = 0;
    sum_index[0] = 0;
    for (int i = 1; i <= n; ++i) {
        if (sum_index.count(prefix_sum[i])) {
            max_len = max(max_len, i - sum_index[prefix_sum[i]]);
        } else {
            sum_index[prefix_sum[i]] = i;
        }
    }
    cout << max_len << "\n";
}