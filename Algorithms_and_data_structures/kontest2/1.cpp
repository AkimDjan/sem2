#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m; cin >> n >> m;
    vector<int> v;
    for (size_t i = 0; i < n; i++) {
        int a; cin >> a;
        v.push_back(a);
    }
    vector<int> second(m);
    unordered_map<int, int> order;
    for (int i = 0; i < m; ++i) {
        cin >> second[i];
        order.emplace(second[i], i);
    }
    vector<int> in_order, not_in_order;
    for (int num : v) {
        if (order.find(num) != order.end()) {
            in_order.push_back(num);
        } else {
            not_in_order.push_back(num);
        }
    }
    sort(in_order.begin(), in_order.end(), [&order](int a, int b) {
        return order[a] < order[b];
    });
    sort(not_in_order.begin(), not_in_order.end());
    vector<int> result;
    result.reserve(n);
    result.insert(result.end(), in_order.begin(), in_order.end());
    result.insert(result.end(), not_in_order.begin(), not_in_order.end());
    for (int i = 0; i < n; i++) {
        cout << result[i] <<" ";
    }
    cout << '\n';
}
