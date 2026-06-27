// гистограммы

#include <bits/stdc++.h>

using namespace std;

int main() {
    
    int n; cin >> n;
    vector<int> h(n + 1);
    for (int i = 1; i <= n; i++) {
        int x; cin >> x;
        h[i] = x;
    }

    stack<pair<int, int>> minl, minr;
    vector<int> ml(n + 1), mr(n + 1);
    for (int i = 1; i <= n; i++) {
        while (!minl.empty() && minl.top().first >= h[i]) {
            minl.pop();
        }
        if (minl.empty()) {
            ml[i] = 0;
        } else {
            ml[i] = minl.top().second;
        }
        minl.push({h[i], i});
    }

    for (int i = n; i >= 1; i--) {
        while (!minr.empty() && minr.top().first >= h[i]) {
            minr.pop();
        }
        if (minr.empty()) {
            mr[i] = n + 1;
        } else {
            mr[i] = minr.top().second;
        }
        minr.push({h[i], i});
    }

    int ans = 0;

    for (int i = 1; i <= n; i++) {
        int l = ml[i] + 1, r = mr[i] - 1;
        ans = max(ans, h[i] * (r - l + 1));
    }

    cout << ans;

    return 0;
}