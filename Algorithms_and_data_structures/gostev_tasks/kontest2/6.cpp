// prefix sums 1d

#include <bits/stdc++.h>

using namespace std;

int main() {
    
    int n, m; cin >> n >> m;
    vector<vector<int>> a(n + 1, vector<int>(m + 1));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            cin >> a[i][j];
        }
    }

    vector<vector<int>> pref(n + 1, vector<int>(m + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (a[i][j] == 1) {
                pref[i][j] = pref[i - 1][j] + 1;
            } else {
                pref[i][j] = 0;
            }
        }
    }
    int glob_ans = 0;
    for (int i = 1; i <= n; i++) {
        vector<int> ml(n + 1), mr(n + 1);
        stack<pair<int, int>> minl, minr;
        int loc_ans = 0;
        for (int j = 1; j <= m; j++) {
            while (!minl.empty() && minl.top().first >= pref[i][j]) {
                minl.pop();
            }
            if (minl.empty()) {
                ml[j] = 0;
            } else {
                ml[j] = minl.top().second;
            }
            minl.push({pref[i][j], j});
        }
        for (int j = m; j >= 1; j--) {
            while (!minr.empty() && minr.top().first >= pref[i][j]) {
                minr.pop();
            }
            if (minr.empty()) {
                mr[j] = m + 1;
            } else {
                mr[j] = minr.top().second;
            }
            minr.push({pref[i][j], j});
        }
        for (int j = 1; j <= m; j++) {
            int l = ml[j] + 1, r = mr[j] - 1;
            loc_ans = max(loc_ans, pref[i][j] * (r - l + 1));
        }
        glob_ans = max(glob_ans, loc_ans);
    }
    cout << glob_ans;
    
    return 0;
}