// prefix sums 2d

#include <iostream>
#include <vector>

using namespace std;

int main() {
    int n, m; 
    cin >> n >> m;

    vector<vector<int>> a(n + 1, vector<int>(m + 1, 0));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            cin >> a[i][j];
        }
    }

    vector<vector<int>> pref(n + 1, vector<int>(m + 1, 0));

    // building 

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            pref[i][j] = pref[i - 1][j] + pref[i][j - 1] - pref[i - 1][j - 1] + a[i][j];
        }
    }

    // (x1, y1, x2, y2)

    int q; cin >> q;
    for (int _ = 0; _ < q; _++) {
        int x1, y1, x2, y2; 
        cin >> x1 >> y1 >> x2 >> y2;
        int ans = pref[x2][y2] - pref[x1 - 1][y2] - pref[x2][y1 - 1] + pref[x1 - 1][y1 - 1];
        cout << ans << '\n';
    }

    return 0;
}