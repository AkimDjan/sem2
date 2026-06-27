// prefix sums 1d

#include <iostream>
#include <vector>

using namespace std;

int main() {
    
    int n; cin >> n;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
    }
    vector<int> pref(n + 1, 0);

    // pref[0] = a[0]

    //pref[i] = pref[i - 1] + a[i]
    for (int i = 1; i <= n; i++) {
        pref[i] = pref[i - 1] + a[i];
    }

    int q; cin >> q;
    for (int _ = 0; _ < q; _++) {
        int l, r; cin >> l >> r;
        int ans = pref[r] - pref[l - 1];
        cout << ans << '\n';
    }

    return 0;
}