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
    if (v.size()<3) {
        cout << "No triplet exists\n";
        return 0;
    }
    vector<int> ans;
    /*
    for (int i=0; i<n; i++) {
        for (int j=i+1; j<n; j++) {
            for (int k=j+1; k<n; k++) {
                ans.push_back(v[i]*v[j]*v[k]);
            }
        }
    }
    int min_val = 1e9;
    for (int num : v) {
        if (num < min_val) {
            min_val = num;
        }
    }
    int max_val = -1e9;
    for (int num : v) {
        if (num < max_val) {
            max_val = num;
        }
    }
    */
    sort(v.begin(), v.end());
    if (v[0]<=0) {
        cout << v[0]*v[n-1]*v[n-2] << "\n";
    } else {
        cout << v[0]*v[1]*v[2] << "\n";
    }
}
