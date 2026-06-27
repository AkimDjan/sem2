#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

bool compare(const string &a, const string &b) {
    return a + b > b + a;
}

int main() {
    int n; cin >> n;
    vector<string> v(n+1);
    for (int i=0; i<n; i++) {
        cin>>v[i];
    }
    sort(v.begin(), v.end(), compare);
    string res;
    for (const string &s : v) {
        res+=s;
    }
    cout << res << "\n";
    return 0;
}