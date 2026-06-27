#include <iostream>
#include <vector>

using namespace std;

vector<int> Merge(const vector<int>& l, const vector<int>& r) {
    vector<int> result;
    result.reserve(l.size() + r.size()); //резервируем сразу нужное количество памяти
    int i = 0, j = 0;
    while (i < l.size() && j < r.size()) {
        if (l[i] < r[j]) {
            result.push_back(l[i]);
            i++;
        } else {
            result.push_back(r[j]);
            j++;
        }
    }
    while (i < l.size()) {
        result.push_back(l[i]);
        i++;
    }
    while (j < r.size()) {
        result.push_back(r[j]);
        j++;
    }
    return result;
}

vector<int> Mergesort(const vector<int> &a) {
    if (a.size() <= 1) {
        return a;
    }
    vector<int> l; vector<int> r;
    for (int i = 0; i < a.size() / 2; i++) {
        l.push_back(a[i]);
    }
    for (int i = a.size() / 2; i < a.size(); i++) {
        r.push_back(a[i]);
    }
    return Merge(Mergesort(l), Mergesort(r));
}

int main() {
    int n; cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        int x; cin >> x;
        a[i] = x;
    }
    a = Mergesort(a);
    for (int i = 0; i < n; i++) {
        cout << a[i] << ' ';
    }
}
