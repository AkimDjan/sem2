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
    sort(v.begin(), v.end());
    int missing=1;
    for (int num : v) {
        if (num == missing) {
            missing++;
        } else if (num > missing) {
            break;
        }
    }
    cout << missing << endl;
}

int main() {
    int n;
    cin >> n;
    vector<int> v(n);
    
    for (int i = 0; i < n; i++) {
        cin >> v[i];
    }
    
    sort(v.begin(), v.end());
    
    int missing = 1; // Минимальное возможное положительное
    
    for (int num : v) {
        if (num == missing) {
            missing++; // Продолжаем искать следующее
        } else if (num > missing) {
            break; // Нашли разрыв
        }
    }
    
    cout << missing << endl;
    return 0;
}