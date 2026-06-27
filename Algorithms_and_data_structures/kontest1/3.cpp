#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int n; cin >> n;
    vector<int> v;
    for (int i=0; i<n; i++) {
        int a; cin >> a;
        v.push_back(a);
    }
    vector<int> zeros;
    for (int i=0; i<n; i++) {
        if (v[i]==0) {
            zeros.push_back(i);
        }
    }
    if (zeros.empty()) {
        cout << 0 << "\n"; 
        return 0;
    }
    /*
    vector<int> differences;
    for (int i=1; i<zeros.size(); i++) {
            differences.push_back(zeros[i]-zeros[i-1]);
    }
    cout<<differences[differences.size()]+differences[differences.size()-1]+1<<"\n";
    Здесь я пон что нехорошо, в ответе просят индекс а я его теряю
    */
    int max_len = 0;
    int best_idx = -1;
    for (int i=0; i<zeros.size(); ++i) {
        int l = (i==0) ? -1 : zeros[i-1];
        int r = (i==zeros.size()-1) ? n : zeros[i+1];
        int current_len = r-l-1;
        if (current_len>max_len) {
            max_len=current_len;
            best_idx=zeros[i];
        }
    }
    if (best_idx==-1) {
        best_idx=zeros[0];
    }
    cout << best_idx << "\n";
    return 0;
}