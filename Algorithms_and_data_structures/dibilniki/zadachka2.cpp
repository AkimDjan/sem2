#include <iostream>
#include <vector>
using namespace std;

vector<int> merge_arrays(const vector<vector<int>>& arrays) {
    vector<int> v;
    for (const auto& arr : arrays) {
        for(auto elem : arr) {
            v.push_back(elem);
        }
    }
    //сортировка подсчетом 
    int mn = 1e9 , mx = -1e9;
    for (int num : v) {
        if (num < mn) mn = num;
        if (num > mx) mx = num;
    }
    int range = mx - mn + 1;
    vector<int> count(range, 0);
    vector<int> result;
    for (int num : v) {
        count[num - mn]++;
    }
    for (int i = 0; i < range; ++i) {
        result.push_back(i + mn);
    }
    return result;
}   


int main() {
    vector<vector<int>> input = {
        {1, 5, 6, 3},
        {2, 1, 7, 3},
        {8, 3, 2}
    };
    vector<int> output = merge_arrays(input);
    for (int i = 0; i < output.size(); i++) {
        cout << output[i] <<" ";
    }
    cout << "\n";
}