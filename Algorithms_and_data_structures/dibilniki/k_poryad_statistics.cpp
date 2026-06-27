#include <iostream>
#include <vector>
#include "quick_sort.cpp"
using namespace std;


int quickSelect(vector<int>& arr, int l, int r, int k) {
    if (l == r) return arr[l];
    int p = partition(arr, l, r);
    if (k == p) return arr[p];
    else if (k < p) return quickSelect(arr, l, p - 1, k);
    else return quickSelect(arr, p + 1, r, k);
}