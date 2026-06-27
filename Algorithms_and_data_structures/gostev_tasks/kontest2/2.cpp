#include <iostream>
#include <vector>
#include <algorithm>




int main() {

    int n;
    std::cin >> n;
    std::vector<int> arr;
    for (size_t i=0; i<n; i++) {
        int t;
        std::cin >> t;
        arr.push_back(t);
    }
    for (size_t i; i<n; i++) {
        std::cout << arr[i]<< " ";
    }
    int m = *std::max_element(arr.begin(), arr.end());
    std::cout << m << "\n";


}