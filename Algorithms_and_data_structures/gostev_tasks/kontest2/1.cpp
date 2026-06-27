#include <iostream>
#include <vector> 
/*
std::vector<int> puzirek(const std::vector<int>& arr) {
    std::vector<int> new_arr = arr;
    int n = arr.size();
    bool flag = true;
    while (flag) {
        for (size_t j; j<n-1; j++) {
            if (new_arr[j]>new_arr[j+1]) {
                flag = true;
                int temp;
                temp = new_arr[j];
                new_arr[j] = new_arr[j+1];
                new_arr[j+1] = temp;
            } else { 
                flag = false;
            }
        }
    }
    return new_arr;
}*/

int main() {
    int n;
    std::cin >> n;
    std::vector<int> arr;
    for (size_t i=0; i<n; i++) {
        int t;
        std::cin >> t;
        arr.push_back(t);
    }
    std::vector<int> new_arr = arr;
    for (size_t i; i<n; i++) {
        std::cout << new_arr[i]<< " ";
    }
    std::cout << "\n";
    bool flag = true;
    while (flag) {
        for (size_t j=0; j<n-1; j++) {
            if (new_arr[j]>new_arr[j+1]) {
                flag = true;
                int temp;
                temp = new_arr[j];
                new_arr[j] = new_arr[j+1];
                new_arr[j+1] = temp;
            } else { 
                flag = false;
            }
        }
    }
    for (size_t i; i<n; i++) {
        std::cout << new_arr[i]<< " ";
    }
    std::cout << "\n";
    return 0;
}

