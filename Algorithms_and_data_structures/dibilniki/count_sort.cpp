#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void countingSort(vector<int>& arr) {
     if (arr.empty()) return;
     const int min_val = *min_element(arr.begin(), arr.end());
     const int max_val = *max_element(arr.begin(), arr.end());
     const int range = max_val - min_val + 1;
     vector<int> count(range), output(arr.size());
     for (const auto num : arr)
          count[num-min_val]++;
     for (int i=1; i<range; i++)
          count[i] += count[i - 1];
     for (int i=arr.size() - 1; i >= 0; i--) {
          output[count[arr[i] - min_val] - 1] = arr[i];
          count[arr[i] - min_val]--;
     }
     arr = output;
}

int main() {
     vector arr = {4, 2, 2, 8, 3, 3, 1};
     cout << "Исходный массив: ";
     for (const int num : arr)
          cout << num << " ";
     cout << "\n";
     countingSort(arr);
     cout << "Отсортированный массив: ";
     for (const int num : arr) {
          cout << num << " ";
     }
     cout << "\n";
}
