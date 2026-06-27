#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
const int MIN_RUN = 32;  // Минимальный размер подмассива для Insertion Sort

// Вставками сортируем подмассив arr[left..right]
void insertionSort(vector<int>& arr, int left, int right) {
    for (int i = left + 1; i <= right; ++i) {
        int key = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

// Слияние двух подмассивов arr[left..mid] и arr[mid+1..right]
void merge(vector<int>& arr, int left, int mid, int right) {
    int len1 = mid - left + 1;
    int len2 = right - mid;
    // Создаем временные массивы
    vector<int> leftArr(arr.begin() + left, arr.begin() + mid + 1);
    vector<int> rightArr(arr.begin() + mid + 1, arr.begin() + right + 1);
    int i = 0, j = 0, k = left;
    // Слияние с выбором минимального элемента
    while (i < len1 && j < len2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k++] = leftArr[i++];
        } else {
            arr[k++] = rightArr[j++];
        }
    }
    // Дописываем оставшиеся элементы
    while (i < len1) arr[k++] = leftArr[i++];
    while (j < len2) arr[k++] = rightArr[j++];
}

// Основная функция Tim Sort
void timSort(vector<int>& arr) {
    int n = arr.size();
    
    // Сортируем подмассивы размера MIN_RUN вставками
    for (int i = 0; i < n; i += MIN_RUN) {
        insertionSort(arr, i, min(i + MIN_RUN - 1, n - 1));
    }
    
    // Слияние подмассивов
    for (int size = MIN_RUN; size < n; size *= 2) {
        for (int left = 0; left < n; left += 2 * size) {
            int mid = left + size - 1;
            int right = min(left + 2 * size - 1, n - 1);
            if (mid < right) {
                merge(arr, left, mid, right);
            }
        }
    }
}

int main() {
    vector<int> arr = {5, 21, 7, 23, 19, 3, 11, 15, 2, 8};
    cout << "До сортировки: ";
    for (int num : arr) cout << num << " ";
    cout << "\n";
    timSort(arr);
    cout << "После сортировки: ";
    for (int num : arr) std::cout << num << " ";
    
    return 0;
}