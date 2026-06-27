#include <iostream>
#include <vector>
using namespace std;


int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];  // Выбираем последний элемент как pivot
    int i = low - 1;        // Индекс для элементов меньше pivot

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);  // Ставим pivot на правильное место
    return i + 1;                      // Возвращаем индекс pivot
}
// Основная функция Quick Sort
void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot_idx = partition(arr, low, high);  // Индекс pivot после разделения
        quickSort(arr, low, pivot_idx - 1);         // Сортируем левую часть
        quickSort(arr, pivot_idx + 1, high);        // Сортируем правую часть
    }
}
int main() {
    vector<int> arr = {3, 6, 8, 10, 1, 2, 1};
    quickSort(arr, 0, arr.size() - 1);
    for (int num : arr) {
        cout << num << " ";  
    }
    cout << "\n";
}