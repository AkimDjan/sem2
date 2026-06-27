#include <iostream>
#include <vector>
using namespace std;

// Функция для перестроения поддерева с корнем в i (max-куча)
void heapify(std::vector<int>& arr, int n, int i) {
    int largest = i;        // Корень
    int left = 2 * i + 1;   // Левый потомок
    int right = 2 * i + 2;  // Правый потомок

    // Если левый потомок больше корня
    if (left < n && arr[left] > arr[largest])
        largest = left;

    // Если правый потомок больше текущего наибольшего
    if (right < n && arr[right] > arr[largest])
        largest = right;

    // Если наибольший — не корень
    if (largest != i) {
        std::swap(arr[i], arr[largest]);
        heapify(arr, n, largest);  // Рекурсивно heapify поддерево
    }
}

// Основная функция Heap Sort
void heapSort(std::vector<int>& arr) {
    int n = arr.size();
    // Построение max-кучи (начинаем с последнего родителя)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
    // Извлечение элементов из кучи
    for (int i = n - 1; i > 0; i--) {
        std::swap(arr[0], arr[i]);  // Перемещаем максимум в конец
        heapify(arr, i, 0);         // Перестраиваем кучу для оставшихся
    }
}

int main() {
    vector<int> arr = {3, 1, 4, 1, 5, 9, 2};
    heapSort(arr);
    for (int num : arr)
        cout << num << " ";
    cout << "\n"; 
}