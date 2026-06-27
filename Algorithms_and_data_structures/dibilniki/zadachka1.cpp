#include <iostream>
#include <stdexcept>
using namespace std;

template <typename T>
class Queue {
private:
    struct Node {
        T val;
        Node* next;
        Node(const T& value) : val(value), next(nullptr) {} //конструктор для нода
    };
    Node* head; //указатель на бошку
    Node* tail; //указатель на хвост
    size_t count; //кол-во элементов
    T summ=0; //сумма

public:
    Queue() : head(nullptr), tail(nullptr), count(0) {} //конструктор

    ~Queue() {                                          //деструктор
        while (!isEmpty()) {
            dequeue();
        }
    }

    void enqueue(const T& value) {                      //добавление в конeц
        Node* newNode = new Node(value);
        if (isEmpty()) {
            head = tail = newNode; 
        } else {
            tail->next = newNode;
            tail = newNode;
        }
        count++;
        summ += value;
    }

    void dequeue() {                                    //удаление из начала
        if (isEmpty()) {
            throw out_of_range("Queue is empty");
        }
        Node* temp = head;
        head = head->next;
        summ -= temp -> val;
        delete temp;
        count--;
        if (isEmpty()) {
            tail = nullptr;
        }
        
    }

    T front() const {                                   //первый элемент
        if (isEmpty()) {
            throw out_of_range("Queue is empty");
        }
        return head->val;
    }
    
    bool isEmpty() const {                              //проверка на пустоту
        return count == 0;
    }
 
    int size() const {                                  //размер
        return count;
    }
};

int main() {
    Queue<int> q;
    cout << "is empty = " << boolalpha << q.isEmpty() << ", size = " << q.size() << "\n";
    q.enqueue(10);
    q.enqueue(20);
    q.enqueue(30);
    cout << "size = " << q.size() << ", front = " << q.front() << "\n";
    q.dequeue();
    cout << "size = " << q.size() << ", front = " << q.front() << "\n";
    
}