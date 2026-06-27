//Задача 2
//lru cache? 
//lru-chache 
//Есть `Key` и `Value` (Value - тяжелый объект, который надо кэшировать)
//Создаем двусвязный список, элементов {key, value}, у нас есть `chache_size`.
//Каждый раз, когда мы используем объект, тогда мы переставляем его в конец списка. Когда надо удалить, удаляем элемент из самого начала. В конец добавим актуальный.
// Хэш-таблица представлена для \<key, Node*>, благодаря этому мы будем за константу находить, где находится узел.

#include <iostream>
#include <unordered_map>
using namespace std;

template <typename T>

class LRU_cache{
private:
    struct Node{
        size_t key;
        T value;
        Node* prev=nullptr;
        Node* next=nullptr;
    };
    size_t size_;
    unordered_map<size_t, Node*> keys;
    Node *head=nullptr, *tail=nullptr;

public:
    LRU_cache(size_t size) : size_(size) {}

    ~LRU_cache() {
        while (head) {
            Node* cur = head;
            head = head->next;
            delete cur;    
        }
    }

    void add(size_t key, T value) {
        if (!tail) {
            head=new Node {key,value};
            tail=head;
        } else {
            tail->next = new Node {key,value};
            tail = tail->next;
        }
        keys[key]=tail; 
    }

    void delet(size_t key) {
        
    }

    void get(size_t key) {

    }
};



int main() {
    cout << "Hello, world!"<< "\n";
}
