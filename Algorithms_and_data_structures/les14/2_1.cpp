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

    void moveToTail(Node* node) {
        if (node == tail) return;
        if (node->prev) node->prev->next = node->next;
        if (node->next) node->next->prev = node->prev;
        if (node == head) head = node->next;
        node->prev = tail;
        node->next = nullptr;
        if (tail) tail->next = node;
        tail = node;
        if (!head) head = tail;
    }

    void removeNode(Node* node) {
        if (node->prev) node->prev->next = node->next;
        if (node->next) node->next->prev = node->prev;
        if (node == head) head = node->next;
        if (node == tail) tail = node->prev;
        delete node;
    }

public:
    LRU_cache(size_t size) {size_=size;}

    ~LRU_cache() {
        while (head) {
            Node* cur = head;
            head = head->next;
            delete cur;    
        }
    }

    void add(size_t key, T value) {
        if (!tail) {
            head = new Node {key,value};
            tail=head;
        } else {
            Node* newnode =  new Node {key,value};
            tail->next = newnode;
            newnode->prev=tail;
            tail = tail->next;
        }
        keys[key]=tail; 
    }

    void delet(size_t key) {
        if (keys.find(key) == keys.end()) return;
        Node* node = keys[key];
        keys.erase(key);
        removeNode(node);
    }

    Т get(size_t key) {
        if (keys.find(key) == keys.end()) {
            throw out_of_range("Key not found");
        }
        Node* node = keys[key];
        moveToTail(node);
        return node->value;
    }
};
