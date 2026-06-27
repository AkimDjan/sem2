#include <iostream>
#include <vector>
#include <queue>
#include <optional>
using namespace std;

//структурка бинарное дерево
struct Node {
    Node* parent = nullptr;
    Node* left = nullptr;
    Node* right = nullptr;
    int value;
};

//обход в глубину DFS
void Depth(Node* head) {
    if (!head) {return;}
    //1) Preorder: 5 3 1 4 7 9 
    Depth(head->left);
    //2) Inorder: 1 3 4 5 7 9
    Depth(head->right);
    //3) Postorder: 1 4 3 9 7 5
    //cout<<head->value; - эту строчку выставляем на место 1, 2 или 3
}

//обход в ширину BFS
void Breath(Node* head) {
    if (!head) {return;}
    queue<Node*> q;
    q.push(head);
    while (q.size()) {
        auto* node = q.front();
        if (node->left) q.push(node->left);
        if (node->right) q.push(node->right);
        cout<<node->value;
        q.pop();
    }
}

//задачка
bool InOrder(Node* head, optional<int>& prev) {
    if (!head) {return true;}
    auto res = InOrder(head->left,prev);
    cout<<head->value;
    InOrder(head->right, prev);
    res = res && (prev<head->value);
    prev = head->val;
    res = res && (InOrder(head->right, prev));
    return res;
}
bool IsSearchTree(Node* head) {
    bool res = InOrder(head, nullopt);
    return res;
}

//Проблема балансировки
struct Node {
    int val;
    Node* left = nullptr;
    Node* right = nullptr;
    int priority;
};








int main() {}