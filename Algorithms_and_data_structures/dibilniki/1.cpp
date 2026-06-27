#include <iostream>
using namespace std;

struct Node {
    int val;
    Node* left = nullptr;
    Node* right = nullptr;
};

void flatten(Node* node, Node*& prev, Node*& head) {
    if (!node) return;
    flatten(node->left, prev, head);
    if (!prev) {
        head = node;
    } else {
        prev->right = node;
    }
    node->left = nullptr;
    prev = node;
    flatten(node->right, prev, head);
}
Node* get_forward_list(Node* root) {
    Node* prev = nullptr;
    Node* head = nullptr;
    flatten(root, prev, head);
    return head;
}

int main() {
    Node* root = new Node{13};
    root->left = new Node{5};
    root->right = new Node{17};
    root->left->right = new Node{10};
    root->right->left = new Node{15};
    Node* curr= get_forward_list(root);
    cout << "list: "; 
    while (curr) {
        cout << curr->val << " ";
        curr = curr->right;
    }
    cout << "\n";
}