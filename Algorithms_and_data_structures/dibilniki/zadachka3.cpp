#include <iostream>
using namespace std;

struct Node {
    int val;
    Node* left = nullptr;
    Node* right = nullptr;
};

Node* get_forward_list(Node* node, Node*& prev, Node*& head) {
    if (!node) return nullptr;
    get_forward_list(node->left, prev, head);
    if (!prev) {head = node;} else {prev->right = node;}
    node->left = nullptr;
    prev = node;
    get_forward_list(node->right, prev, head);
    return head;
}

int main() {
    Node* root = new Node{13};
    root->left = new Node{5};
    root->right = new Node{17};
    root->left->right = new Node{10};
    root->right->left = new Node{15};
    Node* prev = nullptr;
    Node* head = nullptr;
    Node* curr= get_forward_list(root,prev,head);
    cout << "list: "; 
    while (curr) {
        cout << curr->val << " ";
        curr = curr->right;
    }
    cout << "\n";
}


// struct Node {
//     int val;
//     Node * left = nullptr;
//     Node * right = nullptr;
// };

// Node* get_forward_list(Node* head, Node* prev) {
//     if (!head) {return nullptr;}
//     Node* right = head->right;
//     Node* origin_head = get_forward_list(head->left, prev);
//     if (prev) {
//         prev->right=head;
//         prev->left=nullptr;
//     } 
//     get_forward_list(right,prev);
//     return origin_head;
// }

// int main() {
//     Node* root = new Node{13,nullptr,nullptr};
//     root->left = new Node{5,nullptr,nullptr};
//     root->right = new Node{17,nullptr,nullptr};
//     root->left->right = new Node{10,nullptr,nullptr};
//     root->right->left = new Node{15,nullptr,nullptr};

//     Node* listHead = get_forward_list(root,nullptr);
//     Node* current = listHead;
//     cout << "spisok: "; 
//     while (current) {
//         cout << current->val << " ";
//         current = current->right;
//     }
//     cout << "\n";
//     return 0;
// }