// Задача 1 [low high] решить через бин дерево поиска и через хэш таблицу
// надо найти сумму в этом диапазоне
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
using namespace std;

struct  Node
{
    int val;
    Node* left=nullptr;
    Node* right=nullptr; 
};

int SumBST(Node* root, int low, int high) {
    //low < q < high
    if (!root) {return 0;}
    int c=0;
    if (root->val>=low && root->val<=high) {
        c+=root->val;
    } 
    if (root->val > low) {
        c += SumBST(root->left, low, high);
    }
    if (root->val < high) {
        c += SumBST(root->right, low, high);
    }
    return c;
}

int main() {
    Node* root = new Node{10, 
        new Node{7, 
            new Node{6, nullptr, nullptr}, 
            new Node{8, nullptr, nullptr}
        }, 
        new Node{15, 
            new Node{11, nullptr, nullptr}, 
            new Node{18, nullptr, nullptr}
        }
    };
    int low = 5, high = 15;
    cout << "Cумма в диапазоне [" << low << ", " << high << "]: ";
    cout << SumBST(root, low, high) << "\n";
}