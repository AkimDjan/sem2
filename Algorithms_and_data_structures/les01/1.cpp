struct Node {
    Node* next = nullptr;
    int val = 0;
};

Node* ReverseList(Node* head) {
    if (head == nullptr) {return nullptr;}
    Node* cur_pos = head;
    Node* next_pos = cur_pos->next;
    Node* prev_pos = nullptr;
    while (cur_pos) {
        cur_pos->next = prev_pos;
        prev_pos = cur_pos;
        cur_pos = next_pos;
        if (cur_pos) {next_pos = cur_pos->next;}
    }
    return prev_pos;
}; //Наша память не может увеличиваться с увеличение входных данных - O(1) (нужно чтобы было O(n))