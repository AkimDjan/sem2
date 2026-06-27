struct Node {
    Node* next = nullptr;
    int val = 0;
};

bool IsCircled(Node* head) {
    Node* fast_ptr = head;
    Node* slow_ptr = head;
    while(fast_ptr != nullptr && fast_ptr->next!=nullptr) {
        slow_ptr=slow_ptr->next;
        fast_ptr=fast_ptr->next->next;
        if (slow_ptr==fast_ptr) {
            return true;
        }
    }
    return false;
}