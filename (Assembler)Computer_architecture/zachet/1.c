
int odd_sum(const int * ptr, int n) {
    int s = 0;
    const int * end = ptr + n;
    while (ptr!=end) {
        if (*(ptr)&1!=0) {++s;++ptr;}
    }
    return s;
}