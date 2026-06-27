
long long mult(long long a, long long b);

void multstore(long long a, long long b, long long* res) {
    long temp = mult(a,b);
    *res = temp;
    
}