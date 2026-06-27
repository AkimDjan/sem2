#include <iostream>
#include <vector>
using namespace std;

unsigned int Partition (unsigned int l, unsigned int r, vector<int>& v) {
    unsigned int lp=l;
    for (unsigned int i=l; i<r;++i) {
        if (v[i]<v[r]) {
            swap(v[i],v[lp]); lp++;
        }
        swap(v[r],v[lp]);
        return lp;
    }
}

int Get_kth(vector<int>& v, unsigned int l = 0, unsigned int r, int k) {
    r = v.size() - 1;
    unsigned int p = Partition(l,r,v);
    if (p == k) {return v[k];} 
    if (k < p) {return Get_kth(v,l,p);}
    return Get_kth(v,p,r);
}