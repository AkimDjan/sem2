#include <stack>
#include <iostream>
#include <vector>

using namespace std;

int trap(vector<int>& heights) {
    stack<int> prevBounds;
    int res = 0;
    prevBounds.push(0);
    for (int i=1, sz=heights.size(); i<sz; ++i)
    {
        while(prevBounds.size() && heights[i] >= heights[prevBounds.top()])
        {
            auto prevBound = prevBounds.top();
            prevBounds.pop();
            if(prevBounds.empty())
                break;
            auto waterLength = i-prevBounds.top()-1;
            auto height = min(heights[i], heights[prevBounds.top()]);
            auto waterDepth = height - heights[prevBound];
            res += waterDepth*waterLength;
        }
        prevBounds.push(i);
    }
    return res;
}

int main() {
    int n; cin >> n;
    vector<int> v;
    for (size_t i = 0; i < n; i++) {
        int a; cin >> a;
        v.push_back(a);
    }
    cout << trap(v) << "\n";
} //если что я не списывал, мы эту задачу разбирали на семинаре