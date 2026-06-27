#include <iostream> 
#include <vector>
#include <stack>


int rain_water(std::vector<int> height) {
    std::stack<int> st;
    int ans=0;
    for (int i=0; i<height.size();++i) {
        if (st.size()==0) {
            st.push(i);
        } else if (height[st.top()] > height[i]) {
            st.push(i);
        } else {
            int prev=0;
            while (st.size()>0 && height[st.top()]<=height[i]) {
                int cur = height[st.top()];
                ans += (cur-prev)*(i-st.top()-1);
                prev = cur;
                st.pop();
            }
            st.push(i);
        }
    }
    return ans;
}