#include <vector>
#include <optional>
#include <cassert>

std::optional<std::pair<int, int>> GetElementsToSum(const std::vector<int>& vec, int sum) {
	
	if (vec.size() < 2) return {};
	
	size_t left = 0;
	size_t right = vec.size() - 1;
	while (left < right) {
		if (vec[left] + vec[right] == sum) {
			return std::pair<int, int>{ vec[left], vec[right] };
		} else if (vec[left] + vec[right] > sum) {
			right--;
		} else {
			left++;
		}
	}
	return {};	
}

int main() {
	// 1st example
	auto res = GetElementsToSum({ 1, 2, 3, 4, 5, 7 }, 7);
	assert(res.has_value());
	auto pair = *res;
	assert(pair.first == 2);
	assert(pair.second == 5);
	
	// 2nd example
	res = GetElementsToSum({ -3, 4, 5 }, -1);
	assert(!res.has_value());
	
	return 0;
}