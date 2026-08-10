import numpy as np

class ShapeMismatchError(Exception):
    pass

def can_satisfy_demand(
    costs: np.ndarray,
    resource_amounts: np.ndarray,
    demand_expected: np.ndarray,
) -> bool:
    
    flag = True

    if costs.shape[0] != resource_amounts.shape[0] or costs.shape[1] != demand_expected.shape[0]:
        raise ShapeMismatchError("shapes of arrays must be equal")
    
    for i in range(len(demand_expected)):
        resource_amounts=resource_amounts-demand_expected[i]*costs[:,i]
        if np.sum(resource_amounts)<0:
            return False

    return flag

#############################

costs = np.eye(2)
resource_amounts = np.full(shape=2, fill_value=3)
demand_expected = np.full(shape=2, fill_value=2)
assert can_satisfy_demand(costs, resource_amounts, demand_expected)
assert not can_satisfy_demand(costs, demand_expected, resource_amounts)