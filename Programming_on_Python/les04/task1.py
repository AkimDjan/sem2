import numpy as np
from enum import Enum
from numbers import Real

class Strategies(Enum):
    BY_GOOD = "year"   # придумайте значение enum'а
    BY_MONTH = "month"  # придумайте значение enum'а

class InconsistentDataError(Exception):
    pass

def get_most_profitable_month_name(
    amounts_of_sold_subscriptions: np.ndarray,
    subscriptions_prices: np.ndarray,
) -> str:

    if len(amounts_of_sold_subscriptions[0])!=len(subscriptions_prices):
        raise InconsistentDataError("subcriptions_prices length must be equal length of amount_of_sild_subcriptions' string")

    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    for i in range(len(amounts_of_sold_subscriptions[0])):
        amounts_of_sold_subscriptions[:, i] *= subscriptions_prices[i]

    total_prices_of_sold_subscriptions = np.sum(amounts_of_sold_subscriptions, axis = 1, keepdims=True)
    
    return MONTHS[np.argmax(total_prices_of_sold_subscriptions)]


def get_mean_profit(
    amounts_of_sold_subscriptions: np.ndarray,
    subscriptions_prices: np.ndarray,
    strategy: Strategies | None = None,
) -> np.ndarray | Real:
    
    if len(amounts_of_sold_subscriptions[0])!=len(subscriptions_prices):
        raise InconsistentDataError("subcriptions_prices length must be equal length of amount_of_sild_subcriptions' string")
    
    prices_of_sold_subscriptions=np.zeros(shape=(12,len(amounts_of_sold_subscriptions[0])))
    for i in range(len(amounts_of_sold_subscriptions[0])):
        prices_of_sold_subscriptions[:, i] += amounts_of_sold_subscriptions[:, i] * subscriptions_prices[i]

    if strategy==None:
        return np.mean(np.sum(prices_of_sold_subscriptions, axis = 1, keepdims=True))
    
    elif strategy==Strategies.BY_MONTH:
        means_of_month=[]
        for i in range(len(prices_of_sold_subscriptions)):
            means_of_month += [float(np.mean(prices_of_sold_subscriptions[i]))]
        return np.array(means_of_month)
    
    elif strategy==Strategies.BY_GOOD:
        means_of_products = []
        prices_of_sold_subscriptions = prices_of_sold_subscriptions.T
        for i in range(len(prices_of_sold_subscriptions)):
            means_of_products += [float(np.mean(prices_of_sold_subscriptions[i]))]
        return np.array(means_of_products)

def sort_month_names_by_profits(
    amounts_of_sold_subscriptions: np.ndarray,
    subscriptions_prices: np.ndarray,
    ascending: bool = True,
) -> list[str]:
    
    if len(amounts_of_sold_subscriptions[0])!=len(subscriptions_prices):
        raise InconsistentDataError("subcriptions_prices length must be equal length of amount_of_sild_subcriptions' string")

    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    for i in range(len(amounts_of_sold_subscriptions[0])):
        amounts_of_sold_subscriptions[:, i] *= subscriptions_prices[i]

    total_prices_of_sold_subscriptions = np.sum(amounts_of_sold_subscriptions, axis = 1)

    array_of_months = []

    if not ascending:
        for i in range(len(total_prices_of_sold_subscriptions)):
            array_of_months += [MONTHS[np.argmax(total_prices_of_sold_subscriptions)]]

            total_prices_of_sold_subscriptions[np.argmax(total_prices_of_sold_subscriptions)] = -10**9

    else:
        for i in range(len(total_prices_of_sold_subscriptions)):
            array_of_months += [MONTHS[np.argmin(total_prices_of_sold_subscriptions)]]

            total_prices_of_sold_subscriptions[np.argmin(total_prices_of_sold_subscriptions)] = 10**9

    return array_of_months
