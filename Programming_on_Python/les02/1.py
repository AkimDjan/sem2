import math
import pathlib
import sys
#from numpy import sign

from typing import Callable, Optional

from utils import read_floats_from_bytes
from utils import visualize_1d_array

sample_amount = int(1e7)                    # количество отсчетов в сигнале                          
sampling_period = 6 * math.pi *1e-7         # период дискретизации
gane = 10                                   # коэффициент усиления
threshold = 4                               # порог
eps = 1e-6                                  # точность

def modulate(signal: float) -> float:
    return math.exp(-0.1 * signal)

def get_signal(
    sampling_period: float,
    sample_amount: int,
    modulation: Optional[Callable[[float], float]] = None,
) -> list[float]:
    if sampling_period<=0:
        raise ValueError("Sampling period must be above 0")
    if sample_amount<=0 or not isinstance(sample_amount,int):
        raise ValueError("Sample amount must be a natural digit")
    signal=[]
    for i in range(sample_amount):
        if modulate(i*sampling_period) == None:
            signal+=[math.sin(i*sampling_period)]
        signal+=[math.sin(i*sampling_period)*modulate(i*sampling_period)]
    return signal


def amplify_signal(
    signal: list[float],
    gane: float,
) -> list[float]:
    return [gane*i for i in signal]


def clip_signal(
    signal: list[float],
    threshold: float,
) -> list[float]:
    res_list=[]
    for i in range(len(signal)):
        if abs(signal[i]) <= threshold:
            res_list+=[signal[i]]
        else:
            sign = 0
            if signal[i]>0:
                sign = 1
            elif signal[i]<0:
                sign = -1
            res_list+=[threshold*sign]
    return res_list


#################################################
# %%timeit -r 1 -n 1
signal = get_signal(sampling_period, sample_amount, modulate)
signal_amplified = amplify_signal(signal, gane)
signal_clipped = clip_signal(signal_amplified, threshold)

print(f"signal size: {sys.getsizeof(signal)} bytes")

assert signal is not signal_amplified
assert signal_amplified is not signal_clipped

visualize_1d_array(ordinate=signal_clipped)

path_to_reference = pathlib.Path("./test_data/signal_clipped.log")
assert path_to_reference.exists(), "no reference data for testing"

signal_referense = read_floats_from_bytes(
    sample_amount, path_to_reference
)

assert all(
    abs(amp - amp_ref) < eps
    for amp, amp_ref in zip(signal_clipped, signal_referense)
)

del signal
del signal_amplified
del signal_clipped
del signal_referense