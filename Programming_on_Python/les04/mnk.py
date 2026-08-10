from numbers import Real
import numpy as np
from utils import visualize_lsm

from utils import visualize_1d


time = np.linspace(0, np.pi * 2, 500)
signal_high_freq = np.sin(time * 10)
signal = 5 * np.sin(time * 0.5) * signal_high_freq

extremum = [(time[np.argmax(signal)], np.max(signal))]
visualize_1d(time, signal, extremum)
"""
class ShapeMismatchError(Exception):
    pass

def get_lsm_coefficients(
    abscissa: np.ndarray,
    ordinates: np.ndarray,
) -> tuple[Real, Real]:
    x = abscissa
    y = ordinates
    
    if len(x) != len(y):
        raise ShapeMismatchError
    
    x_ = np.mean(x)
    y_ = np.mean(y)
    x2_ = np.mean(x ** 2)
    xy_ = np.mean(x * y)
    
    a = (xy_ - x_ * y_) / (x2_ - x_**2)
    b = y_ - a * x_
    
    return a, b

    return 0, 0

abscissa = np.linspace(0, 10, 100)
ordinates_experiment = abscissa * 5 + 3
ordinates_experiment += 2 * np.random.normal(size=abscissa.size)

incline, shift = get_lsm_coefficients(abscissa, ordinates_experiment)
print(
    f"computed incline: {incline:.2f};",
    f"computed shift: {shift:.2f};",
    sep="\n",
)

visualize_lsm(
    abscissa=abscissa,
    ordinates_experiment=ordinates_experiment,
    ordinates_computed=incline * abscissa + shift,
)
"""