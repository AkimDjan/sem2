from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

def visualize_lsm(
    abscissa: np.ndarray,
    ordinates_experiment: np.ndarray,
    ordinates_theory: np.ndarray,
    ordinates_possible_max: np.ndarray,
    ordinates_possible_min: np.ndarray,
) -> None:
    plt.style.use("ggplot")
    plt.figure(figsize=(17, 9))
    plt.scatter(abscissa, ordinates_experiment, color='b', alpha=0.4)
    plt.plot(abscissa, ordinates_theory, color='b')
    plt.plot(abscissa, ordinates_possible_min, '--', color='b', alpha=0.4)
    plt.plot(abscissa, ordinates_possible_max, '--', color='b', alpha=0.4)
    
    plt.title("f(x) = 5.11x + 2.05")
    plt.grid(color='grey')
    plt.legend(['input_data', 'approximation', 'σ-coridor'])
    plt.xlim(abscissa.min(), abscissa.max());
    plt.show()


abscissa = np.linspace(0, 10, 100)
ordinates_experimental = 5 * abscissa + 2.5
ordinates_experimental += np.random.normal(
    size=ordinates_experimental.size,
    scale=2.5
)
ordinates_possible_max = abscissa * 5.5 + 4
ordinates_possible_min = abscissa * 4.6 + 1

visualize_lsm(
    abscissa=abscissa,
    ordinates_experiment=ordinates_experimental,
    ordinates_theory=abscissa * 5 + 2.5,
    ordinates_possible_max=ordinates_possible_max,
    ordinates_possible_min=ordinates_possible_min,
)