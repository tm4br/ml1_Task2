from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Plotter:

    def __init__(self):
        pass

    @staticmethod
    def yerrs_from_confidences(confidences: List[Tuple[float, float]]) -> List[float]:
        yerrs = []
        for interval in confidences:
            lower, upper = interval
            yerrs.append((upper-lower) / 2)
        return yerrs

    def plot(
            self,
            x: List,
            y: List,
            title: str,
            xlabel: str,
            ylabel: str,
            confidence: List[Tuple[float, float]],
            output_path: Path,
            plot_type: str = "bar",  # "bar" or "line"
            rotation: int = 0,
            fig: Axes | Figure = plt.figure(figsize=(10, 8)),
    ):
        errors = self.yerrs_from_confidences(confidence)

        if plot_type == "bar":
            plt.bar(x, y, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
        elif plot_type == "line":
            upper = np.add(y, errors)
            lower = np.subtract(y, errors)

            plt.plot(x, y, color='blue', label="Line Plot", marker='o', markersize=5)
            plt.plot(x, upper, label='Line 1', color='red', linestyle='--')
            plt.plot(x, lower, label='Line 1', color='red', linestyle='--')
        else:
            raise ValueError(f"Invalid plot_type '{plot_type}'. Use 'bar' or 'line'.")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=rotation)
        #plt.ylim(0, 1)

        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.show()

