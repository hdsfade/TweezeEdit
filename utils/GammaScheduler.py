import math
from typing import List, Optional


class GammaScheduler:
    """
    Generates a list of regularization gamma values (float) 
    to schedule the regularization strength during the TweezeEdit process.
    """

    @staticmethod
    def generate(
        length: int,
        mode: str = "constant",
        value: float = 1.0,
        k: int = 0,
        custom_list: Optional[List[float]] = None
    ) -> List[float]:
        """
        :param length: Total number of generation steps (list length).
        :param mode: Scheduling strategy ('constant', 'step', 'custom_tail', 'linear', 'cosine').
        :param value: The base/maximum regularization strength (0.0 to 1.0).
        :param k: Threshold parameter (e.g., number of initial steps for 'step' mode).
        :param custom_list: List to be appended at the end for 'custom_tail' mode.
        """

        if mode == "constant":
            # Maintain the same strength throughout the entire process
            return [float(value)] * length

        elif mode == "step":
            # Apply 'value' for the first k steps, then drop to 0.0
            return [float(value)] * k + [0.0] * (length - k)

        elif mode == "custom_tail":
            # Keep 'value' for the start, then append a specific list at the end
            if custom_list is None:
                raise ValueError(
                    "custom_list must be provided for 'custom_tail' mode")

            head_len = length - len(custom_list)
            if head_len < 0:
                # If custom_list is longer than total length, truncate it from the start
                return [float(x) for x in custom_list[-length:]]
            return [float(value)] * head_len + [float(x) for x in custom_list]

        else:
            raise ValueError(f"Unsupported mode: {mode}")

# --- Examples ---

# Example 1: 15 steps, first 10 steps at 1.0, then 0.0
# [1.0, 1.0, ..., 0.0, 0.0]
# gamma_step = GammaScheduler.generate(length=15, mode="step", value=1.0, k=10)

# # Example 2: 15 steps, keep 1.0 strength but end with a specific sequence
# # [1.0, 1.0, ..., 0.0, 0.8, 0.8, 0.8, 0.0, 0.0]
# tail_seq = [0.0, 0.8, 0.8, 0.8, 0.0, 0.0]
# gamma_custom = GammaScheduler.generate(length=15, mode="custom_tail", value=1.0, custom_list=tail_seq)
