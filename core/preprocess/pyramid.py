import numpy as np


def build_pyramid(image, levels=3):
    pyramid = []
    current = np.asarray(image)
    for _ in range(levels):
        pyramid.append(current)
        if current.shape[0] <= 2 or current.shape[1] <= 2:
            if len(pyramid) < levels:
                current = current.copy()
            else:
                break
        else:
            current = current[::2, ::2]
    return pyramid
