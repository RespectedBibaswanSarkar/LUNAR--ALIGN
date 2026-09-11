import numpy as np


def select_spatial_matches(points1, points2, rows=4, cols=4, max_per_cell=20):
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    if len(points1) == 0 or len(points2) == 0:
        return {"points1": np.empty((0, 2), dtype=float), "points2": np.empty((0, 2), dtype=float), "coverage": 0.0}

    H, W = points1[:, 1].max() + 1, points1[:, 0].max() + 1
    if H <= 0 or W <= 0:
        return {"points1": points1, "points2": points2, "coverage": 0.0}

    grid = np.zeros((rows, cols), dtype=int)
    accepted = []
    for p1, p2 in zip(points1, points2):
        r = min(rows - 1, int(p1[1] / max(H / rows, 1)))
        c = min(cols - 1, int(p1[0] / max(W / cols, 1)))
        if grid[r, c] < max_per_cell:
            grid[r, c] += 1
            accepted.append((p1, p2))

    sel_p1 = np.array([p[0] for p in accepted], dtype=float)
    sel_p2 = np.array([p[1] for p in accepted], dtype=float)
    occupied = np.count_nonzero(grid > 0)
    coverage = occupied / (rows * cols)
    return {"points1": sel_p1, "points2": sel_p2, "coverage": float(coverage)}
