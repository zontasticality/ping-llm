"""Vivaldi Network Coordinate System baseline for RTT prediction."""

import numpy as np


def fit_vivaldi(measurements, dim=4, n_epochs=5, cc=0.25, ce=0.5):
    """
    Fit Vivaldi coordinates from RTT measurements.

    Args:
        measurements: list of (src_key, dst_key, rtt_ms)
        dim: coordinate dimension
        n_epochs: passes over data
        cc: coordinate correction factor
        ce: error estimate EMA factor

    Returns:
        dict mapping ip_key -> (coord, height, error_estimate)
    """
    coords = {}
    heights = {}
    errors = {}
    rng = np.random.RandomState(42)

    def ensure(key):
        if key not in coords:
            coords[key] = rng.normal(0, 0.01, dim)
            heights[key] = 0.0
            errors[key] = 1.0

    for epoch in range(n_epochs):
        order = list(range(len(measurements)))
        rng.shuffle(order)
        total_rel_err = 0.0
        count = 0

        for idx in order:
            src, dst, rtt = measurements[idx]
            if rtt <= 0:
                continue

            ensure(src)
            ensure(dst)

            diff = coords[src] - coords[dst]
            dist = np.linalg.norm(diff)
            pred = dist + heights[src] + heights[dst]
            pred = max(pred, 0.001)

            err = rtt - pred
            rel_err = abs(err) / rtt
            total_rel_err += rel_err
            count += 1

            e_sum = errors[src] + errors[dst]
            if e_sum < 1e-10:
                continue

            w_src = errors[src] / e_sum
            w_dst = errors[dst] / e_sum
            errors[src] = ce * w_src * rel_err + (1 - ce * w_src) * errors[src]
            errors[dst] = ce * w_dst * rel_err + (1 - ce * w_dst) * errors[dst]

            delta_src = cc * w_src
            delta_dst = cc * w_dst

            if dist > 1e-10:
                unit = diff / dist
            else:
                unit = rng.normal(0, 1, dim)
                unit /= np.linalg.norm(unit) + 1e-10

            # unit points from dst toward src.
            # err > 0 (underestimate): push apart → src moves in +unit direction
            coords[src] += delta_src * err * unit
            heights[src] += delta_src * err
            heights[src] = max(heights[src], 0.0)

            coords[dst] -= delta_dst * err * unit
            heights[dst] += delta_dst * err
            heights[dst] = max(heights[dst], 0.0)

        if count > 0 and (epoch == 0 or epoch == n_epochs - 1):
            print(f"    Vivaldi epoch {epoch+1}/{n_epochs}: "
                  f"avg relative error={total_rel_err / count:.4f}")

    return {k: (coords[k], heights[k], errors[k]) for k in coords}


def predict_vivaldi(coords_dict, src_key, dst_key):
    """Predict RTT from Vivaldi coordinates. Returns None if either IP unseen."""
    if src_key not in coords_dict or dst_key not in coords_dict:
        return None
    src_coord, src_height, _ = coords_dict[src_key]
    dst_coord, dst_height, _ = coords_dict[dst_key]
    return max(float(np.linalg.norm(src_coord - dst_coord) + src_height + dst_height), 0.001)
