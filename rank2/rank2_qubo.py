#!/usr/bin/env python3
# rank2_qubo.py — Simple Shapely arrangement for rank-2 QUBO

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set
import numpy as np
from shapely.geometry import LineString, Polygon, Point, box
from shapely.ops import unary_union, polygonize


@dataclass(frozen=True)
class Predicate:
    a: float
    b: float
    c: float
    index: int


class Rank2QUBO:
    """
    f(x) = (v1^T x)^2 + (v2^T x)^2 + d^T x,  x ∈ {0,1}^n
    √λ already absorbed into v1, v2.

    Build arrangement inside a big square window:
      1) clip each predicate line a x + b y + c = 0 to window edges,
      2) add window edges,
      3) polygonize linework → bounded faces,
      4) one interior probe per face → support → evaluate objective.
    """

    def __init__(
        self, v1: np.ndarray, v2: np.ndarray, d: np.ndarray, bbox_scale: float = 10.0
    ):
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        d = np.asarray(d, dtype=float)
        if not (v1.ndim == v2.ndim == d.ndim == 1 and v1.shape == v2.shape == d.shape):
            raise ValueError("v1, v2, d must be 1D arrays of equal length")
        self.n = v1.size
        self.v1, self.v2, self.d = v1, v2, d

        # Predicates: g_i^+(ξ) = a_i ξ1 + b_i ξ2 + c_i
        self.a = 2.0 * v1
        self.b = 2.0 * v2
        self.c = (v1 * v1) + (v2 * v2) + d
        self.predicates: List[Predicate] = [
            Predicate(float(self.a[i]), float(self.b[i]), float(self.c[i]), i)
            for i in range(self.n)
        ]

        # Window and linework
        self.window: Polygon = self._choose_window(bbox_scale)
        lines = self._lines_clipped_to_window(self.window)
        linework = lines + _box_edges(self.window)

        # Split & polygonize to faces (bounded)
        merged = unary_union(linework)
        faces_all = list(polygonize(merged))  # iterable of Polygon
        # Keep faces whose representative point lies in a shrunken window (drop outer annulus)
        inner = self.window.buffer(
            -1e-9
        )  # Tiny shrink to avoid boundary-touching artifacts
        self.faces: List[Polygon] = [
            p for p in faces_all if inner.contains(p.representative_point())
        ]

    # ---------- Predicates & objective ----------

    def predicate_coeffs(self, i: int) -> Tuple[float, float, float]:
        return float(self.a[i]), float(self.b[i]), float(self.c[i])

    def predicate_values(self, xi1: float, xi2: float) -> np.ndarray:
        return self.a * xi1 + self.b * xi2 + self.c

    def support_from_probe(self, xi1: float, xi2: float) -> np.ndarray:
        return (self.predicate_values(xi1, xi2) > 0.0).astype(int)

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise ValueError("x shape mismatch")
        xi1 = float(self.v1 @ x)
        xi2 = float(self.v2 @ x)
        return (xi1 * xi1) + (xi2 * xi2) + float(self.d @ x)

    # ---------- Solve by face sampling ----------

    def solve_by_faces(self) -> Tuple[float, np.ndarray, Tuple[float, float]]:
        best_val = -np.inf
        best_x = np.zeros(self.n, dtype=int)
        best_probe = (0.0, 0.0)
        seen: Set[Tuple[int, ...]] = set()

        for poly in self.faces:
            p: Point = poly.representative_point()  # guaranteed inside poly
            xi1, xi2 = float(p.x), float(p.y)
            x = self.support_from_probe(xi1, xi2)
            key = tuple(int(t) for t in x)
            if key in seen:
                continue
            seen.add(key)
            val = self.objective(x)
            if val > best_val:
                best_val, best_x, best_probe = val, x, (xi1, xi2)
        return best_val, best_x, best_probe

    # ---------- Internals ----------

    def _choose_window(self, bbox_scale: float) -> Polygon:
        eps = 1e-12
        cand = []
        for a, b, c in zip(self.a, self.b, self.c):
            if abs(a) > eps:
                cand.append(abs(c / a))
            if abs(b) > eps:
                cand.append(abs(c / b))
        S = bbox_scale * max(1.0, max(cand) if cand else 1.0)
        return box(-S, -S, +S, +S)

    def _lines_clipped_to_window(self, win: Polygon) -> List[LineString]:
        eps = 1e-12
        x_min, y_min, x_max, y_max = win.bounds
        segs: List[LineString] = []
        for a, b, c in zip(self.a, self.b, self.c):
            if abs(a) < eps and abs(b) < eps:
                continue
            pts = []
            if abs(a) > eps:
                for y in (y_min, y_max):
                    x = -(b * y + c) / a
                    if x_min - 1e-9 <= x <= x_max + 1e-9:
                        pts.append((x, y))
            if abs(b) > eps:
                for x in (x_min, x_max):
                    y = -(a * x + c) / b
                    if y_min - 1e-9 <= y <= y_max + 1e-9:
                        pts.append((x, y))
            if len(pts) < 2:
                continue
            uniq = list(
                {(round(px, 12), round(py, 12)): (px, py) for (px, py) in pts}.values()
            )
            if len(uniq) < 2:
                continue
            # Pick two farthest points across the window
            d2_best, pair = -1.0, (uniq[0], uniq[1])
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    dx = uniq[i][0] - uniq[j][0]
                    dy = uniq[i][1] - uniq[j][1]
                    d2 = dx * dx + dy * dy
                    if d2 > d2_best:
                        d2_best, pair = d2, (uniq[i], uniq[j])
            if d2_best > eps:
                segs.append(LineString([pair[0], pair[1]]))
        return segs


def _box_edges(win: Polygon) -> List[LineString]:
    x_min, y_min, x_max, y_max = win.bounds
    return [
        LineString([(x_min, y_min), (x_max, y_min)]),
        LineString([(x_max, y_min), (x_max, y_max)]),
        LineString([(x_max, y_max), (x_min, y_max)]),
        LineString([(x_min, y_max), (x_min, y_min)]),
    ]


def main():
    # --- Quick smoke test ---
    rng = np.random.default_rng()#12)
    n = 35
    v1 = rng.standard_normal(n)
    v2 = rng.standard_normal(n)
    d = rng.normal(scale=0.2, size=n)

    qubo = Rank2QUBO(v1, v2, d, bbox_scale=8.0)
    print("bounded faces:", len(qubo.faces))
    best_val, best_x, probe = qubo.solve_by_faces()
    print("best value:", best_val)
    print("probe ξ:", probe)
    print("support size:", int(best_x.sum()))


if __name__ == "__main__":
    main()
