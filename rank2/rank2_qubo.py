#!/usr/bin/env python3
# rank2_qubo.py — Shapely arrangement + Matplotlib visualization + dual zonotope (exact 2D)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set, Iterable
import math
import numpy as np
from shapely.geometry import LineString, Polygon, Point, box
from shapely.ops import unary_union, polygonize

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


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

    Arrangement:
      1) clip each predicate line a x + b y + c = 0 to window edges,
      2) add window edges,
      3) polygonize linework → bounded faces,
      4) one interior probe per face → support → evaluate objective.

    Dual zonotope (exact):
      Z = sum_i [-g_i, g_i] with generators g_i = (a_i, b_i).
      Construct boundary by a support-function sweep over θ∈[0,2π):
        events at θ = angle(g_i) ± π/2 (mod 2π); when crossing, flip sign of ⟨u,g_i⟩,
        update support point p ← p - 2 s_i g_i. Sequence of p’s are the vertices.
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
        self.lines = lines  # store predicate segments
        self.window_edges = _box_edges(self.window)

        # Split & polygonize to faces (bounded)
        merged = unary_union(self.lines + self.window_edges)
        faces_all = list(polygonize(merged))  # iterable of Polygon
        inner = self.window.buffer(-1e-9)
        self.faces: List[Polygon] = [
            p for p in faces_all if p.is_valid and inner.contains(p.representative_point())
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
            p: Point = poly.representative_point()
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

    # ---------- Visualization: arrangement ----------

    def plot(
        self,
        ax: Axes | None = None,
        *,
        show_faces: bool = False,
        face_alpha: float = 0.15,
        show_window: bool = True,
        show_probes: bool = False,
        probe_size: int = 18,
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        if show_faces:
            for poly in self.faces:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=face_alpha, linewidth=0)

        for seg in self.lines:
            (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
            ax.plot([x1, x2], [y1, y2], linewidth=1)

        if show_window:
            for edge in self.window_edges:
                (x1, y1), (x2, y2) = edge.coords[0], edge.coords[-1]
                ax.plot([x1, x2], [y1, y2], linewidth=1)

        if show_probes:
            for poly in self.faces:
                p = poly.representative_point()
                ax.plot(p.x, p.y, marker="o", markersize=probe_size / 3, linestyle="None")

        x_min, y_min, x_max, y_max = self.window.bounds
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\xi_1$"); ax.set_ylabel(r"$\xi_2$")
        ax.set_title("Rank-2 QUBO: predicate arrangement in $\mathbb{R}^2$")
        return ax

    # ---------- Visualization: dual zonotope ----------

    def zonotope_vertices(self, normalize: bool = False) -> np.ndarray:
        """
        Exact vertices of Z = sum_i [-g_i, g_i], g_i=(a_i,b_i), via support-function sweep.

        normalize=True uses unit generators g_i / ||g_i|| (combinatorics unchanged;
        visualization rescales edges uniformly).
        Returns an array of shape (k, 2) in CCW order (k ≤ 2m).
        """
        gens = np.column_stack([self.a, self.b]).astype(float)
        if normalize:
            norms = np.linalg.norm(gens, axis=1)
            mask = norms > 0
            gens[mask] = gens[mask] / norms[mask, None]

        m = gens.shape[0]
        if m == 0:
            return np.zeros((0, 2))

        # Critical angles where sign flips: τ_i = angle(g_i) + π/2  (and +π)
        phi = np.arctan2(gens[:, 1], gens[:, 0])  # angle of generator
        tau = (phi + 0.5 * np.pi) % (2 * np.pi)

        events: List[Tuple[float, int]] = []
        for i in range(m):
            t0 = float(tau[i] % (2 * np.pi))
            t1 = float((tau[i] + np.pi) % (2 * np.pi))
            events.append((t0, i))
            events.append((t1, i))
        events.sort(key=lambda t: t[0])

        # Initial direction u(θ0): pick just before the first event
        θ0 = events[0][0] - 1e-9
        if θ0 < 0:
            θ0 += 2 * np.pi
        u = np.array([math.cos(θ0), math.sin(θ0)], dtype=float)

        # Initial signs and support point
        dots = gens @ u
        signs = np.where(dots >= 0.0, 1.0, -1.0)
        p = (signs[:, None] * gens).sum(axis=0)

        verts: List[np.ndarray] = [p.copy()]
        for θ, i in events:
            # Crossing an event for generator i flips its sign
            s_old = signs[i]
            signs[i] = -s_old
            # Update support point: p_new = p - 2*s_old*g_i
            p = p - 2.0 * s_old * gens[i]
            # Record vertex at event
            if len(verts) == 0 or np.linalg.norm(p - verts[-1]) > 1e-12:
                verts.append(p.copy())

        # Close polygon (ensure first != last numerically)
        if np.linalg.norm(verts[0] - verts[-1]) > 1e-12:
            verts.append(verts[0].copy())

        # Remove the closing duplicate for return (matplotlib can close itself)
        verts = verts[:-1]
        return np.asarray(verts)

    def plot_zonotope(
        self,
        ax: Axes | None = None,
        *,
        normalize: bool = False,
        fill: bool = True,
        alpha: float = 0.2,
        show_edges: bool = True,
        show_generators: bool = False,
        gen_scale: float = 1.0,
    ) -> Axes:
        """
        Plot the dual zonotope Z = sum_i [-g_i, g_i], g_i=(a_i,b_i).

        normalize=True draws with unit generators; fill controls polygon fill;
        show_generators draws generator vectors from the origin.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        V = self.zonotope_vertices(normalize=normalize)
        if V.shape[0] >= 3:
            if fill:
                ax.fill(V[:, 0], V[:, 1], alpha=alpha, linewidth=0)
            if show_edges:
                # close loop
                X = np.append(V[:, 0], V[0, 0]); Y = np.append(V[:, 1], V[0, 1])
                ax.plot(X, Y, linewidth=1)

        if show_generators:
            gens = np.column_stack([self.a, self.b]).astype(float)
            if normalize:
                norms = np.linalg.norm(gens, axis=1)
                mask = norms > 0
                gens[mask] = gens[mask] / norms[mask, None]
            for g in gens:
                ax.arrow(0.0, 0.0, gen_scale * g[0], gen_scale * g[1],
                         head_width=0.0, head_length=0.0, length_includes_head=True, linewidth=1)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title("Dual zonotope of predicate normals")
        return ax

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


def _demo():
    rng = np.random.default_rng(7)
    n = 10
    v1 = rng.standard_normal(n)
    v2 = rng.standard_normal(n)
    d = rng.normal(scale=0.2, size=n)

    qubo = Rank2QUBO(v1, v2, d, bbox_scale=8.0)
    print("bounded faces:", len(qubo.faces))
    best_val, best_x, probe = qubo.solve_by_faces()
    print("best value:", best_val)
    print("probe ξ:", probe)
    print("support size:", int(best_x.sum()))

    # Arrangement
    ax1 = qubo.plot(show_faces=True, show_window=True, show_probes=True)
    plt.show()

    # Dual zonotope (raw vs normalized generators)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    qubo.plot_zonotope(ax=axs[0], normalize=False, fill=True, alpha=0.25, show_edges=True, show_generators=True)
    axs[0].set_title("Zonotope (raw generators)")
    qubo.plot_zonotope(ax=axs[1], normalize=True, fill=True, alpha=0.25, show_edges=True, show_generators=True)
    axs[1].set_title("Zonotope (normalized generators)")
    plt.show()


if __name__ == "__main__":
    _demo()
