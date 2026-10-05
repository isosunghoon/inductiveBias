"""Circular constellation explorer.

주어진 점 개수 N에 대해 "원형(circular) 컨스텔레이션"을 만든다.
- hex     : 삼각(육방) 격자를 원으로 잘라낸 것  <- 첨부 이미지 형태
- square  : 정사각 격자를 원으로 잘라낸 것
- rings   : 동심원 링 배치 (APSK 스타일)
- spiral  : Fermat/해바라기 나선 (phyllotaxis)

사용법
    python temp.py                     # 슬라이더 GUI (TkAgg/Qt 등 GUI 백엔드 필요)
    python temp.py --grid 7 19 37 61   # 여러 N을 한 장에 비교해 PNG로 저장
    python temp.py --gif               # N을 훑는 애니메이션 GIF 저장 (headless OK)
"""

from __future__ import annotations

import argparse
import math

import numpy as np

SQRT3 = math.sqrt(3.0)


# --------------------------------------------------------------------------- #
# 점 생성
# --------------------------------------------------------------------------- #
def _hex_norm(pts: np.ndarray) -> np.ndarray:
    """삼각 격자의 육각 노름. 같은 값을 갖는 점들이 하나의 육각 껍질을 이룬다."""
    x, y = pts[:, 0], pts[:, 1]
    return np.maximum.reduce([np.abs(x - y / SQRT3),
                              np.abs(2.0 * y / SQRT3),
                              np.abs(x + y / SQRT3)])


def _take_closest(pts: np.ndarray, n: int, cost: np.ndarray | None = None) -> np.ndarray:
    """비용이 작은 순서로 n개를 고른다. 동률이면 반지름 -> 각도로 tie-break."""
    r = np.hypot(pts[:, 0], pts[:, 1])
    th = np.arctan2(pts[:, 1], pts[:, 0])
    if cost is None:
        cost = r
    order = np.lexsort((th, np.round(r, 9), np.round(cost, 9)))
    return pts[order[:n]]


def _round_shells(pts: np.ndarray, t: float) -> np.ndarray:
    """각 육각 껍질의 반지름을 그 껍질의 평균 반지름 쪽으로 t만큼 당긴다.

    t=0이면 원래 격자(껍질 경계가 육각형), t=1이면 모든 껍질이 정확한 원이 된다.
    각도는 건드리지 않으므로 육각 배열의 결은 남고 윤곽만 둥글어진다.
    """
    if t <= 0 or len(pts) == 0:
        return pts
    r = np.hypot(pts[:, 0], pts[:, 1])
    shell = np.round(_hex_norm(pts), 6)
    out = pts.copy()
    for s in np.unique(shell):
        m = shell == s
        if s == 0:
            continue
        target = r[m].mean()
        scale = ((1.0 - t) * r[m] + t * target) / np.where(r[m] > 0, r[m], 1.0)
        out[m] = pts[m] * scale[:, None]
    return out


# shape -> (껍질 선택 기준이 육각 노름인가, 껍질 원형화 강도 t)
_SHAPE_SPEC = {"hexagon": (True, 0.0), "circle": (False, 0.0),
               "round": (True, 0.6), "rounder": (True, 1.0)}


def hex_constellation(n: int, center: str = "point", shape: str = "circle") -> np.ndarray:
    """삼각 격자를 shape 경계로 잘라낸 constellation (최밀 충전 + shaping)."""
    m = int(math.ceil(math.sqrt(n))) + 4
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
    x = i.ravel() + 0.5 * j.ravel()          # a1 = (1, 0)
    y = (SQRT3 / 2.0) * j.ravel()            # a2 = (1/2, sqrt(3)/2)
    pts = np.column_stack([x, y])
    if center == "hole":                     # 삼각형 무게중심 기준 (3회 대칭)
        pts -= np.array([0.5, SQRT3 / 6.0])
    elif center == "bond":                   # 격자변 중점 기준 (2회 대칭)
        pts -= np.array([0.5, 0.0])
    use_hex, t = _SHAPE_SPEC[shape]
    pts = _take_closest(pts, n, _hex_norm(pts) if use_hex else None)
    return _round_shells(pts, t)


def square_constellation(n: int, center: str = "point", shape: str = "circle") -> np.ndarray:
    m = int(math.ceil(math.sqrt(n))) + 3
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
    pts = np.column_stack([i.ravel().astype(float), j.ravel().astype(float)])
    if center in ("hole", "bond"):
        pts -= np.array([0.5, 0.5])
    return _take_closest(pts, n)


def ring_constellation(n: int, center: str = "point", shape: str = "circle") -> np.ndarray:
    """동심원 링 배치. 링 간격과 링 위 점 간격이 비슷해지도록 점 수를 배분한다."""
    if n <= 0:
        return np.zeros((0, 2))
    has_center = center == "point"
    left = n - 1 if has_center else n
    pts = [np.zeros((1, 2))] if has_center else []

    k = 1
    while left > 0:
        # 반지름 k인 링의 둘레(2*pi*k)를 간격 1로 채운다.
        cnt = min(left, max(1, int(round(2.0 * math.pi * k))))
        if left - cnt < 0.4 * cnt:           # 마지막 자투리는 이 링에 합친다
            cnt = left
        ang = np.arange(cnt) * (2.0 * math.pi / cnt)
        ang += (math.pi / cnt) * (k % 2)     # 링마다 반 칸 회전
        pts.append(np.column_stack([k * np.cos(ang), k * np.sin(ang)]))
        left -= cnt
        k += 1
    return np.vstack(pts)[:n]


def spiral_constellation(n: int, center: str = "point", shape: str = "circle") -> np.ndarray:
    """황금각 나선 (해바라기 배치): 격자가 아니면서도 국소적으로 육방 배열."""
    idx = np.arange(n, dtype=float)
    if center != "point":
        idx += 0.5
    r = np.sqrt(idx)
    th = idx * math.pi * (3.0 - math.sqrt(5.0))
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


MODES = {
    "hex": hex_constellation,
    "square": square_constellation,
    "rings": ring_constellation,
    "spiral": spiral_constellation,
}
CENTERS = ("point", "hole", "bond")
# 경계 모양: 육각형 <- 기본 원형 -> 더 둥글게
SHAPES = ("hexagon", "circle", "round", "rounder")


def make_constellation(n: int, mode: str = "hex", center: str = "point",
                       normalize: bool = True, shape: str = "circle") -> np.ndarray:
    pts = MODES[mode](int(n), center, shape)
    if normalize and len(pts):
        p = np.mean(np.sum(pts ** 2, axis=1))      # 평균 전력
        if p > 0:
            pts = pts / math.sqrt(p)               # 단위 평균 전력으로 정규화
    return pts


# --------------------------------------------------------------------------- #
# 지표
# --------------------------------------------------------------------------- #
def metrics(pts: np.ndarray) -> dict:
    """constellation 품질 지표. d_min은 평균전력으로 정규화해 비교 가능하게 만든다."""
    n = len(pts)
    out = {"N": n, "d_min": float("nan"), "R": float("nan"),
           "papr": float("nan"), "fill": float("nan")}
    if n == 0:
        return out
    r2 = np.sum(pts ** 2, axis=1)
    pwr = float(np.mean(r2))
    out["R"] = math.sqrt(float(r2.max()))
    out["papr"] = float(r2.max() / pwr) if pwr > 0 else float("nan")
    if n > 1:
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                     pts[:, None, 1] - pts[None, :, 1])
        np.fill_diagonal(d, np.inf)
        dmin = float(d.min())
        out["d_min"] = dmin / math.sqrt(pwr) if pwr > 0 else dmin
        # 반지름 R 원 안에 지름 d_min 원판 N개를 넣었을 때의 면적 점유율
        if out["R"] > 0:
            out["fill"] = n * (dmin / 2.0) ** 2 / (out["R"] + dmin / 2.0) ** 2
    return out


def fmt_metrics(m: dict) -> str:
    return (f"N = {m['N']}   d_min/√P̄ = {m['d_min']:.4f}   "
            f"PAPR = {m['papr']:.3f}   packing = {m['fill']:.3f}")


# --------------------------------------------------------------------------- #
# 그리기
# --------------------------------------------------------------------------- #
DOT = "#E08A3C"
ACC = "#F0C9A8"


def draw(ax, pts: np.ndarray, *, show_circle=True, show_axes=True,
         lim=None, size=None, title=None, boundary="circle"):
    """점이 서로 겹치지도 떨어지지도 않게 마커 크기를 최근접 거리에 맞춰 그린다."""
    from matplotlib.patches import Circle

    ax.clear()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    d = _nn_dist(pts)
    if lim is None:
        lim = (np.abs(pts).max() + d) * 1.05 if len(pts) else 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    if show_axes:
        ax.axhline(0, color=ACC, lw=0.8, zorder=0)
        ax.axvline(0, color=ACC, lw=0.8, zorder=0)
    if show_circle and len(pts):
        R = float(np.hypot(pts[:, 0], pts[:, 1]).max())
        if boundary == "hexagon":
            ax.add_patch(hexagon_patch(R + d / 2, fill=False, ec=ACC, lw=1.2,
                                       ls="--", zorder=0))
        else:
            ax.add_patch(Circle((0, 0), R + d / 2, fill=False, ec=ACC, lw=1.2,
                                ls="--", zorder=0))

    if size is None:
        # data 단위 -> points 환산 후, 지름이 최근접 거리의 62%가 되게 한다.
        width_pt = ax.get_window_extent().width / ax.figure.dpi * 72.0
        size = max((0.62 * d * width_pt / (2.0 * lim)) ** 2, 1.5)
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=size, c=DOT, edgecolors="none", zorder=2)
    if title:
        ax.set_title(title, fontsize=10, color="#444")


def hexagon_patch(circumradius: float, **kw):
    """격자와 방향이 맞는 정육각형 (꼭짓점이 theta = 0, 60, ... 방향)."""
    from matplotlib.patches import Polygon
    th = np.arange(6) * (math.pi / 3.0)
    return Polygon(np.column_stack([circumradius * np.cos(th),
                                    circumradius * np.sin(th)]), closed=True, **kw)


def _nn_dist(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 1.0
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return float(d.min())


# --------------------------------------------------------------------------- #
# 인터랙티브 (슬라이더)
# --------------------------------------------------------------------------- #
def interactive(n0: int, mode: str, center: str, nmax: int, normalize: bool,
                shape: str = "circle"):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

    fig = plt.figure(figsize=(9.5, 7.2))
    fig.canvas.manager.set_window_title("Circular constellation")
    ax = fig.add_axes([0.30, 0.16, 0.66, 0.78])

    ax_n = fig.add_axes([0.30, 0.06, 0.55, 0.03])
    s_n = Slider(ax_n, "points N", 1, nmax, valinit=n0, valstep=1)

    ax_mode = fig.add_axes([0.03, 0.66, 0.20, 0.22]); ax_mode.set_title("lattice", fontsize=9)
    r_mode = RadioButtons(ax_mode, tuple(MODES), active=list(MODES).index(mode))

    ax_ctr = fig.add_axes([0.03, 0.46, 0.20, 0.16]); ax_ctr.set_title("center", fontsize=9)
    r_ctr = RadioButtons(ax_ctr, CENTERS, active=CENTERS.index(center))

    ax_shape = fig.add_axes([0.03, 0.26, 0.20, 0.18]); ax_shape.set_title("boundary", fontsize=9)
    r_shape = RadioButtons(ax_shape, SHAPES, active=SHAPES.index(shape))

    ax_chk = fig.add_axes([0.03, 0.04, 0.20, 0.18]); ax_chk.set_title("view", fontsize=9)
    c_chk = CheckButtons(ax_chk, ["unit power", "circle", "axes", "fixed zoom"],
                         [normalize, True, True, True])

    ax_btn = fig.add_axes([0.86, 0.05, 0.11, 0.045])
    b_save = Button(ax_btn, "save png")

    for a in (ax_mode, ax_ctr, ax_shape, ax_chk):
        for s in a.spines.values():
            s.set_visible(False)

    state = {}

    def redraw(_=None):
        n = int(s_n.val)
        norm, circ, axes_on, fixed = c_chk.get_status()
        pts = make_constellation(n, r_mode.value_selected, r_ctr.value_selected,
                                 norm, r_shape.value_selected)
        lim = 2.2 if (norm and fixed) else None
        draw(ax, pts, show_circle=circ, show_axes=axes_on, lim=lim,
             boundary=r_shape.value_selected)
        m = metrics(pts)
        ax.set_title(fmt_metrics(m), fontsize=10, color="#444")
        state["pts"] = pts
        fig.canvas.draw_idle()

    def save(_):
        n = int(s_n.val)
        name = f"constellation_{r_mode.value_selected}_{r_shape.value_selected}_{n}.png"
        fig.savefig(name, dpi=160, bbox_inches="tight", facecolor="white")
        print("saved", name)

    s_n.on_changed(redraw)
    r_mode.on_clicked(redraw)
    r_ctr.on_clicked(redraw)
    r_shape.on_clicked(redraw)
    c_chk.on_clicked(redraw)
    b_save.on_clicked(save)
    redraw()
    plt.show()


# --------------------------------------------------------------------------- #
# headless 출력
# --------------------------------------------------------------------------- #
def save_grid(ns, mode, center, normalize, out, shape="circle"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(len(ns), 4)
    rows = int(math.ceil(len(ns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for a in axes[len(ns):]:
        a.axis("off")
    for a, n in zip(axes, ns):
        pts = make_constellation(n, mode, center, normalize, shape)
        m = metrics(pts)
        draw(a, pts, lim=2.2 if normalize else None, boundary=shape,
             title=f"N={n}   d_min={m['d_min']:.3f}\nPAPR={m['papr']:.2f}")
    fig.suptitle(f"circular constellation — {mode} lattice, "
                 f"boundary={shape}, center={center}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    print("saved", out)


def save_gif(nmax, mode, center, normalize, out, step=1, fps=8, shape="circle"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ns = list(range(1, nmax + 1, step))
    fig, ax = plt.subplots(figsize=(5.2, 5.4))

    def frame(n):
        pts = make_constellation(n, mode, center, normalize, shape)
        draw(ax, pts, lim=2.2 if normalize else None, boundary=shape)
        ax.set_title(fmt_metrics(metrics(pts)), fontsize=9, color="#444")

    anim = FuncAnimation(fig, frame, frames=ns, interval=1000 / fps)
    anim.save(out, writer=PillowWriter(fps=fps))
    print("saved", out)


def centered_hex_number(k: int) -> int:
    """k개 껍질이 꽉 찬 육각 배열의 점 개수: 1, 7, 19, 37, 61, 91, 127, 169, ..."""
    return 3 * k * (k + 1) + 1


def best_hex_count(target: int) -> tuple[int, int]:
    """target에 가장 가까운, 껍질이 딱 떨어지는(= '야무진') 점 개수와 껍질 수 k."""
    k = max(range(0, 200), key=lambda k: -abs(centered_hex_number(k) - target))
    return centered_hex_number(k), k


def layout_figure(side: float, margin: float, dia: float, target: int, out: str,
                  unit: str = "cm"):
    """한 변 side인 정육각형 안에, 테두리 margin을 남기고 지름 dia인 원을
    target개 근처의 '야무진'(껍질이 딱 떨어지는) 육각 배열로 배치한 도면."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    n, k = best_hex_count(target)
    r = dia / 2.0
    # 정육각형을 t만큼 안쪽으로(변에 수직으로) 줄이면 외접반지름이 t*2/sqrt(3)만큼 준다.
    inset = lambda t: side - t * 2.0 / SQRT3
    r_centers = inset(margin + r)          # 원 '중심'이 놓일 수 있는 육각형
    pitch = r_centers / k if k else 0.0

    pts = hex_constellation(n, shape="hexagon") * pitch   # 격자 간격 1 -> pitch

    fig, ax = plt.subplots(figsize=(7.6, 8.0))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(hexagon_patch(side, fill=False, ec="#333", lw=2.0, zorder=3))
    ax.add_patch(hexagon_patch(inset(margin), fill=False, ec="#B0B0B0", lw=1.0,
                               ls="--", zorder=3))
    ax.add_patch(hexagon_patch(r_centers, fill=False, ec=ACC, lw=1.0, ls=":", zorder=3))

    # 실제 원(지름 dia)은 한 변의 0.1%도 안 돼 점으로도 안 보인다 -> 배율을 곱해 그린다.
    mag = max(1.0, 0.20 * pitch / r)
    for x, y in pts:
        ax.add_patch(Circle((x, y), r * mag, fc=DOT, ec="none", zorder=4))

    lim = side * 1.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim * 1.02)
    ax.set_title(f"hexagon side {side:g}{unit} / margin {margin:g}{unit} / "
                 f"circle Ø{dia:g}{unit}  →  N = {n}", fontsize=13)
    info = (f"N = {n} points: {k} complete hexagonal shells (perfect 6-fold symmetry); "
            f"nearest snug counts to {target} are {centered_hex_number(k - 1)} and "
            f"{centered_hex_number(k + 1)}\n"
            f"pitch (center-to-center) = {pitch:.3f} {unit}    "
            f"gap between circle edges = {pitch - dia:.3f} {unit}\n"
            f"outer shell corners sit exactly {margin:g} {unit} inside the border "
            f"(solid = hexagon, dashed = margin line, dotted = center locus)\n"
            f"circles drawn {mag:.0f}x their true size — a real "
            f"{dia:g} {unit} circle is invisible at this scale\n"
            f"lattice vector a1 = ({pitch:.3f}, 0), a2 = ({pitch / 2:.3f}, "
            f"{pitch * SQRT3 / 2:.3f}) {unit}")
    # 치수: 아래 변 길이와 오른쪽 꼭짓점의 마진
    y_bot = -side * SQRT3 / 2.0
    ax.annotate("", xy=(-side / 2, y_bot - side * 0.06),
                xytext=(side / 2, y_bot - side * 0.06),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=1.0))
    ax.text(0, y_bot - side * 0.10, f"side = {side:g} {unit}", ha="center",
            va="top", fontsize=10, color="#444")
    ax.annotate(f"margin {margin:g} {unit}", xy=(side, 0),
                xytext=(side * 0.80, side * 0.34), fontsize=9.5, color="#777",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.9))
    fig.text(0.5, 0.015, info, ha="center", va="bottom", fontsize=9.5, color="#444")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"saved {out}  (N={n}, k={k}, pitch={pitch:.4f}{unit})")


def blueprint(side: float, margin: float, radius: float, n: int, out: str,
              unit: str = "cm", scale: float = 100.0, detail_scale: float = 0.0):
    """실측 축척 도면 1장.

    margin = 가장 바깥 원의 '가장자리'에서 육각형 경계까지의 수직 거리.
    원은 확대 없이 실제 크기(반지름 radius)로 그리고, 종이 위 치수가
    1:scale과 정확히 일치하도록 figure 크기와 축 범위를 맞춘다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    k = round((math.sqrt(12 * n - 3) - 3) / 6)          # n = 3k(k+1)+1 의 역
    if centered_hex_number(k) != n:
        raise SystemExit(f"N={n}은 껍질이 딱 떨어지지 않는다. 가까운 값: "
                         f"{centered_hex_number(k)}, {centered_hex_number(k + 1)}")
    # 정육각형을 변에 수직으로 t만큼 줄이면 외접반지름이 t*2/sqrt(3)만큼 준다.
    inset = lambda t: side - t * 2.0 / SQRT3
    r_margin = inset(margin)                    # 마진선
    r_centers = inset(margin + radius)          # 원 중심이 놓이는 육각형
    pitch = r_centers / k
    row = pitch * SQRT3 / 2.0
    pts = hex_constellation(n, shape="hexagon") * pitch

    # ---- 종이 배치 (inch). 본도는 1:scale로 정확히 맞춘다 ------------------ #
    unit_cm = 1.0 if unit == "cm" else 0.1          # 도면 단위 -> cm
    cm2in = lambda v: v * unit_cm / scale / 2.54    # 도면 단위 -> 종이 inch
    # 상세도: 원 + 마진이 다 들어오는 창을 잡고, 종이 폭 ~3.3in이 되는 나이스한 축척
    zw = zh = (radius + margin) * 1.9
    if not detail_scale:
        want = 2 * zw * unit_cm / (3.3 * 2.54)
        detail_scale = next(s for s in (1, 2, 2.5, 5, 10, 20, 25, 50, 100)
                            if s >= want - 1e-9)
    x_hw = side * 1.10
    y_hi = side * SQRT3 / 2 * 1.10
    y_lo = -side * SQRT3 / 2 * 1.10 - side * 0.42          # 아래쪽 치수선 띠
    draw_w, draw_h = cm2in(2 * x_hw), cm2in(y_hi - y_lo)
    det_w = 2 * zw * unit_cm / detail_scale / 2.54         # 상세도 폭
    col_w = max(det_w, 3.3)
    pad_l, pad_r, pad_t, gap, tb_h = 0.55, 0.5, 0.5, 0.45, 1.75
    w_in = pad_l + draw_w + gap + col_w + pad_r
    h_in = pad_t + draw_h + 0.25 + tb_h

    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([pad_l / w_in, (0.25 + tb_h) / h_in, draw_w / w_in, draw_h / h_in])
    ax.set_xlim(-x_hw, x_hw)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- 본도 ------------------------------------------------------------- #
    ax.add_patch(hexagon_patch(side, fill=False, ec="#111", lw=1.6, zorder=5))
    ax.add_patch(hexagon_patch(r_margin, fill=False, ec="#C05A2B", lw=0.7,
                               ls=(0, (7, 4)), zorder=4))
    ax.add_patch(hexagon_patch(r_centers, fill=False, ec="#9AA0A6", lw=0.5,
                               ls=(0, (1, 3)), zorder=4))
    # 위치 심볼(고정 크기, 축척 무관) + 실제 크기 원
    paper_cm = 2 * radius * unit_cm / scale      # 종이 위 원 지름 (cm)
    ring = 0.09 * scale / unit_cm if paper_cm < 0.05 else 0.0   # 종이 위 0.9mm 심볼
    for x, y in pts:
        if ring:
            ax.add_patch(Circle((x, y), ring, fc="none", ec="#C9C9C9", lw=0.35, zorder=5))
        ax.add_patch(Circle((x, y), radius, fc=DOT, ec="#A85A18",
                            lw=0.0 if ring else 0.25, zorder=6))
    ax.plot([-x_hw, x_hw], [0, 0], color="#CCC", lw=0.4, ls=(0, (14, 3, 2, 3)), zorder=1)
    ax.plot([0, 0], [y_lo, y_hi], color="#CCC", lw=0.4, ls=(0, (14, 3, 2, 3)), zorder=1)

    # ---- 치수선 ------------------------------------------------------------ #
    arr = dict(arrowprops=dict(arrowstyle="<->", color="#333", lw=0.8,
                               shrinkA=0, shrinkB=0))
    def dim_h(x0, x1, y, label, va="bottom", dy=0.014):
        ax.annotate("", xy=(x0, y), xytext=(x1, y), **arr)
        for xe in (x0, x1):                      # 치수보조선
            ax.plot([xe, xe], [y, y + side * 0.02 * (1 if va == "bottom" else -1)],
                    color="#999", lw=0.4, zorder=2)
        ax.text((x0 + x1) / 2, y + side * dy * (1 if va == "bottom" else -1), label,
                ha="center", va=va, fontsize=7.5, color="#111")

    y_side = -side * SQRT3 / 2 - side * 0.06
    dim_h(-side / 2, side / 2, y_side, f"side a = {side:g}", va="top")
    dim_h(-SQRT3 / 2 * side, SQRT3 / 2 * side, y_lo + side * 0.20,
          f"across flats  \u221a3\u00b7a = {SQRT3 * side:.2f}")
    dim_h(-side, side, y_lo + side * 0.07, f"across corners  2a = {2 * side:g}")
    dim_h(0, pitch, side * 0.09, f"pitch p = {pitch:.3f}")
    ax.annotate("", xy=(-pitch / 2, 0), xytext=(-pitch / 2, row), **arr)
    ax.text(-pitch / 2 - side * 0.012, row / 2, f"row = \u221a3p/2 = {row:.3f}",
            ha="right", va="center", fontsize=7.5, color="#111", rotation=90)
    ax.annotate(f"outermost circle centres: hexagon of side {r_centers:.3f}",
                xy=(r_centers * math.cos(math.pi / 3), r_centers * math.sin(math.pi / 3)),
                xytext=(-side * 1.05, y_hi * 0.93), fontsize=7, color="#666",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.7))
    ax.add_patch(Circle((r_centers, 0), side * 0.055, fill=False, ec="#666", lw=0.7,
                        ls=(0, (4, 3)), zorder=7))
    ax.annotate("DETAIL A", xy=(r_centers - side * 0.04, side * 0.045),
                xytext=(side * 0.42, y_hi * 0.62), fontsize=8, color="#111",
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.7))

    # ---- 스케일 바 --------------------------------------------------------- #
    bar, bx, by = 500.0, side * 0.45, y_lo + side * 0.34
    for i in range(5):
        ax.add_patch(Rectangle((bx + i * bar / 5, by), bar / 5, side * 0.012,
                               fc="#111" if i % 2 == 0 else "#FFF", ec="#111",
                               lw=0.5, zorder=6))
    ax.text(bx, by - side * 0.016, "0", ha="center", va="top", fontsize=6.5)
    ax.text(bx + bar, by - side * 0.016, f"{bar:g} {unit}", ha="center", va="top",
            fontsize=6.5)
    ax.text(bx + bar / 2, by + side * 0.022, f"SCALE 1:{scale:g}", ha="center",
            va="bottom", fontsize=6.5, color="#555")

    # ---- DETAIL A (오른쪽 열, 자체 축척) ----------------------------------- #
    zx = r_centers
    d_left = (pad_l + draw_w + gap) / w_in
    d_w_in = 2 * zw * unit_cm / detail_scale / 2.54     # 상세도도 축척이 정확하다
    d_h_in = 2 * zh * unit_cm / detail_scale / 2.54
    d_bottom = (0.25 + tb_h + draw_h - d_h_in - 0.25) / h_in
    axd = fig.add_axes([d_left, d_bottom, d_w_in / w_in, d_h_in / h_in])
    axd.set_xlim(zx - zw * 1.1, zx + zw * 0.9)
    axd.set_ylim(-zh, zh)
    axd.set_aspect("equal")
    axd.set_xticks([]); axd.set_yticks([])
    for sp in axd.spines.values():
        sp.set_color("#111"); sp.set_linewidth(0.8)
    axd.add_patch(hexagon_patch(side, fill=False, ec="#111", lw=1.6))
    axd.add_patch(hexagon_patch(r_margin, fill=False, ec="#C05A2B", lw=0.8, ls=(0, (6, 4))))
    axd.add_patch(hexagon_patch(r_centers, fill=False, ec="#9AA0A6", lw=0.6, ls=(0, (1, 3))))
    axd.add_patch(Circle((r_centers, 0), radius, fc=DOT, ec="#A85A18", lw=0.4))
    nx, ny = math.cos(math.pi / 6), math.sin(math.pi / 6)   # 변에 수직인 방향
    p0 = (r_centers + nx * radius, ny * radius)
    p1 = (r_centers + nx * (radius + margin), ny * (radius + margin))
    axd.annotate("", xy=p0, xytext=p1, arrowprops=dict(arrowstyle="<->", color="#111", lw=0.9))
    axd.annotate(f"margin {margin:g} {unit}\n(circle edge \u2192 border,\n normal to it)",
                 xy=((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2),
                 xytext=(zx - zw * 1.02, zh * 0.62), fontsize=7, color="#111",
                 ha="left", va="top",
                 arrowprops=dict(arrowstyle="-", color="#888", lw=0.6))
    axd.annotate("", xy=(r_centers - radius, -zh * 0.30),
                 xytext=(r_centers + radius, -zh * 0.30),
                 arrowprops=dict(arrowstyle="<->", color="#111", lw=0.9))
    axd.text(r_centers, -zh * 0.36, f"\u00d8{2 * radius:g} {unit}", ha="center", va="top",
             fontsize=7, color="#111")
    axd.plot([zx - zw * 1.1, zx + zw * 0.9], [0, 0], color="#CCC", lw=0.4,
             ls=(0, (14, 3, 2, 3)))
    axd.set_title(f"DETAIL A   SCALE 1:{detail_scale:g}", fontsize=8, color="#111", pad=4)

    # ---- 주기 ------------------------------------------------------------- #
    notes = [
        "NOTES",
        f"1.  N = {n} circles on a triangular (hexagonal) lattice,",
        f"     {k} complete shells about the centre, 6-fold symmetric.",
        f"2.  MARGIN {margin:g} {unit} = clear distance from the outermost",
        f"     circle EDGE to the hexagon border, measured normal",
        "     to the border.  Corner circles govern; see DETAIL A.",
        f"3.  Circle \u00d8{2 * radius:g} {unit} drawn at true size." +
        ("  Grey ring is a" if ring else ""),
        ("     locator symbol only and is NOT to scale." if ring else
         "     No exaggeration anywhere on this sheet."),
        f"4.  Lattice: a1 = ({pitch:.4f}, 0), a2 = ({pitch / 2:.4f}, {row:.4f}) {unit}.",
        f"5.  Shell radii: k\u00b7p, k = 1..{k}  (outermost {r_centers:.4f} {unit}).",
        f"6.  All dimensions in {'centimetres' if unit == 'cm' else 'millimetres'}.",
    ]
    ty = d_bottom - 0.30 / h_in
    for i, line in enumerate(notes):
        fig.text(d_left, ty, line, fontsize=7.4, va="top", color="#111",
                 weight="bold" if i == 0 else "normal")
        ty -= (0.20 if i == 0 else 0.155) / h_in

    # ---- 표제란 ------------------------------------------------------------ #
    tb = Rectangle((pad_l / w_in, 0.35 / h_in), 1 - (pad_l + pad_r) / w_in,
                   tb_h / h_in, transform=fig.transFigure, fill=False, ec="#111", lw=1.1)
    fig.patches.append(tb)
    rows = [
        ("HEXAGONAL CIRCLE ARRAY", f"N = {n}", f"SCALE 1:{scale:g}   (sheet {w_in:.2f} x {h_in:.2f} in)"),
        (f"hexagon side a = {side:g} {unit}", f"across flats = {SQRT3 * side:.3f} {unit}",
         f"across corners = {2 * side:g} {unit}"),
        (f"circle \u00d8 = {2 * radius:g} {unit}  (r = {radius:g})",
         f"margin to border = {margin:g} {unit}", f"centre locus hexagon side = {r_centers:.4f} {unit}"),
        (f"pitch p = {pitch:.4f} {unit}", f"row spacing = {row:.4f} {unit}",
         f"clear gap between circles = {pitch - 2 * radius:.4f} {unit}"),
        (f"shells k = {k}", f"total circle area = {n * math.pi * radius ** 2:.4f} {unit}\u00b2",
         f"hexagon area = {1.5 * SQRT3 * side ** 2:.1f} {unit}\u00b2   "
         f"(fill {n * math.pi * radius ** 2 / (1.5 * SQRT3 * side ** 2) * 100:.5f} %)"),
    ]
    y = (0.35 + tb_h) / h_in - 0.24 / h_in
    for i, (c1, c2, c3) in enumerate(rows):
        for xf, txt in ((pad_l / w_in + 0.010, c1), (0.36, c2), (0.63, c3)):
            fig.text(xf, y, txt, fontsize=8.2, va="center", color="#111",
                     weight="bold" if i == 0 and xf < 0.2 else "normal")
        y -= 0.28 / h_in

    fig.savefig(out, dpi=400, facecolor="white")
    if out.lower().endswith(".png"):
        fig.savefig(out[:-4] + ".pdf", facecolor="white")   # 1:scale 그대로 인쇄 가능
    print(f"saved {out} (+pdf)  N={n} k={k} pitch={pitch:.4f} row={row:.4f} "
          f"centres_side={r_centers:.4f} sheet={w_in:.2f}x{h_in:.2f}in @1:{scale:g}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=61, help="초기 점 개수")
    p.add_argument("--mode", choices=list(MODES), default="hex")
    p.add_argument("--center", choices=CENTERS, default="point")
    p.add_argument("--shape", choices=SHAPES, default="circle",
                   help="경계 모양: hexagon(완전 육각) | circle(기본) | round | rounder(더 둥글게)")
    p.add_argument("--nmax", type=int, default=300, help="슬라이더/GIF 최대 N")
    p.add_argument("--raw", action="store_true", help="평균전력 정규화 끄기")
    p.add_argument("--grid", type=int, nargs="*", help="비교용 PNG로 저장할 N 목록")
    p.add_argument("--gif", action="store_true", help="N을 훑는 GIF 저장")
    p.add_argument("--step", type=int, default=1, help="GIF에서 N 증가 폭")
    p.add_argument("--out", default=None)
    p.add_argument("--layout", action="store_true",
                   help="실제 치수(육각형 한 변/마진/원 지름) 기반 배치 도면 1장")
    p.add_argument("--side", type=float, default=1100.0, help="육각형 한 변 길이")
    p.add_argument("--margin", type=float, default=5.0, help="테두리 마진")
    p.add_argument("--dia", type=float, default=1.0, help="원 지름")
    p.add_argument("--target", type=int, default=160, help="목표 원 개수")
    p.add_argument("--unit", default="cm")
    p.add_argument("--blueprint", action="store_true",
                   help="실제 축척 도면 1장 (원을 실제 크기로 그림)")
    p.add_argument("--radius", type=float, default=1.0, help="원 반지름 (도면용)")
    p.add_argument("--scale", type=float, default=100.0, help="축척 1:scale")
    a = p.parse_args()

    norm = not a.raw
    if a.blueprint:
        blueprint(a.side, a.margin, a.radius, a.n, a.out or f"blueprint_{a.n}.png",
                  a.unit, a.scale)
    elif a.layout:
        layout_figure(a.side, a.margin, a.dia, a.target,
                      a.out or f"layout_{int(a.side)}_{a.target}.png", a.unit)
    elif a.grid is not None:
        ns = a.grid or [1, 7, 19, 37, 61, 91, 127, 169]
        save_grid(ns, a.mode, a.center, norm,
                  a.out or f"constellation_{a.mode}_{a.shape}_grid.png", a.shape)
    elif a.gif:
        save_gif(a.nmax, a.mode, a.center, norm,
                 a.out or f"constellation_{a.mode}_{a.shape}.gif", a.step, shape=a.shape)
    else:
        interactive(a.n, a.mode, a.center, a.nmax, norm, a.shape)


if __name__ == "__main__":
    main()
