#!/usr/bin/env python3
"""
ストリーム筆跡アニメーション - 単一画像レンダリングのエントリポイント

1枚のカラー画像を「ペン先が連続した軌跡を滑りながら、進みつつインクを落とす」
ホワイトボードアニメーションにレンダリングする。
全体は3つのパートに分かれる:
  起筆(ink)   ペン先がインクの流れに沿って黒い線画を敷いていく
  着彩(color) 同じ軌跡を折り返し、ペン先が原色に持ち替えて画面を彩る
  凝視(gaze)  筆を置いた後に静止し、完成した元画像を見せる

「コマ単位で飛び飛びに変化する」やり方とは異なり、本レンダラーは描画順序を
ペン先の運動折れ線として扱う。隣接する着地点の間を補間し、インクブラシが
ペン先の滑りに合わせて連続的にインクを落とすことで、途切れのない筆跡の流れを生む。
"""
from __future__ import annotations

import argparse
import datetime
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
# リソースの場所
# ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _SCRIPT_DIR.parent / "assets"
DEFAULT_HAND_PNG = _ASSETS_DIR / "drawing-hand.png"


def _imread_any(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """
    画像を読み込む。日本語/空白などの非 ASCII 文字を含む Windows パスにも対応。
    まず np.fromfile でバイト列を読み、次に cv2.imdecode でデコードすることで、
    cv2.imread の非 ASCII パスに対する互換性問題を回避する。
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, flags)


# ──────────────────────────────────────────────────────────────
# レンダリングパラメータの集約場所
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    fps: int = 60                  # 高フレームレート出力でペン先の移動をより連続的な筆記に近づける
    grid_edge: int = 10            # グリッドを小さくして線画の出現のブロック感を減らす
    sample_step: int = 2           # ペン先軌跡のピクセルサンプリング間隔
    cap_long_edge: int = 1080      # 入力画像の長辺の上限
    brush_radius: int = 40         # 着彩フェーズの円形インクブラシ半径
    ink_weight: int = 2            # 線画パートの重み: 筆跡を観察する時間を多めに確保する
    color_weight: int = 1          # 着彩パートの重み
    gaze_seconds: float = 3.0      # 凝視パートの基準秒数
    ink_threshold: int = 10        # グレースケール値がこれ未満のピクセルを「インク」とみなす
    ink_reveal_radius: int = 4     # ペン先が軌跡1区間で線画を出現させられる半径
    target_hand_height: int = 493  # 手素材をスケーリングした後の目標の高さ（1080p 基準で調整）
    # 素材内でのペン先の正規化座標（0..1）。インクを落とす点を素材のどのピクセルに合わせるかを決める。
    # 内蔵の drawing-hand.png はトリミング後、ペン先が画像の左上角に来るためアンカーは (0, 0)。
    # ここで記述しているのは実際にインクが接触する点であり、手画像の外枠ではない。
    tip_anchor_x: float = 0.0
    tip_anchor_y: float = 0.0
    canvas_hex: str = "#F6F1E3"    # キャンバスの下地色
    match_bg: bool = True          # 元画像の背景をキャンバスの下地色に染め、起筆/着彩で背景を揃える
    match_bg_threshold: int = 28   # 元画像の背景色との差がこの値未満なら背景とみなす（BGR 3チャンネルの和）
    steps_per_frame: int = 4       # 1フレームあたりに進める着地点数の基準
    # ── contour-wipe 着彩モード専用 ──
    color_fill: str = "contour-wipe"  # 着彩スタイル: "contour-wipe" 輪郭を認識して上から下へ走査(既定) | "brush" 軌跡に沿って塗る
    wipe_decay: float = 0.86       # 抵抗場が1行ずつ下方向へ減衰する係数（半減期≈4.6px）
    wipe_delay_ratio: float = 0.04  # 輪郭部で出現の先端が差し引かれるピクセル比率（×h、[12,52] にクランプ）
    wipe_blocks: int = 18          # ペン先が横方向に往復して掃く回数
    # ── 起筆パートの適応的ポーズ（「ペンを持ち替える呼吸」のリズムを再現）──
    # pause_mode: "heavy" はっきりしたポーズ(既定)、"auto" 内容の密度で自動的に段階分け、"off" ポーズなし、"light" 少量
    pause_mode: str = "heavy"
    pause_ratio_heavy: float = 0.03   # 低密度(ゆっくりしたリズム)でのポーズ比率: 約 3% のフレームをポーズに使う
    pause_ratio_light: float = 0.008  # 中密度でのポーズ比率: 約 0.8%
    # 密度の段階分け閾値: 「1セルあたりのフレーム数」(frames_per_cell) で内容に対する尺の余裕度を測る。
    # >= heavy_fpc なら余裕が大きい → heavy 段階(ポーズ多め)、>= light_fpc なら適度 → light 段階、
    # < light_fpc なら内容が密で尺が厳しい → ポーズなし。
    pause_heavy_fpc: float = 0.7
    pause_light_fpc: float = 0.4
    # ── 筆跡パスのモード ──
    # ink_path_mode: "grid" グリッドのセル中心を補間(既定) | "skeleton" スケルトンレベルのピクセル追跡
    ink_path_mode: str = "grid"
    skeleton_min_points: int = 8        # スケルトン筆画の最小点数（断片を除外する）
    skeleton_resample_spacing: float = 2.5  # スケルトンの再サンプリング間隔（ピクセル）


# ──────────────────────────────────────────────────────────────
# 小さなユーティリティ
# ──────────────────────────────────────────────────────────────
def _hex_to_bgr(hex_color: str) -> np.ndarray:
    digits = hex_color.lstrip("#")
    if len(digits) != 6:
        raise ValueError(f"不正な色値: {hex_color}")
    r = int(digits[0:2], 16)
    g = int(digits[2:4], 16)
    b = int(digits[4:6], 16)
    return np.array([b, g, r], dtype=np.uint8)


def _bounding_box(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0), (0, 0)
    return (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max()))


# ──────────────────────────────────────────────────────────────
# インクのグリッド分割
# ──────────────────────────────────────────────────────────────
def _to_grid_blocks(image: np.ndarray, edge: int) -> np.ndarray:
    """HxW（x C）の画像を (行数, 列数, edge, edge[, C]) のブロックビューに切り分ける。"""
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    if h % edge or w % edge:
        raise ValueError(f"画像サイズ {w}x{h} は {edge} の整数倍である必要があります")
    rows, cols = h // edge, w // edge
    if image.ndim == 2:
        return image.reshape(rows, edge, cols, edge).transpose(0, 2, 1, 3)
    return image.reshape(rows, edge, cols, edge, image.shape[2]).transpose(0, 2, 1, 3, 4)


def _active_mask(threshold_map: np.ndarray, edge: int, threshold: int) -> np.ndarray:
    """どのグリッドがインクを含むか: ブロック内に閾値未満のグレースケール値のピクセルがあれば真。"""
    blocks = _to_grid_blocks(threshold_map, edge)
    return np.any(blocks < threshold, axis=(2, 3))


# ──────────────────────────────────────────────────────────────
# インクの流れのクラスタリング + 密度勾配ウォーク
# ──────────────────────────────────────────────────────────────
def _label_components(active: np.ndarray) -> tuple[np.ndarray, int]:
    """インクのセルに 8 連結の連結成分ラベリングを行い、(ラベル図, 成分数) を返す。"""
    n, labels = cv2.connectedComponents(active.astype(np.uint8), connectivity=8)
    return labels, n - 1  # 背景ラベル 0 を除く


def _component_cells(labels: np.ndarray, label: int) -> list[tuple[int, int]]:
    coords = np.argwhere(labels == label)
    return [(int(r), int(c)) for r, c in coords]


def _merge_small_components(
    components: list[list[tuple[int, int]]],
    merge_threshold: int,
) -> list[list[tuple[int, int]]]:
    """
    小さな連結成分（セル数 ≤ merge_threshold）を空間的に最も近い大きな連結成分へ統合する。
    1〜2 セルの断片が大きな文字の塊の間に大量に挟まり、「文字を1ブロック描き切る前に
    別の場所へ飛んでしまう」現象を防ぐ。
    統合先となる大きな連結成分がなければそのまま残す（インクを一切捨てない）。
    """
    if not components:
        return components
    big = [c for c in components if len(c) > merge_threshold]
    small = [c for c in components if len(c) <= merge_threshold]
    if not small or not big:
        return components

    # 各大領域の重心をあらかじめ計算する
    centroids = []
    for cells in big:
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        centroids.append((sum(rs) / len(rs), sum(cs) / len(cs)))

    # 各小断片を最も近い大領域へ統合する
    merged = [list(cells) for cells in big]  # コピー、追記可能
    for cells in small:
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        cr = sum(rs) / len(rs)
        cc = sum(cs) / len(cs)
        best = min(
            range(len(big)),
            key=lambda i: (centroids[i][0] - cr) ** 2 + (centroids[i][1] - cc) ** 2,
        )
        merged[best].extend(cells)
    return merged


def _bounds(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _split_bridge_connected_component(
    cells: list[tuple[int, int]],
    min_side_cells: int = 20,
) -> list[list[tuple[int, int]]]:
    """Split a very wide component when it is connected only by a thin bridge.

    A baseline, arrow, or stray outline can join separate objects into one
    connected component.  Drawing that component with one nearest-neighbour
    walk makes the pen alternate between those objects.  Valleys in the
    vertical ink projection are reliable weak-bridge signals at grid scale.
    """
    if len(cells) < min_side_cells * 2:
        return [cells]

    min_row, min_col, max_row, max_col = _bounds(cells)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    if width < 16 or height < 10:
        return [cells]

    counts = {col: 0 for col in range(min_col, max_col + 1)}
    for _, col in cells:
        counts[col] += 1
    valley_limit = max(3, int(np.ceil(height * 0.30)))
    edge_guard = 4
    valleys: list[tuple[int, int]] = []
    start: int | None = None
    for col in range(min_col, max_col + 2):
        low = col <= max_col and counts[col] <= valley_limit
        if low and start is None:
            start = col
        elif not low and start is not None:
            end = col - 1
            if (
                end - start + 1 >= 2
                and start > min_col + edge_guard
                and end < max_col - edge_guard
            ):
                valleys.append((start, end))
            start = None
    if not valleys:
        return [cells]

    # Prefer the broadest empty corridor.  It is much less likely to be an
    # internal detail of a character than a one-column dip.
    start, end = max(valleys, key=lambda band: (band[1] - band[0], -band[0]))
    cut = (start + end) // 2
    left = [cell for cell in cells if cell[1] <= cut]
    right = [cell for cell in cells if cell[1] > cut]
    if len(left) < min_side_cells or len(right) < min_side_cells:
        return [cells]
    return (
        _split_bridge_connected_component(left, min_side_cells)
        + _split_bridge_connected_component(right, min_side_cells)
    )


def _split_bridge_connected_components(
    components: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    return [
        piece
        for cells in components
        for piece in _split_bridge_connected_component(cells)
    ]


def _boxes_touch(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    margin: int = 2,
) -> bool:
    """Whether two component boxes belong to the same visual region."""
    a_top, a_left, a_bottom, a_right = first
    b_top, b_left, b_bottom, b_right = second
    return not (
        a_right + margin < b_left
        or b_right + margin < a_left
        or a_bottom + margin < b_top
        or b_bottom + margin < a_top
    )


def _group_adjacent_stroke_groups(
    groups: list[tuple[str, list[tuple[int, int]]]],
) -> list[list[tuple[str, list[tuple[int, int]]]]]:
    """Keep overlapping label parts and outline pieces in one draw region."""
    regions: list[list[tuple[str, list[tuple[int, int]]]]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for group in groups:
        group_box = _bounds(group[1])
        touching = [index for index, box in enumerate(boxes) if _boxes_touch(group_box, box)]
        if not touching:
            regions.append([group])
            boxes.append(group_box)
            continue
        target = touching[0]
        regions[target].append(group)
        top, left, bottom, right = boxes[target]
        boxes[target] = (
            min(top, group_box[0]), min(left, group_box[1]),
            max(bottom, group_box[2]), max(right, group_box[3]),
        )
        # Merge any regions newly bridged by the expanded box.
        for index in reversed(touching[1:]):
            regions[target].extend(regions.pop(index))
            other = boxes.pop(index)
            top, left, bottom, right = boxes[target]
            boxes[target] = (
                min(top, other[0]), min(left, other[1]),
                max(bottom, other[2]), max(right, other[3]),
            )
    return regions


def classify_stroke_groups(
    active: np.ndarray,
) -> list[tuple[str, list[tuple[int, int]]]]:
    """Classify connected ink regions as a main subject, text, or local contour."""
    labels, count = _label_components(active)
    components = [
        _component_cells(labels, label)
        for label in range(1, count + 1)
    ]
    components = [cells for cells in components if cells]
    if not components:
        return []

    # A long ground line may connect a mountain, a character, and a crowd.
    # Split that weak connection before any region ordering is decided.
    components = _split_bridge_connected_components(components)

    # 小さな断片を最も近い大領域へ統合し、断片が大きな文字塊の連続描画を分断しないようにする
    total_cells = sum(len(c) for c in components)
    merge_threshold = max(3, int(total_cells * 0.005))
    components = _merge_small_components(components, merge_threshold)

    subject_index = max(range(len(components)), key=lambda index: len(components[index]))
    groups: list[tuple[str, list[tuple[int, int]], tuple[int, int, int]]] = []
    for index, cells in enumerate(components):
        min_row, min_col, max_row, max_col = _bounds(cells)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        density = len(cells) / (height * width)
        if index == subject_index:
            kind, rank = "subject", 0
        elif height >= 2 and width / height >= 2.2 and density >= 0.5:
            kind, rank = "text", 1
        else:
            kind, rank = "contour", 2
        groups.append((kind, cells, (rank, min_row, min_col)))

    groups.sort(key=lambda group: group[2])
    return [(kind, cells) for kind, cells, _ in groups]


def _density_seed(cells: Sequence[tuple[int, int]], radius: int = 2) -> tuple[int, int]:
    """局所的に隣接が最も密なセルを起筆点に選び、「インクの最も濃い所から書き始める」動きを再現する。"""
    cell_set = set(cells)
    best = cells[0]
    best_score = -1
    for (r, c) in cells:
        score = sum(
            1
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if (r + dr, c + dc) in cell_set
        )
        if score > best_score:
            best_score = score
            best = (r, c)
    return best


def _gradient_walk(cells: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    密度勾配に導かれる貪欲ウォーク: 密度が最も高いシードセルから出発し、
    毎ステップ「未訪問の隣接セルのうち局所密度が最も高く、進行方向との角度差が最小」の
    セルを選ぶことで、「できるだけインクに沿い、折り返しの少ない」連続した筆跡を作る。
    到達可能な隣接セルがなくなったら、全体で最も近い未訪問セルへ飛んで続行する。
    """
    if not cells:
        return []

    cell_set = set(cells)
    seed = _density_seed(cells)
    visited: set[tuple[int, int]] = {seed}
    path: list[tuple[int, int]] = [seed]
    current = seed
    prev_dir = (0, 0)

    while len(visited) < len(cells):
        neighbors = [
            (r, c)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc)
            and (r := current[0] + dr, c := current[1] + dc) in cell_set
            and (r, c) not in visited
        ]
        if neighbors:
            def cost(cell: tuple[int, int]) -> tuple:
                # 隣接が多いほど良い（負号で最小化）、方向変化が小さいほど良い、最後は位置で安定ソート
                local = sum(
                    1
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (cell[0] + dr, cell[1] + dc) in cell_set
                    and (cell[0] + dr, cell[1] + dc) not in visited
                )
                step = (cell[0] - current[0], cell[1] - current[1])
                turn = (step[0] - prev_dir[0]) ** 2 + (step[1] - prev_dir[1]) ** 2
                return (-local, turn, cell[0], cell[1])

            nxt = min(neighbors, key=cost)
        else:
            # 筆が途切れる: 最も近い未訪問セルへ飛ぶ
            unvisited = [cell for cell in cells if cell not in visited]
            nxt = min(
                unvisited,
                key=lambda cell: (
                    (cell[0] - current[0]) ** 2 + (cell[1] - current[1]) ** 2,
                    cell[0],
                    cell[1],
                ),
            )

        prev_dir = (nxt[0] - current[0], nxt[1] - current[1])
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return path


def _nearest_neighbor_order(
    cells: Sequence[tuple[int, int]], seed: tuple[int, int]
) -> list[tuple[int, int]]:
    """seed から出発し、毎ステップ最も近い未訪問セルへ進んで連続した筆跡を作る。"""
    if not cells:
        return []
    remaining = list(cells)
    ordered: list[tuple[int, int]] = []
    current = seed if seed in remaining else remaining[0]
    while remaining:
        ordered.append(current)
        remaining.remove(current)
        if not remaining:
            break
        current = min(
            remaining,
            key=lambda cell: (cell[0] - ordered[-1][0]) ** 2
            + (cell[1] - ordered[-1][1]) ** 2,
        )
    return ordered


def _text_scan_order(
    cells: Sequence[tuple[int, int]], segment_cols: int = 4
) -> list[tuple[int, int]]:
    """
    文字領域専用の描き方: 横方向にセグメント単位で走査し、文字を書く動きを再現する。
    セルを列方向にいくつかのセグメント（各セグメントは segment_cols 列幅）へ切り、
    セグメント間は列の左から右へ進む。セグメント内は最近傍でインクに沿って連続的に進む
    （柵状に1行ずつ走査するのではない）ことで、「1ブロックを描き切らないまま
    次のセグメントの上端から始まり、後から戻って描き足す」感じを避ける。
    """
    if not cells:
        return []
    if segment_cols < 1:
        segment_cols = 1
    left_col = min(col for _, col in cells)
    # 「開始列 // segment_cols」でバケットに分け、番号の小さい（左寄りの）バケットから描く
    buckets: dict[int, list[tuple[int, int]]] = {}
    for cell in cells:
        bucket_key = (cell[1] - left_col) // segment_cols
        buckets.setdefault(bucket_key, []).append(cell)

    ordered: list[tuple[int, int]] = []
    prev_tail: tuple[int, int] | None = None
    for key in sorted(buckets):
        seg_cells = buckets[key]
        # セグメントの開始点: できるだけ前セグメントの出口に近づけ、セグメント間の飛びを減らす
        if prev_tail is not None:
            seed = min(
                seg_cells,
                key=lambda cell: (cell[0] - prev_tail[0]) ** 2
                + (cell[1] - prev_tail[1]) ** 2,
            )
        else:
            seed = min(seg_cells, key=lambda cell: (cell[0], cell[1]))
        seg_order = _nearest_neighbor_order(seg_cells, seed)
        ordered.extend(seg_order)
        prev_tail = seg_order[-1]
    return ordered


def _order_stream_by_kind(
    kind: str, cells: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """領域の種類ごとに描き方を選ぶ: 文字は横方向のセグメント走査、主体/輪郭は密度ウォーク。"""
    if kind == "text":
        return _text_scan_order(cells)
    return _gradient_walk(cells)


def _chain_region_paths(
    groups: list[tuple[str, list[tuple[int, int]]]],
) -> list[tuple[int, int]]:
    """Finish every component in one visual region before leaving it."""
    paths = [_order_stream_by_kind(kind, cells) for kind, cells in groups]
    remaining = [path for path in paths if path]
    ordered: list[tuple[int, int]] = []
    tail: tuple[int, int] | None = None
    while remaining:
        if tail is None:
            pick_index = 0  # groups retain subject/text/contour priority.
        else:
            pick_index = min(
                range(len(remaining)),
                key=lambda index: min(
                    (remaining[index][0][0] - tail[0]) ** 2
                    + (remaining[index][0][1] - tail[1]) ** 2,
                    (remaining[index][-1][0] - tail[0]) ** 2
                    + (remaining[index][-1][1] - tail[1]) ** 2,
                ),
            )
        path = remaining.pop(pick_index)
        if tail is not None and len(path) > 1:
            head_distance = (path[0][0] - tail[0]) ** 2 + (path[0][1] - tail[1]) ** 2
            end_distance = (path[-1][0] - tail[0]) ** 2 + (path[-1][1] - tail[1]) ** 2
            if end_distance < head_distance:
                path.reverse()
        ordered.extend(path)
        tail = path[-1]
    return ordered


def cluster_ink_streams(active: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    インクのセルを意味単位でいくつかのインクの流れにまとめる:
    主体(subject) → 文字(text) → 局所輪郭(contour)。
    各流れの内部は種類に応じた描き方を選ぶ（文字はセグメント走査、それ以外は密度ウォーク）。
    流れ同士は「出口から入口への最近傍」で動的に連結し、必要なら流れ全体を反転して
    ペンの飛びを減らす。
    戻り値は連結して並べ替え済みの複数の筆跡の流れ。
    """
    if not active.any():
        return []
    groups = classify_stroke_groups(active)
    # A stream is now a complete visual region, not merely one connected
    # component.  Thus a label's border, its characters, and its arrow cannot
    # be interrupted by a different object that happens to be closer.
    regions = _group_adjacent_stroke_groups(groups)
    streams = [_chain_region_paths(region) for region in regions]
    streams = [s for s in streams if s]
    if not streams:
        return []

    # 連結: 主体（最初の1本）で始め、以降は入口が現在の出口に最も近い流れを毎回選ぶ。
    # 必要に応じてその流れ全体を反転し、開始点を前の流れの出口に近づける。
    ordered: list[list[tuple[int, int]]] = []
    remaining = list(streams)
    tail: tuple[int, int] | None = None
    while remaining:
        if tail is None:
            pick_idx = 0  # classify で主体はすでに先頭に並んでいる
        else:
            def dist_to_tail(stream: list[tuple[int, int]]) -> int:
                head = stream[0]
                return (head[0] - tail[0]) ** 2 + (head[1] - tail[1]) ** 2
            pick_idx = min(range(len(remaining)), key=lambda i: dist_to_tail(remaining[i]))
        pick = remaining.pop(pick_idx)
        # 必要に応じて反転: 末尾が pick の終点の方が始点より近ければ反転する
        if tail is not None and len(pick) > 1:
            head = pick[0]
            end = pick[-1]
            d_end = (end[0] - tail[0]) ** 2 + (end[1] - tail[1]) ** 2
            d_head = (head[0] - tail[0]) ** 2 + (head[1] - tail[1]) ** 2
            if d_end < d_head:
                pick = pick[::-1]
        ordered.append(pick)
        tail = pick[-1]
    return ordered


def flatten_streams(streams: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for stream in streams for cell in stream]


# ──────────────────────────────────────────────────────────────
# ペン先 / 手のオーバーレイ
# ──────────────────────────────────────────────────────────────
def _load_hand(path: Path, target_h: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    手の素材を読み込み、目標の高さに合わせて等比スケーリングする。
    マスクにはアルファチャンネルを優先して使い、アルファがなければ「ほぼ白なら背景」判定に戻す。
    戻り値は (手のBGR, 正規化マスク[0..1])。失敗時は None。
    """
    if not path.exists():
        return None
    raw = _imread_any(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None

    if raw.ndim == 3 and raw.shape[2] == 4:
        hand = raw[:, :, :3]
        mask = raw[:, :, 3]
    else:
        hand = raw
        gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # 有効領域にトリミングする
    (x0, y0), (x1, y1) = _bounding_box(mask)
    if x1 <= x0 or y1 <= y0:
        return None
    hand = hand[y0:y1 + 1, x0:x1 + 1]
    mask = mask[y0:y1 + 1, x0:x1 + 1]

    scale = target_h / hand.shape[0]
    new_w = max(1, int(round(hand.shape[1] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    hand = cv2.resize(hand, (new_w, target_h), interpolation=interp)
    mask = cv2.resize(mask, (new_w, target_h), interpolation=interp)
    mask = mask.astype(np.float32) / 255.0

    # マスク外の領域を黒で埋め、後段のマスク合成をしやすくする
    hand[mask <= 0] = 0
    return hand, mask


def _procedural_tip(target_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    フォールバック用のペン先: マーカーペンをプログラムで描く（軸のグラデーション + 丸い先端のソフトエッジ + 影）。
    外部画像に一切依存しないため、素材が欠けていても出力できる。
    """
    w = max(1, int(target_h * 0.34))
    h = target_h
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # 影: オフセットした暗い帯をぼかして下敷きにする
    shadow = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(shadow, (3, int(h * 0.06)), (w - 2, int(h * 0.62)), 90, thickness=-1)
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
    rgba[:, :, 3] = shadow

    # ペン軸: 明から暗への縦方向グラデーション
    for y in range(h):
        t = y / max(1, h - 1)
        shade = int(220 - 130 * t)
        rgba[y, :, 0:3] = (shade, shade, shade + 10)
    cv2.rectangle(rgba, (4, int(h * 0.04)), (w - 4, int(h * 0.58)), (0, 0, 0), thickness=1)

    # 丸いペン先（暖色でインクを表現）
    tip_cy = int(h * 0.70)
    cv2.circle(rgba, (w // 2, tip_cy), max(3, w // 4), (70, 90, 230), thickness=-1)

    # 丸い先端 + ペン軸の外輪郭からアルファマスクを合成する
    body_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(body_mask, (3, int(h * 0.04)), (w - 3, tip_cy), 255, thickness=-1)
    cv2.circle(body_mask, (w // 2, tip_cy), max(3, w // 4), 255, thickness=-1)
    body_mask = cv2.GaussianBlur(body_mask, (7, 7), 0)

    hand = rgba[:, :, :3]
    mask = np.maximum(rgba[:, :, 3], body_mask).astype(np.float32) / 255.0
    hand[mask <= 0] = 0
    return hand, mask


class TipOverlay:
    """ペン先/手をキャンバスに貼り付け、指定した「ペン先アンカー」をインクの着地点に合わせる（アルファ合成つき）。"""

    def __init__(
        self,
        hand: np.ndarray,
        mask: np.ndarray,
        tip_anchor_x: float = 0.0,
        tip_anchor_y: float = 0.0,
    ) -> None:
        self.hand = hand
        self.mask = mask
        self.h, self.w = hand.shape[:2]
        self.mask_inv = 1.0 - mask
        # 素材内でのペン先のピクセル座標（インクの着地点をここに合わせる）
        # Map normalized anchors exactly onto the source image's pixel range.
        self.tip_px = int(round((self.w - 1) * np.clip(tip_anchor_x, 0.0, 1.0)))
        self.tip_py = int(round((self.h - 1) * np.clip(tip_anchor_y, 0.0, 1.0)))

    def stamp(self, canvas: np.ndarray, x: int, y: int) -> np.ndarray:
        """素材のペン先アンカーをキャンバス座標 (x, y)（＝インクの着地点）に合わせる。"""
        # 素材の左上角 = インクの着地点 - ペン先オフセット
        anchor_x = x - self.tip_px
        anchor_y = y - self.tip_py
        h_canvas, w_canvas = canvas.shape[:2]

        x0 = max(0, anchor_x)
        y0 = max(0, anchor_y)
        x1 = min(w_canvas, anchor_x + self.w)
        y1 = min(h_canvas, anchor_y + self.h)
        if x1 <= x0 or y1 <= y0:
            return canvas

        sx0 = x0 - anchor_x
        sy0 = y0 - anchor_y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        region = canvas[y0:y1, x0:x1]
        hand_region = self.hand[sy0:sy1, sx0:sx1]
        mask_region = self.mask[sy0:sy1, sx0:sx1]
        inv_region = self.mask_inv[sy0:sy1, sx0:sx1]

        for c in range(3):
            region[:, :, c] = (
                region[:, :, c] * inv_region + hand_region[:, :, c] * mask_region
            )
        canvas[y0:y1, x0:x1] = region
        return canvas


# ──────────────────────────────────────────────────────────────
# インクブラシ
# ──────────────────────────────────────────────────────────────
def _feathered_disk(radius: int) -> np.ndarray:
    """半径 r で縁をガウスぼかしした円形マスクを生成する。値域は 0..1。"""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    dist = np.sqrt(x * x + y * y).astype(np.float32)
    return np.clip(1.0 - (dist - radius * 0.75) / (radius * 0.25), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────
# contour-wipe 着彩のユーティリティ
# ──────────────────────────────────────────────────────────────
def _ease_in_out_sine(t: float | np.ndarray) -> float | np.ndarray:
    """サインイージング: 始点と終点は遅く、中間は速い。スカラーまたは配列を受け取り、同じ形で返す。"""
    return -(np.cos(np.pi * t) - 1.0) / 2.0


def _build_wipe_wave(width: int) -> np.ndarray:
    """
    2周波のサイン波による境界をあらかじめ計算し、出現の先端を直線ではなく波打つ形にする。
    戻り値は (W,) の float32 配列で、値域はおおよそ [-1.35, 1.35]。
    """
    wave_px1 = max(24.0, width / 20.0)
    wave_px2 = max(8.0, width / 72.0)
    xs = np.arange(width, dtype=np.float32)
    return np.sin(xs / wave_px1) + 0.35 * np.sin(xs / wave_px2 + 1.7)


# ──────────────────────────────────────────────────────────────
# スケルトンレベルの筆画追跡（whiteboard-video-engine の preprocess.py から移植）
# Zhang-Suen 細線化 → 8近傍で最も直進する辺を追跡 → ピクセル単位の順序つき筆画
# ──────────────────────────────────────────────────────────────
_SKEL_NEIGHBORS_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
]


def _zhang_suen_skeleton(mask: np.ndarray, max_iterations: int = 160) -> np.ndarray:
    """
    Zhang-Suen の2サブイテレーション細線化で、二値の前景マスクを 1px 幅のスケルトンまで細める。
    入力: bool/uint8 の2次元配列（True/1 = 前景の筆跡）。
    出力: bool のスケルトン図（入力と同じ形）。
    """
    img = np.pad(mask.astype(np.uint8), 1, mode="constant")
    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            center = img[1:-1, 1:-1]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            # 0→1 遷移の回数（時計回りに一周）
            transitions = sum(
                (neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1) for i in range(8)
            )
            count = sum(neighbors)
            if step == 0:
                marker = (
                    (center == 1) & (count >= 2) & (count <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
                )
            else:
                marker = (
                    (center == 1) & (count >= 2) & (count <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
                )
            if np.any(marker):
                center[marker] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _skel_neighbors(skel: np.ndarray, point: tuple[int, int]) -> list[tuple[int, int]]:
    """
    スケルトン点 point の有効な 8 近傍を返す。
    重要: 斜め方向の隣接点と現在の点の間にすでに直交の橋がある場合、その斜め隣接点はスキップする。
    これにより T 字/十字の交差部で三角形の細かい筆画が生じるのを防ぎつつ、
    純粋な斜めの中心線は保持する。
    """
    x, y = point
    h, w = skel.shape
    result: list[tuple[int, int]] = []
    for dx, dy in _SKEL_NEIGHBORS_8:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < w and 0 <= ny < h and skel[ny, nx]):
            continue
        if dx != 0 and dy != 0 and (skel[y, nx] or skel[ny, x]):
            continue  # 直交の橋ですでに接続済み、冗長な斜めはスキップ
        result.append((nx, ny))
    return result


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """無向辺の正規化: (A,B) と (B,A) を同じキーに写す。"""
    return (a, b) if a <= b else (b, a)


def _choose_next(
    prev: tuple[int, int],
    cur: tuple[int, int],
    candidates: list[tuple[int, int]],
    visited_edges: set,
) -> tuple[int, int] | None:
    """
    交差点で「最も直進する未訪問の辺」を選んで進む。
    現在の進行方向と候補方向のコサイン類似度で「直進度」を測り、最大のものを取る。
    """
    fresh = [p for p in candidates if _edge_key(cur, p) not in visited_edges and p != prev]
    if not fresh:
        return None
    vx, vy = cur[0] - prev[0], cur[1] - prev[1]
    vlen = math.hypot(vx, vy)
    return max(
        fresh,
        key=lambda p: (
            (vx * (p[0] - cur[0]) + vy * (p[1] - cur[1]))
            / (vlen * math.hypot(p[0] - cur[0], p[1] - cur[1]) or 1.0)
        ),
    )


def trace_8connected(skel: np.ndarray, min_points: int = 8) -> list[list[tuple[int, int]]]:
    """
    1px のスケルトンを追跡して順序つきの筆画列にする。

    - 開始点の優先順位: 次数=1 の端点 → 次数>2 の交差点 → その他
    - 交差点では最も直進する未訪問の辺に沿って進む（分岐ごとに短い筆画へ切らない）
    - 無向辺の集合で訪問を管理する（ピクセルは再利用可、辺は再走行不可）
    - 行き止まり（未訪問の辺なし）で停止し、残った分岐は後続の開始点が拾う
    - 長さが min_points 未満の断片は破棄する

    戻り値は list[list[(x,y)]]。各要素は筆画の方向に沿った順序つきのピクセル座標。
    """
    ys, xs = np.nonzero(skel)
    points = [(int(x), int(y)) for x, y in zip(xs, ys)]
    if not points:
        return []
    degrees = {p: len(_skel_neighbors(skel, p)) for p in points}
    starts = (
        [p for p in points if degrees[p] == 1]
        + [p for p in points if degrees[p] > 2]
        + points
    )
    visited_edges: set = set()
    strokes: list[list[tuple[int, int]]] = []
    for start in starts:
        for nb in _skel_neighbors(skel, start):
            edge = _edge_key(start, nb)
            if edge in visited_edges:
                continue
            path = [start]
            prev, cur = start, nb
            visited_edges.add(edge)
            while True:
                path.append(cur)
                next_pt = _choose_next(prev, cur, _skel_neighbors(skel, cur), visited_edges)
                if next_pt is None:
                    break
                visited_edges.add(_edge_key(cur, next_pt))
                prev, cur = cur, next_pt
            if len(path) >= min_points:
                strokes.append(path)
    return strokes


# ── スケルトン筆画の後処理（再サンプリング + 平滑化 + 並べ替え）──
def _stroke_cumulative_length(points: list[tuple[float, float]]) -> list[float]:
    """各点までの累積弧長 [0, d01, d012, ...]。"""
    cum = [0.0]
    for a, b in zip(points, points[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return cum


def _resample_stroke_points(
    points: list[tuple[float, float]], spacing: float
) -> list[tuple[float, float]]:
    """弧長に沿って spacing 間隔で等距離に再サンプリングし、ピクセルのギザギザを取る。"""
    if len(points) < 2:
        return list(points)
    cum = _stroke_cumulative_length(points)
    total = cum[-1]
    if total < spacing:
        return [points[0], points[-1]]
    n = max(2, int(round(total / spacing)))
    result: list[tuple[float, float]] = []
    for i in range(n + 1):
        target = total * i / n
        # 二分探索で位置を特定する
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            result.append(points[0])
            continue
        seg_start = cum[lo - 1]
        seg_len = cum[lo] - seg_start
        t = (target - seg_start) / seg_len if seg_len > 0 else 0.0
        ax, ay = points[lo - 1]
        bx, by = points[lo]
        result.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return result


def _chaikin_smooth(
    points: list[tuple[float, float]], iterations: int = 1
) -> list[tuple[float, float]]:
    """Chaikin の角切り平滑化: 各区間を 0.25/0.75 の2点で置き換え、始点と終点は保持する。"""
    pts = list(points)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        smoothed = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            smoothed.append((a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25))
            smoothed.append((a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75))
        smoothed.append(pts[-1])
        pts = smoothed
    return pts


def _order_skeleton_strokes(strokes: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """
    筆画の並べ替え: 上から下、左から右、長い筆画を優先。
    order_strokes の簡易版 — バウンディングボックスの左上角 + 負の長さで辞書順に並べる。
    """
    def sort_key(s):
        if not s:
            return (0, 0, 0, 0)
        xs = [p[0] for p in s]
        ys = [p[1] for p in s]
        length = _stroke_cumulative_length(s)[-1]
        return (min(ys) // 12, min(xs), min(ys), -length)
    return sorted(strokes, key=sort_key)


# ──────────────────────────────────────────────────────────────
# 尺 / パートの分割
# ──────────────────────────────────────────────────────────────
@dataclass
class PhasePlan:
    ink_frames: int
    color_frames: int
    gaze_frames: int
    ratio_label: str


def plan_phases(total_ms: int, cfg: Config) -> PhasePlan:
    """
    全体の尺を 起筆/着彩/凝視 の3パートに分割する。
    凝視パートはまず基準秒数で場所を確保し、残りの尺を重みに従って起筆と着彩へ配分する。
    残りが重みの合計で割り切れない場合、余りは凝視パートに足して精度の損失を避ける。
    """
    weight_sum = cfg.ink_weight + cfg.color_weight
    gaze_ms = int(cfg.gaze_seconds * 1000)
    anim_ms = total_ms - gaze_ms
    remainder = anim_ms % weight_sum
    if remainder:
        anim_ms -= remainder
        gaze_ms += remainder

    ink_frames = round(anim_ms * cfg.ink_weight / weight_sum * cfg.fps / 1000)
    color_frames = round(anim_ms * cfg.color_weight / weight_sum * cfg.fps / 1000)
    gaze_frames = round(gaze_ms * cfg.fps / 1000)
    if ink_frames <= 0 and color_frames <= 0:
        ink_frames = color_frames = 0
    return PhasePlan(ink_frames, color_frames, gaze_frames, f"{cfg.ink_weight}:{cfg.color_weight}")


# ──────────────────────────────────────────────────────────────
# レンダラー本体
# ──────────────────────────────────────────────────────────────
class StreamBoardRenderer:
    """1回のレンダリングに必要な状態をすべて保持し、メソッドをインスタンスに紐づける。"""

    def __init__(
        self,
        image_bgr: np.ndarray,
        cfg: Config,
        hand_png: Path | None,
        bare_tip: bool,
    ) -> None:
        self.cfg = cfg
        self.canvas_bgr = _hex_to_bgr(cfg.canvas_hex)

        # 出力サイズを計算: 長辺を cap に制限し、grid_edge の偶数倍に揃える（エンコードが偶数を要求）
        h0, w0 = image_bgr.shape[:2]
        scale = cfg.cap_long_edge / max(h0, w0)
        w = int(round(w0 * scale))
        h = int(round(h0 * scale))
        align = cfg.grid_edge if cfg.grid_edge % 2 == 0 else cfg.grid_edge * 2
        w = (w // align) * align
        h = (h // align) * align
        self.out_w = max(align, w)
        self.out_h = max(align, h)

        self.color_img = cv2.resize(image_bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY)
        self.thresh_map = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
        self.active = _active_mask(self.thresh_map, cfg.grid_edge, cfg.ink_threshold)
        self.grid_blocks = _to_grid_blocks(self.thresh_map, cfg.grid_edge)
        self.ink_pixels = self.thresh_map < cfg.ink_threshold
        self.ink_paint = np.repeat(self.thresh_map[:, :, None], 3, axis=2).astype(np.float32)

        # 元画像の背景をキャンバスの下地色に染める（影響するのは color_img のみで、ink_pixels / ink_paint には触れない）。
        # これにより着彩/凝視フェーズの背景が起筆(線画)フェーズと揃い、背景色の唐突な変化を避けられる。
        # ink_pixels を算出した後に実行するため、線画の品質にはまったく影響しない。
        if cfg.match_bg:
            self._match_original_background()

        # セル空間のクラスタリング（grid モードの筆跡パス + contour-wipe の抵抗場に依然として必要）
        self.ink_streams = cluster_ink_streams(self.active)

        # 筆跡パス: ink_path_mode に応じて grid（セル中心の補間）か skeleton（スケルトン追跡）を選ぶ
        if cfg.ink_path_mode == "skeleton":
            self.skeleton_strokes = self._build_skeleton_path()
            if self.skeleton_strokes:
                self.stroke_path = [pt for stroke in self.skeleton_strokes for pt in stroke]
            else:
                # スケルトン追跡で筆画が得られない場合: 空のパスを残さず、確実にセル中心パスへ戻す
                self.stroke_path = flatten_streams(self.ink_streams)
        else:
            self.skeleton_strokes = []
            self.stroke_path = flatten_streams(self.ink_streams)

        # キャンバス（インクブラシの累積合成をしやすくするため浮動小数のバッファを使う）
        self.drawn = np.zeros((self.out_h, self.out_w, 3), dtype=np.float32)
        self.drawn[...] = self.canvas_bgr.astype(np.float32)

        # ペン先のオーバーレイ
        self.tip: TipOverlay | None = None
        if not bare_tip:
            hand_data = _load_hand(hand_png, cfg.target_hand_height) if hand_png else None
            tip_anchor_x = cfg.tip_anchor_x
            tip_anchor_y = cfg.tip_anchor_y
            if hand_data is None:
                hand_data = _procedural_tip(cfg.target_hand_height)
                tip_anchor_x = 0.5
                tip_anchor_y = 0.70
            self.tip = TipOverlay(
                hand_data[0], hand_data[1],
                tip_anchor_x=tip_anchor_x,
                tip_anchor_y=tip_anchor_y,
            )

    # ── 元画像の背景をキャンバスの下地色に染める（影響は color_img のみ、線画のインクには触れない）──
    def _match_original_background(self) -> None:
        """
        元画像の四隅をサンプリングして背景色の基準とし、それとの差が閾値未満のピクセルを canvas_hex に置き換える。
        着彩/凝視フェーズの背景を起筆(線画)フェーズと揃え、背景色の唐突な変化を避ける。
        カラーの内容（背景との差が大きい部分）は元の色のまま影響を受けない。
        """
        img = self.color_img
        h, w = img.shape[:2]
        margin = max(3, min(h, w) // 50)
        samples = [
            img[:margin, :margin], img[:margin, -margin:],
            img[-margin:, :margin], img[-margin:, -margin:],
        ]
        bg_color = np.median(np.concatenate([s.reshape(-1, 3) for s in samples]), axis=0)
        diff = np.abs(img.astype(np.int16) - bg_color.astype(np.int16)).sum(axis=2)
        bg_mask = diff < self.cfg.match_bg_threshold
        img[bg_mask] = self.canvas_bgr

    # ── 筆跡の中心点（ピクセル座標）──
    def _cell_center(self, cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        e = self.cfg.grid_edge
        return (c * e + e // 2, r * e + e // 2)  # (x, y)

    # ── スケルトンレベルの筆跡パス（Zhang-Suen 細線化 + 8近傍で最も直進する辺の追跡）──
    def _build_skeleton_path(self) -> list[list[tuple[int, int]]]:
        """
        スケルトン追跡でピクセル単位の順序つき筆画列を生成し、グリッドのセル中心の補間を置き換える。
        ペン先が実際のスケルトンをたどるため、グリッド中心より元画像の線に忠実になる。
        交差点では最も直進する辺に沿って進み、三角形の細かい筆画を避ける。
        """
        cfg = self.cfg
        skel = _zhang_suen_skeleton(self.ink_pixels, max_iterations=160)
        raw_strokes = trace_8connected(skel, min_points=cfg.skeleton_min_points)
        if not raw_strokes:
            print("  [warn] スケルトン追跡で筆画が得られません。セル中心パスへフォールバックします")
            return []

        spacing = cfg.skeleton_resample_spacing
        processed: list[list[tuple[int, int]]] = []
        for stroke in raw_strokes:
            pts = [(float(x), float(y)) for x, y in stroke]
            pts = _resample_stroke_points(pts, spacing)
            pts = _chaikin_smooth(pts, iterations=1)
            pts = _resample_stroke_points(pts, spacing)
            if len(pts) >= 2 and _stroke_cumulative_length(pts)[-1] > 2.0:
                processed.append([(int(round(x)), int(round(y))) for x, y in pts])

        processed = _order_skeleton_strokes(processed)
        total_pts = sum(len(s) for s in processed)
        print(f"  スケルトン追跡: {len(processed)} 本の筆画, {total_pts} 個のサンプル点")
        return processed

    # ── contour-wipe の抵抗場（遅延構築、着彩フェーズ全体で再利用）──
    def _build_resistance_field(self) -> np.ndarray:
        """
        線画のインクから「抵抗場」を構築する: 輪郭部の抵抗≈1 で、下方向へ 1 行ずつ decay で指数的に減衰する。
        出現の先端は高い抵抗に当たるとピクセル数を差し引かれ、「まず輪郭で止まり、その後ゆっくり越える」動きになる。

        抵抗場は全編を通して静的で着彩の進捗に依存しないため、1回だけ計算して self._resistance にキャッシュする。
        """
        if getattr(self, "_resistance", None) is not None:
            return self._resistance

        h, w = self.out_h, self.out_w
        cfg = self.cfg

        # 1) 墨線の二値画像（uint8 0/255）
        ink_u8 = (self.ink_pixels.astype(np.uint8)) * 255

        # 2) 膨張: 円形の構造要素で輪郭を太らせ、せき止める帯を作る
        spread = int(np.clip(min(w, h) // 64, 3, 17))
        if spread % 2 == 0:  # 構造要素の半径は正の奇数である必要がある
            spread = max(3, spread - 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spread, spread))
        dilated = cv2.dilate(ink_u8, kernel, iterations=1)

        # 3) ガウスぼかし: 硬い縁をグラデーションの帯に変える（半径は正の奇数である必要がある）
        blur_r = max(1, int(round(min(w, h) / 220.0)))
        if blur_r % 2 == 0:
            blur_r += 1
        resistance = cv2.GaussianBlur(dilated, (blur_r, blur_r), 0).astype(np.float32)

        # 4) [0,1] に正規化する
        peak = float(resistance.max())
        if peak > 1e-6:
            resistance /= peak
        else:
            # 墨線なし（真っ白な画像）: 抵抗場は常に 0 となり、contour-wipe はまっすぐな走査に縮退する
            resistance = np.zeros((h, w), dtype=np.float32)

        # 5) 1行ずつ下方向へ因果的に decay を伝播させ、各輪郭が下方へ指数減衰する影を落とすようにする
        decay = cfg.wipe_decay
        for row in range(1, h):
            resistance[row] = np.maximum(resistance[row], resistance[row - 1] * decay)

        self._resistance = resistance
        return resistance

    # ── 着地点に「インクの点」を置く: 線画フェーズは閾値画像を、着彩フェーズは原色を置く ──
    def _reveal_ink_segment(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> None:
        """Reveal only the original line-art pixels touched by one pen movement."""
        segment = np.zeros((self.out_h, self.out_w), dtype=np.uint8)
        thickness = max(1, self.cfg.ink_reveal_radius * 2 + 1)
        cv2.line(segment, start, end, 255, thickness=thickness, lineType=cv2.LINE_AA)
        revealed = (segment > 0) & self.ink_pixels
        self.drawn[revealed] = self.ink_paint[revealed]

    def _ink_stamp(self, cell: tuple[int, int]) -> None:
        r, c = cell
        e = self.cfg.grid_edge
        block = self.grid_blocks[r, c]
        ink_region = block < self.cfg.ink_threshold
        # 閾値画像は単チャンネルなので、3チャンネルのキャンバスへ複製する
        paint = np.repeat(block[:, :, None], 3, axis=2)
        target = self.drawn[r * e:r * e + e, c * e:c * e + e]
        target[ink_region] = paint[ink_region]

    def _color_stamp(self, px: int, py: int, disk: np.ndarray) -> None:
        radius = self.cfg.brush_radius
        h, w = self.out_h, self.out_w
        y0, y1 = max(0, py - radius), min(h, py + radius + 1)
        x0, x1 = max(0, px - radius), min(w, px + radius + 1)
        if y1 <= y0 or x1 <= x0:
            return
        by0, by1 = y0 - (py - radius), disk.shape[0] - ((py + radius + 1) - y1)
        bx0, bx1 = x0 - (px - radius), disk.shape[1] - ((px + radius + 1) - x1)
        m = disk[by0:by1, bx0:bx1]
        inv = 1.0 - m
        target = self.drawn[y0:y1, x0:x1]
        source = self.color_img[y0:y1, x0:x1].astype(np.float32)
        for ch in range(3):
            target[:, :, ch] = target[:, :, ch] * inv + source[:, :, ch] * m

    # ── 現在のキャンバスのスナップショット（ペン先を含む）を何フレームか書き出す ──
    def _snapshot_with_tip(self, px: int, py: int) -> np.ndarray:
        snap = self.drawn.astype(np.uint8)  # astype はすでに新しい配列を返すため copy 不要
        if self.tip is not None:
            self.tip.stamp(snap, px, py)
        return snap

    def _build_stroke_samples(
        self, path: list[tuple[int, int]]
    ) -> tuple[list[tuple[int, int]], set[int], list[int]]:
        """
        筆跡の折れ線を補間し、連続したペン先のピクセル座標列にする。
        隣接するセル中心の間を sample_step ピクセルで一様にサンプリングし、途切れのない滑走軌跡を作る。

        戻り値は (samples, pen_lifts, sample_cell_index):
          samples         —— ペン先のピクセル座標リスト
          pen_lifts       —— 「ペンを持ち上げる」サンプル点のインデックス集合（隣接しないセルへ切り替わる箇所）
          sample_cell_index —— 各サンプル点が属する cell の path 内でのインデックス。
                              「インクの出現の進捗」と「ペン先の位置」を厳密に同期させるために使う。
        """
        samples: list[tuple[int, int]] = []
        pen_lifts: set[int] = set()
        sample_cell_index: list[int] = []
        for idx, cell in enumerate(path):
            cx, cy = self._cell_center(cell)
            if idx == 0:
                samples.append((cx, cy))
                sample_cell_index.append(idx)
                continue
            prev_cell = path[idx - 1]
            prev = self._cell_center(prev_cell)
            cell_distance = math.hypot(cell[0] - prev_cell[0], cell[1] - prev_cell[1])
            if cell_distance > math.sqrt(2):
                pen_lifts.add(len(samples))
                samples.append((cx, cy))
                sample_cell_index.append(idx)
                continue
            steps = max(
                1, int(math.hypot(cx - prev[0], cy - prev[1]) / self.cfg.sample_step)
            )
            for s in range(1, steps + 1):
                samples.append(
                    (int(prev[0] + (cx - prev[0]) * s / steps),
                     int(prev[1] + (cy - prev[1]) * s / steps))
                )
                sample_cell_index.append(idx)
        return samples, pen_lifts, sample_cell_index

    def _frame_progress_indices(self, n_steps: int, target_frames: int) -> list[int]:
        """
        n_steps 個のペン先位置と target_frames 個の目標フレームが与えられたとき、
        各目標フレームで取るべきペン先位置のインデックスを返す（一様な写像で軌跡全体をカバーする）。
        target_frames <= n_steps ならダウンサンプリング、> なら繰り返しサンプリングになる。
        target_frames <= 0（全体の尺≤凝視パートでこのパートにフレームがない場合など）は空を返し、フレームを一切生成しない。
        """
        if n_steps == 0 or target_frames <= 0:
            return []
        if target_frames == 1:
            return [n_steps - 1]
        return [
            round(f * (n_steps - 1) / (target_frames - 1))
            for f in range(target_frames)
        ]

    def _pause_frame_indices(
        self, target_frames: int, n_cells: int
    ) -> set[int]:
        """
        適応的ポーズ: 内容の密度で段階分けしてポーズ比率を決め、ポーズフレームをタイムライン上に均等に配置する。
        戻り値は「フリーズ（直前フレームの進捗を繰り返す）が必要な」フレームインデックスの集合。

        段階分けの指標には「1セルあたりのフレーム数」frames_per_cell = target_frames / n_cells を使う:
        セル数よりフレーム数が大幅に多い（値が大きい）なら内容に対して尺に余裕がある → ポーズを多くしてペンの持ち替えの呼吸を再現。
        セル数よりフレーム数が少ない（値が小さい）なら内容が密で尺が厳しい → ポーズなし。

        pause_mode で強制的に上書きできる: "off" 無効、"light"/"heavy" 段階固定、"auto" 自動。
        """
        mode = self.cfg.pause_mode
        if mode == "off" or target_frames < 8 or n_cells <= 0:
            return set()

        if mode == "light":
            ratio = self.cfg.pause_ratio_light
        elif mode == "heavy":
            ratio = self.cfg.pause_ratio_heavy
        else:  # auto: 「1セルあたりのフレーム数」で自動的に段階分けする
            fpc = target_frames / n_cells
            if fpc >= self.cfg.pause_heavy_fpc:
                ratio = self.cfg.pause_ratio_heavy
            elif fpc >= self.cfg.pause_light_fpc:
                ratio = self.cfg.pause_ratio_light
            else:
                return set()  # 内容が密: テンポを速くしてポーズなし

        # ポーズフレーム数は最低 0。先頭と末尾のフレームでポーズしないよう target_frames-2 にクランプする
        pause_count = min(
            max(0, int(round(target_frames * ratio))),
            max(0, target_frames - 2),
        )
        if pause_count <= 0:
            return set()

        # 等分して挿入: target_frames を pause_count+1 等分し、ポーズを内部の分割点に置く
        # 先頭と末尾は使わない(1/(n+1) の分子を 1 から始める)ことで、冒頭と結末が途切れないようにする
        return {
            max(1, min(target_frames - 2,
                       round((idx + 1) * target_frames / (pause_count + 1))))
            for idx in range(pause_count)
        }

    # ── 起筆パート: stroke_path に沿って線画を敷き、ペン先の滑走とインクの出現を厳密に同期させる ──
    def lay_down_ink(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        """起筆パートのエントリポイント: ink_path_mode に応じてスケルトン追跡かグリッドセルパスへ振り分ける。"""
        if self.cfg.ink_path_mode == "skeleton" and self.skeleton_strokes:
            return self._lay_down_ink_skeleton(writer, target_frames)
        return self._lay_down_ink_grid(writer, target_frames)

    # ── grid モード: グリッドのセル中心を補間したパスに沿ってインクを出現させる（従来ロジック）──
    def _lay_down_ink_grid(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        path = self.stroke_path
        n = len(path)
        if n == 0:
            print("  インクがないため、起筆パートをスキップします")
            for _ in range(target_frames):
                writer.write(self._snapshot_with_tip(self.out_w // 2, self.out_h // 2))
            return

        samples, pen_lifts, sample_cell_index = self._build_stroke_samples(path)
        sample_idx_for_frame = self._frame_progress_indices(len(samples), target_frames)

        # 適応的ポーズ: 密度で段階分けして「フリーズフレーム」（ペン先が止まり、インクの出現も進まない）を選び、
        # 人がペンを持ち替える/呼吸するリズムを再現する。
        pause_frames = self._pause_frame_indices(target_frames, n)
        if pause_frames:
            print(f"  適応的ポーズ: {len(pause_frames)} フレームをフリーズ (モード={self.cfg.pause_mode})")

        written = 0
        cells_revealed = 0  # ブロック単位で出現済みのセル数（増分、ペン先の進捗に厳密に追随する）
        last_sample_idx: int | None = None
        for fi, si in enumerate(sample_idx_for_frame):
            # ポーズフレーム: 直前フレームのペン先位置と進捗を再利用し、インクは出現させず、スナップショットを1フレームだけ書く（ペン先はフリーズ）
            if fi in pause_frames and last_sample_idx is not None:
                sx, sy = samples[last_sample_idx]
                writer.write(self._snapshot_with_tip(sx, sy))
                written += 1
                if (fi + 1) % max(1, target_frames // 10) == 0:
                    print(f"  起筆の進捗: {int((fi + 1) / target_frames * 100)}%")
                continue

            # ペン先の軌跡に沿って出現させる（筆跡の流れる感じを保つ）
            if last_sample_idx is None:
                self._reveal_ink_segment(samples[si], samples[si])
            else:
                for sample_idx in range(last_sample_idx + 1, si + 1):
                    if sample_idx in pen_lifts:
                        continue
                    self._reveal_ink_segment(
                        samples[sample_idx - 1], samples[sample_idx]
                    )

            # ブロック単位の出現: 「現在ペン先がいる cell」まで厳密に出現させ、ペン先と描画の同期と文字の完全さを保証する。
            # sample_cell_index[si] は現在のフレームでペン先が属するセルのインデックスで、そこまで出現させる。
            target_cell = sample_cell_index[si]
            while cells_revealed <= target_cell and cells_revealed < n:
                self._ink_stamp(path[cells_revealed])
                cells_revealed += 1

            sx, sy = samples[si]
            writer.write(self._snapshot_with_tip(sx, sy))
            written += 1
            last_sample_idx = si
            if (fi + 1) % max(1, target_frames // 10) == 0:
                print(f"  起筆の進捗: {int((fi + 1) / target_frames * 100)}%")

        # 仕上げの保険: すべてのセルのインクが完全に出現していることを保証し、フレーム数を埋める
        while cells_revealed < n:
            self._ink_stamp(path[cells_revealed])
            cells_revealed += 1
        last = samples[-1]
        while written < target_frames:
            writer.write(self._snapshot_with_tip(*last))
            written += 1
        print(f"  起筆完了: {n} セル, {written} フレーム")

    # ── skeleton モード: スケルトンのピクセルパスに沿ってインクを出現させる（ペン先が実際のスケルトンをたどる）──
    def _lay_down_ink_skeleton(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        """
        スケルトンモードの起筆: ペン先がスケルトンのピクセル点に沿って滑り、_reveal_ink_segment で元画像のインクを出現させる。
        スケルトンはすでにピクセル単位で正確なため、セルの完全さを保証する必要がなく、ブロック単位の出現（_ink_stamp）は行わない。
        筆画をまたぐ箇所ではペンを持ち上げる（pen_lifts）印をつけ、補間をスキップする。
        """
        strokes = self.skeleton_strokes
        if not strokes:
            return self._lay_down_ink_grid(writer, target_frames)

        # 複数の筆画を連続したサンプル点列に平坦化し、筆画をまたぐ箇所にペンを持ち上げる印をつける
        samples: list[tuple[int, int]] = []
        pen_lifts: set[int] = set()
        for si, stroke in enumerate(strokes):
            if si > 0:
                pen_lifts.add(len(samples))  # 筆画の間でペンを持ち上げる
            samples.extend(stroke)

        n = len(samples)
        if n == 0:
            print("  スケルトンの筆画がないため、起筆パートをスキップします")
            for _ in range(target_frames):
                writer.write(self._snapshot_with_tip(self.out_w // 2, self.out_h // 2))
            return

        sample_idx_for_frame = self._frame_progress_indices(n, target_frames)

        # 適応的ポーズ（密度判定にはセル数ではなく筆画数を使う）
        pause_frames = self._pause_frame_indices(target_frames, len(strokes))
        if pause_frames:
            print(f"  適応的ポーズ: {len(pause_frames)} フレームをフリーズ (モード={self.cfg.pause_mode})")

        written = 0
        last_sample_idx: int | None = None
        report_step = max(1, target_frames // 10)
        for fi, si in enumerate(sample_idx_for_frame):
            # ポーズフレーム: ペン先をフリーズする
            if fi in pause_frames and last_sample_idx is not None:
                sx, sy = samples[last_sample_idx]
                writer.write(self._snapshot_with_tip(sx, sy))
                written += 1
                if (fi + 1) % report_step == 0:
                    print(f"  起筆の進捗: {int((fi + 1) / target_frames * 100)}%")
                continue

            # スケルトンに沿ってインクを出現させる: 前フレームのサンプル点から現フレームのサンプル点まで、区間ごとに元画像のインクを出現させる
            if last_sample_idx is None:
                self._reveal_ink_segment(samples[si], samples[si])
            else:
                for idx in range(last_sample_idx + 1, si + 1):
                    if idx in pen_lifts:
                        continue
                    self._reveal_ink_segment(samples[idx - 1], samples[idx])

            sx, sy = samples[si]
            writer.write(self._snapshot_with_tip(sx, sy))
            written += 1
            last_sample_idx = si
            if (fi + 1) % report_step == 0:
                print(f"  起筆の進捗: {int((fi + 1) / target_frames * 100)}%")

        # 仕上げの保険: フレーム数を埋める
        last = samples[-1]
        while written < target_frames:
            writer.write(self._snapshot_with_tip(*last))
            written += 1
        print(f"  起筆完了(スケルトン): {n} サンプル点, {written} フレーム")

    # ── 着彩パートのエントリポイント: color_fill に応じて対応するスタイルへ振り分ける ──
    def wash_color(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        if self.cfg.color_fill == "contour-wipe":
            return self.wash_color_contour(writer, target_frames)
        return self.wash_color_brush(writer, target_frames)

    # ── brush: 筆画の軌跡に沿って円形のインクブラシで原色を敷く（既定のスタイル）──
    def wash_color_brush(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        path = self.stroke_path
        n = len(path)
        disk = _feathered_disk(self.cfg.brush_radius)
        if n == 0:
            print("  インクがないため、着彩パートをスキップします")
            gaze = self.color_img
            for _ in range(target_frames):
                writer.write(gaze)
            return

        centers = [self._cell_center(cell) for cell in path]
        cell_idx_for_frame = self._frame_progress_indices(n, target_frames)

        written = 0
        last_cell_idx: int | None = None
        for fi, ci in enumerate(cell_idx_for_frame):
            # 現在のフレームの進捗に合わせて着彩する: 前フレームのセルから現在のセルまで塗り足し、
            # ダウンサンプリングで飛ばされうる中間のセルもまとめて塗ることで原色の連続性を保つ。
            if last_cell_idx is None:
                self._color_stamp(*centers[ci], disk)
            else:
                for cell_idx in range(last_cell_idx + 1, ci + 1):
                    self._color_stamp(*centers[cell_idx], disk)

            cx, cy = centers[ci]
            writer.write(self._snapshot_with_tip(cx, cy))
            written += 1
            last_cell_idx = ci
            if (fi + 1) % max(1, target_frames // 10) == 0:
                print(f"  着彩の進捗: {int((fi + 1) / target_frames * 100)}%")

        # 仕上げの保険
        last = centers[-1]
        while written < target_frames:
            writer.write(self._snapshot_with_tip(*last))
            written += 1
        print(f"  着彩完了: {n} セル, {written} フレーム")

    # ── contour-wipe: 輪郭を認識しながら上から下へ走査して着彩する ──
    def wash_color_contour(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        """
        色は筆画の軌跡に沿って塗るのではなく、全体を上から下へ一度に走査する出現の先端で塗る。
        先端は輪郭に当たるとまず引っかかり（抵抗≈1 で delay_px 分差し引かれる）、
        その下方に伸びる減衰する影に従ってゆっくり越えていくため、
        「色が線に沿って広がっていく」印象になる。ペン先は横方向に往復して掃き、手で塗っている様子を再現する。
        """
        cfg = self.cfg
        h, w = self.out_h, self.out_w

        if target_frames <= 0:
            print("  着彩フレームがないため、contour-wipe パートをスキップします")
            return

        # 一度だけ事前計算する: 抵抗場、波の境界、差し引くピクセル数、行座標のグリッド
        resistance = self._build_resistance_field()
        wave = _build_wipe_wave(w)
        delay_px = int(np.clip(h * cfg.wipe_delay_ratio, 12, 52))
        blocks = max(1, cfg.wipe_blocks)
        ys = np.arange(h, dtype=np.float32)[:, None]   # (H,1)、フレームごとに再利用する

        # self.drawn を「線画が描き終わった」状態に戻す（brush は lay_down_ink の後に続くので、ここでも同様に続ける）
        # color_img が出現の目標となる
        color_src = self.color_img.astype(np.float32)

        print(f"  contour-wipe: {w}x{h}, delay_px={delay_px}, 往復数={blocks}")

        written = 0
        # 出現の先端は -delay_px から h+delay_px まで走査し、全体をカバーする
        sweep = h + 2 * delay_px
        report_step = max(1, target_frames // 10)

        for fi in range(target_frames):
            # 全体の進捗（サインイージングつき）: 0 → 1
            if target_frames == 1:
                progress = 1.0
            else:
                progress = fi / (target_frames - 1)
            lead = _ease_in_out_sine(progress) * sweep - delay_px

            # 出現マスク: y <= lead + wave[x] - resistance[y,x]*delay_px
            threshold = lead + wave[None, :] - resistance * delay_px  # (H,W)
            reveal = ys <= threshold                        # (H,W) bool

            # 原色を drawn バッファへ出現させる
            self.drawn[reveal] = color_src[reveal]

            # ペン先の横方向の掃き: blocks 回の往復、奇数回目は逆方向
            lane = (fi / blocks * 2.0) % 1.0               # 片道の正規化進捗 0..1
            lane = _ease_in_out_sine(lane)
            forward = (int(fi // blocks) % 2 == 0)         # 偶数回目は順方向、奇数回目は逆方向
            cursor_x = int(lane * w) if forward else int((1.0 - lane) * w)
            cursor_x = max(0, min(w - 1, cursor_x))

            # カーソルの y = 現在の列で出現済みピクセルの最も下の行
            col_revealed = np.where(reveal[:, cursor_x])[0]
            cursor_y = int(col_revealed[-1]) if col_revealed.size > 0 else 0

            writer.write(self._snapshot_with_tip(cursor_x, cursor_y))
            written += 1
            if (fi + 1) % report_step == 0:
                print(f"  着彩の進捗(contour-wipe): {int((fi + 1) / target_frames * 100)}%")

        # 仕上げの保険: 画像全体が出現済みであることを保証する（最終フレームは進捗=1 で lead≈h+delay_px となり、理論上は全面をカバーする）
        full_reveal = np.ones((h, w), dtype=bool)
        self.drawn[full_reveal] = color_src[full_reveal]
        last = self._snapshot_with_tip(w // 2, h - 1)
        while written < target_frames:
            writer.write(last)
            written += 1
        print(f"  contour-wipe 完了: {written} フレーム")

    def render_to(self, raw_path: Path, total_ms: int) -> Path:
        cfg = self.cfg
        plan = plan_phases(total_ms, cfg)
        ink_cells = len(self.stroke_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_path), fourcc, cfg.fps, (self.out_w, self.out_h))

        print(f"  インクの流れ: {len(self.ink_streams)} 本, インクのセル: {ink_cells}")
        print(
            f"  尺: {total_ms}ms -> 起筆 {plan.ink_frames}f / "
            f"着彩 {plan.color_frames}f / 凝視 {plan.gaze_frames}f (重み {plan.ratio_label})"
        )

        started = time.time()
        self.lay_down_ink(writer, plan.ink_frames)
        self.wash_color(writer, plan.color_frames)
        # 凝視: 完成した元画像
        gaze_img = self.color_img
        for _ in range(plan.gaze_frames):
            writer.write(gaze_img)
        writer.release()
        print(f"  レンダリング所要時間: {time.time() - started:.1f}s")
        return raw_path


# ──────────────────────────────────────────────────────────────
# トランスコード（システムの ffmpeg を優先、PyAV を代替とし、どちらもなければ mp4v のまま）
# ──────────────────────────────────────────────────────────────
def transcode_h264(src: Path, dst: Path) -> Path:
    """
    mp4v の元動画を H.264（yuv420p）へトランスコードし、プレイヤーの互換性を高める。

    優先順位:
      1. システムの ffmpeg サブプロセス（エンコード効率が最も高くファイルサイズも最小、CRF=20）
      2. PyAV（pip だけでインストールでき、システムの ffmpeg が不要。エンコード効率はやや劣るため CRF=28 でサイズを抑える）
      3. どちらもない場合: 元の mp4v エンコードのまま残し、警告を出す
    """
    # 経路1: システムの ffmpeg（推奨、サイズが最適）
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(src),
            "-c:v", "libx264",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(dst),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            src.unlink(missing_ok=True)
            print(f"  H.264 トランスコード完了(ffmpeg): {dst}")
            return dst
        print(f"  [warn] ffmpeg のトランスコードに失敗しました: {res.stderr.strip()}")

    # 経路2: PyAV（代替、pip だけでインストール可能）
    try:
        return _transcode_with_pyav(src, dst)
    except ImportError:
        pass
    except Exception as e:
        print(f"  [warn] PyAV のトランスコードに失敗しました: {e}")

    # 経路3: どちらもないため mp4v のまま残す
    print(f"  [warn] ffmpeg と PyAV が見つかりません。元の mp4v エンコードのまま残します: {src}")
    print(f"         どちらかを入れれば H.264 になります: pip install av  もしくは  システムに ffmpeg をインストール")
    return src


def _transcode_with_pyav(src: Path, dst: Path) -> Path:
    """
    PyAV を使って Python 内で H.264 トランスコードを行う。PyAV が未インストールなら ImportError を投げる。
    PyAV 同梱の libx264 はシステムの ffmpeg よりエンコード効率が低い（同じ CRF ならサイズが数倍になる）ため、
    CRF=28 でサイズと画質のバランスを取る。
    """
    import av
    input_container = av.open(str(src), mode="r")
    in_stream = input_container.streams.video[0]
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height
    fps = in_stream.average_rate

    output_container = av.open(str(dst), mode="w")
    out_stream = output_container.add_stream("h264", rate=fps)
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "28", "preset": "medium"}

    for frame in input_container.decode(video=0):
        packet = out_stream.encode(frame)
        if packet:
            output_container.mux(packet)
    # flush
    packet = out_stream.encode(None)
    if packet:
        output_container.mux(packet)

    output_container.close()
    input_container.close()
    src.unlink(missing_ok=True)
    print(f"  H.264 トランスコード完了(PyAV): {dst}")
    return dst


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="1枚の画像をストリーム筆跡ホワイトボードアニメーション動画にレンダリングする"
    )
    p.add_argument("image", help="入力画像のパス (PNG/JPG/JPEG/BMP/TIFF)")
    p.add_argument("--out-dir", default="./out", help="出力ディレクトリ (既定: ./out)")
    p.add_argument("--total-ms", type=int, default=10000, help="動画の総尺、単位はミリ秒 (既定: 10000)")
    p.add_argument("--bare-tip", action="store_true", help="ペン先/手のオーバーレイを重ねない")
    p.add_argument(
        "--pen-image", default=str(DEFAULT_HAND_PNG),
        help="カスタムのペン先/手の素材パス (既定: skill 内蔵の drawing-hand.png)",
    )
    p.add_argument("--fps", type=int, default=None, help="既定のフレームレートを上書きする")
    p.add_argument("--grid-edge", type=int, default=None, help="既定のグリッド辺長を上書きする")
    p.add_argument("--brush-radius", type=int, default=None, help="既定のインクブラシ半径を上書きする")
    p.add_argument(
        "--color-fill", default="contour-wipe", choices=["brush", "contour-wipe"],
        help="着彩フェーズの着彩スタイル: contour-wipe 輪郭を認識して上から下へ走査 (既定); brush 筆画の軌跡に沿って塗る",
    )
    p.add_argument(
        "--wipe-decay", type=float, default=None,
        help="contour-wipe: 抵抗場が1行ずつ下方向へ減衰する係数 (既定 0.86、小さいほど速く輪郭を越える)",
    )
    p.add_argument(
        "--wipe-delay-ratio", type=float, default=None,
        help="contour-wipe: 輪郭部で先端を差し引く比率×h (既定 0.04、大きいほど輪郭部での滞在が長い)",
    )
    p.add_argument(
        "--wipe-blocks", type=int, default=None,
        help="contour-wipe: ペン先が横方向に往復して掃く回数 (既定 18)",
    )
    p.add_argument(
        "--pause", default="heavy", choices=["auto", "off", "light", "heavy"],
        help="起筆パートのポーズのリズム: heavy はっきり(既定); auto 密度で自動的に段階分け; off 無効; light 少量",
    )
    p.add_argument(
        "--ink-path", default="grid", choices=["grid", "skeleton"],
        help="起筆パートの筆跡パス: grid グリッドのセル中心を補間(既定); skeleton スケルトンレベルのピクセル追跡(線により正確に沿う)",
    )
    return p.parse_args(argv)


def _build_cfg(args: argparse.Namespace) -> Config:
    kw: dict = {}
    if args.fps is not None:
        kw["fps"] = args.fps
    if args.grid_edge is not None:
        kw["grid_edge"] = args.grid_edge
    if args.brush_radius is not None:
        kw["brush_radius"] = args.brush_radius
    if args.color_fill is not None:
        kw["color_fill"] = args.color_fill
    if args.wipe_decay is not None:
        kw["wipe_decay"] = args.wipe_decay
    if args.wipe_delay_ratio is not None:
        kw["wipe_delay_ratio"] = args.wipe_delay_ratio
    if args.wipe_blocks is not None:
        kw["wipe_blocks"] = args.wipe_blocks
    if args.pause is not None:
        kw["pause_mode"] = args.pause
    if args.ink_path is not None:
        kw["ink_path_mode"] = args.ink_path
    return Config(**kw)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _build_cfg(args)

    print("=" * 56)
    print("ストリーム筆跡アニメーション レンダラー")
    print("=" * 56)

    image_bgr = _imread_any(args.image)
    if image_bgr is None:
        print(f"[err] 画像を読み込めません: {args.image}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"stream_{ts}.mp4"
    h264_path = out_dir / f"stream_{ts}_h264.mp4"

    pen_png = Path(args.pen_image) if args.pen_image else None
    renderer = StreamBoardRenderer(image_bgr, cfg, pen_png, args.bare_tip)
    print(f"  入力: {args.image}")
    print(f"  出力サイズ: {renderer.out_w}x{renderer.out_h}, フレームレート: {cfg.fps}")

    renderer.render_to(raw_path, args.total_ms)
    final = transcode_h264(raw_path, h264_path)

    size_mb = final.stat().st_size / (1024 * 1024)
    print(f"\n最終的な動画: {final}")
    print(f"  ファイルサイズ: {size_mb:.2f} MB")
    print("=" * 56)
    print("完了")
    # 最終行に最終パスを出力し、上位レイヤーが取得しやすいようにする
    print(f"OUTPUT={final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
