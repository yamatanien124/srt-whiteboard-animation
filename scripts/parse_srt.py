#!/usr/bin/env python3
"""
SRT 解析 + シーン分割の提案

.srt 字幕を構造化された字幕エントリに解析し、「1シーンあたり 25〜35 秒のナレーション」
という目安で字幕をシーンにグループ化して、各シーンの開始・終了時刻、合計尺
（→ sceneDurationMs）、テキストを出力する。

用途: srt-whiteboard-animation ワークフローのステップ 1 の入力材料として使う。
物語のイベントを読み取り、挿絵の方針を計画し、各画像の注釈に対する
sceneDurationMs を決定する。

使い方:
  python parse_srt.py <字幕.srt> [--target-sec 30] [--min-sec 25] [--max-sec 35]

出力: JSON（stdout）、フィールド:
  cues    各字幕エントリ: {index, startMs, endMs, durMs, text}
  scenes  提案シーン: {sceneIndex, startMs, endMs, sceneDurationMs, cueRange, text}
stderr には人間が読みやすいサマリを出力する。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def _to_ms(h: str, m: str, s: str, ms: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms.ljust(3, "0"))


def parse_srt(text: str) -> list[dict]:
    """SRT テキストを字幕エントリのリストに解析する。余分な空行・BOM・ミリ秒区切りのカンマ/ピリオドを許容する。"""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues: list[dict] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        # タイムコードを含む行を探す
        time_line_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if time_line_idx is None:
            continue
        times = _TIME.findall(lines[time_line_idx])
        if len(times) < 2:
            continue
        start = _to_ms(*times[0])
        end = _to_ms(*times[1])
        body = " ".join(lines[time_line_idx + 1:]).strip()
        cues.append({
            "index": len(cues) + 1,
            "startMs": start,
            "endMs": end,
            "durMs": max(0, end - start),
            "text": body,
        })
    return cues


def group_scenes(cues: list[dict], target_sec: float, min_sec: float, max_sec: float) -> list[dict]:
    """
    目標尺に従って連続する字幕をシーンにまとめる。target 付近まで積み上がったらシーンを区切るが、
    min 未満・max 超過にはしない（max を超える場合は強制的に区切る）。
    """
    scenes: list[dict] = []
    bucket: list[dict] = []
    target_ms, min_ms, max_ms = target_sec * 1000, min_sec * 1000, max_sec * 1000

    def flush() -> None:
        if not bucket:
            return
        start = bucket[0]["startMs"]
        end = bucket[-1]["endMs"]
        scenes.append({
            "sceneIndex": len(scenes) + 1,
            "startMs": start,
            "endMs": end,
            "sceneDurationMs": max(0, end - start),
            "cueRange": [bucket[0]["index"], bucket[-1]["index"]],
            "text": " ".join(c["text"] for c in bucket).strip(),
        })
        bucket.clear()

    for cue in cues:
        # このエントリを現在のシーンに入れると max を超える場合は、先にシーンを区切る（長すぎるシーンを避ける）
        if bucket:
            span_with = cue["endMs"] - bucket[0]["startMs"]
            if span_with > max_ms:
                flush()
        bucket.append(cue)
        span = bucket[-1]["endMs"] - bucket[0]["startMs"]
        # 目標に達し、かつ min を下回らなければシーンを区切る
        if span >= target_ms and span >= min_ms:
            flush()
    flush()
    return scenes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SRT 解析 + シーン分割の提案")
    p.add_argument("srt", help="字幕ファイルのパス (.srt)")
    p.add_argument("--target-sec", type=float, default=30.0, help="1シーンあたりの目標ナレーション秒数（デフォルト 30）")
    p.add_argument("--min-sec", type=float, default=25.0, help="1シーンの最短秒数（デフォルト 25）")
    p.add_argument("--max-sec", type=float, default=35.0, help="1シーンの最長秒数（デフォルト 35）")
    args = p.parse_args(argv)

    try:
        raw = Path(args.srt).read_text(encoding="utf-8-sig")
    except OSError as e:
        print(f"[err] 字幕を読み込めません: {e}", file=sys.stderr)
        return 1

    cues = parse_srt(raw)
    if not cues:
        print("[err] 字幕エントリを 1 件も解析できませんでした。SRT のフォーマットを確認してください", file=sys.stderr)
        return 1
    scenes = group_scenes(cues, args.target_sec, args.min_sec, args.max_sec)

    total_ms = cues[-1]["endMs"] - cues[0]["startMs"]
    print(f"字幕エントリ: {len(cues)}  合計尺: {total_ms/1000:.1f}s  提案シーン: {len(scenes)}", file=sys.stderr)
    for s in scenes:
        print(f"  シーン{s['sceneIndex']:>2}  {s['startMs']/1000:6.1f}-{s['endMs']/1000:6.1f}s "
              f"({s['sceneDurationMs']/1000:4.1f}s, 字幕{s['cueRange'][0]}-{s['cueRange'][1]}): "
              f"{s['text'][:40]}", file=sys.stderr)

    json.dump({"cues": cues, "scenes": scenes}, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
