#!/usr/bin/env python3
"""
ストリーミング手書きアニメーション - 環境セットアップスクリプト

役割:
  1. skill ディレクトリ配下に隔離された Python 仮想環境を作る（既にあれば再利用）
  2. 実行に必要なサードパーティライブラリが import できるか確認する
  3. 不足しているライブラリを自動で補う
  4. 最終行に ENV_PY=<インタプリタのパス> を出力し、呼び出し側が拾えるようにする

使い方:
  python prepare_env.py          # 環境構築 + 依存の補完を行い、ENV_PY を出力
  python prepare_env.py --check  # 確認のみ。不足があれば非ゼロの終了コードで終了
"""
from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

# skill のルートディレクトリ = このスクリプトから 2 階層上
SKILL_ROOT = Path(__file__).resolve().parent.parent
VENV_ROOT = SKILL_ROOT / ".venv"

# インタプリタ上の import 名 -> pip でのインストール名
DEPS: dict[str, str] = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "av": "av",  # PyAV: pip だけで入る H.264 エンコーダ。システムの ffmpeg は不要
    "PIL": "Pillow",  # render_annotation_preview.py が領域番号のプレビュー画像を描画する（日本語ラベルを含む）
}


def interpreter_path() -> Path:
    """仮想環境内の python 実行ファイルの場所（クロスプラットフォーム対応）。"""
    if sys.platform.startswith("win"):
        return VENV_ROOT / "Scripts" / "python.exe"
    return VENV_ROOT / "bin" / "python"


def ensure_venv(check_only: bool) -> Path:
    py = interpreter_path()
    if VENV_ROOT.exists() and py.exists():
        print(f"[ok] 既存の仮想環境を再利用: {VENV_ROOT}")
        return py

    if check_only:
        print(f"[err] 仮想環境がまだ作成されていません: {VENV_ROOT}")
        sys.exit(1)

    print(f"[..] 仮想環境を作成: {VENV_ROOT}")
    venv.create(str(VENV_ROOT), with_pip=True)
    print("[ok] 仮想環境の準備が完了")
    return py


def can_import(py: Path, import_name: str) -> bool:
    probe = subprocess.run(
        [str(py), "-c", f"import {import_name}"],
        capture_output=True,
    )
    return probe.returncode == 0


def install(py: Path, packages: list[str]) -> bool:
    if not packages:
        return True
    print(f"[..] 依存関係をインストール: {', '.join(packages)}")
    res = subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", *packages],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"[err] インストールに失敗:\n{res.stderr}")
        return False
    print("[ok] 依存関係のインストールが完了")
    return True


def main() -> None:
    check_only = "--check" in sys.argv

    py = ensure_venv(check_only)

    missing: list[str] = []
    for import_name, pip_name in DEPS.items():
        if can_import(py, import_name):
            print(f"[ok] {pip_name}")
        else:
            print(f"[miss] {pip_name}")
            missing.append(pip_name)

    if missing:
        if check_only:
            print(f"\n依存関係が {len(missing)} 件不足: {', '.join(missing)}")
            sys.exit(1)
        if not install(py, missing):
            sys.exit(1)

    # 最終行: 呼び出し側が拾うための取り決め出力
    print(f"\nENV_PY={py}")


if __name__ == "__main__":
    main()
