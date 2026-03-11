"""
run_all.py — 4개 모델(localvit, mlpmixer, poolformer, vit)을 순차적으로 학습시키는 스크립트.

상단 DEFAULTS 딕셔너리에서 공통 기본값을 설정하세요.
각 값은 CLI 인자로 train.py에 전달됩니다.
None으로 설정하면 해당 인자는 전달하지 않습니다 (config/base.yaml 기본값 사용).
"""

import subprocess
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# 기본값 설정 (None = base.yaml / 모델별 yaml 기본값 사용)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "project":          "sample",  # W&B project name
    "base_config":      "./config/base.yaml",
    "epochs":           None,               # None → base.yaml 값(500) 사용
    "save_best":        None,
    "seed":             None,
}

# 학습할 모델 목록 (순서대로 실행)
MODELS = [
    "vit",
    "localvit",
    "mlpmixer",
    "poolformer",
]

# 모델별 config 파일 경로
MODEL_CONFIGS = {
    "vit":        "./config/vit.yaml",
    "localvit":   "./config/localvit.yaml",
    "mlpmixer":   "./config/mlpmixer.yaml",
    "poolformer": "./config/poolformer.yaml",
}
# ─────────────────────────────────────────────────────────────────────────────


def build_cmd(model: str) -> list[str]:
    cmd = [sys.executable, "train.py"]

    if DEFAULTS.get("base_config"):
        cmd += ["--base_config", DEFAULTS["base_config"]]

    cmd += ["--config", MODEL_CONFIGS[model]]

    if DEFAULTS.get("project"):
        cmd += ["--project", DEFAULTS["project"]]

    if DEFAULTS.get("epochs") is not None:
        cmd += ["--epochs", str(DEFAULTS["epochs"])]

    if DEFAULTS.get("seed") is not None:
        cmd += ["--seed", str(DEFAULTS["seed"])]

    if DEFAULTS.get("save_best") is True:
        cmd += ["--save_best"]

    return cmd


def main():
    total = len(MODELS)
    results = {}

    for i, model in enumerate(MODELS, 1):
        cmd = build_cmd(model)
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] Starting: {model}")
        print(f"CMD: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        ret = subprocess.run(cmd)
        results[model] = ret.returncode

        if ret.returncode != 0:
            print(f"\n[WARNING] {model} exited with code {ret.returncode}. Continuing...\n")

    # 최종 요약
    print(f"\n{'='*60}")
    print("All runs complete. Summary:")
    for model, code in results.items():
        status = "OK" if code == 0 else f"FAILED (code {code})"
        print(f"  {model:<12} {status}")
    print(f"{'='*60}")

    if any(code != 0 for code in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
