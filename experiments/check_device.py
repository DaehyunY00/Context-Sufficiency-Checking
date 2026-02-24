from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def load_config(config_path: str | Path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def probe_mps_status():
    macos_version = str(platform.mac_ver()[0] or "").strip()
    built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    error = ""
    hint = ""
    if built and not available:
        try:
            _ = torch.ones(1, device="mps")
        except Exception as exc:
            error = str(exc)
        if macos_version:
            try:
                major = int(macos_version.split(".")[0])
            except ValueError:
                major = 0
            if major >= 13:
                if "dev" in str(torch.__version__).lower():
                    hint = (
                        "개발 버전(PyTorch nightly)에서 MPS 감지 이슈 가능성이 있습니다. "
                        "안정 버전으로 재설치 후 재확인하세요."
                    )
                else:
                    hint = (
                        "MPS 미활성은 macOS/파이썬/torch 조합 문제일 수 있습니다. "
                        "Python 3.11~3.12 환경에서 torch 안정 버전 재설치를 권장합니다."
                    )
    return {
        "torch_version": str(torch.__version__),
        "macos_version": macos_version,
        "mps_built": built,
        "mps_available": available,
        "probe_error": error,
        "diagnosis_hint": hint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPS/GPU 사용 가능 여부 점검")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device_pref = cfg.get("run", {}).get("device_preference", ["mps", "cpu"])

    status = probe_mps_status()
    print("[장치 점검 결과]")
    print(f"- torch 버전: {status['torch_version']}")
    print(f"- macOS 버전: {status['macos_version'] or 'unknown'}")
    print(f"- mps 빌드 포함 여부: {status['mps_built']}")
    print(f"- mps 사용 가능 여부: {status['mps_available']}")
    print(f"- 설정된 우선순위: {device_pref}")

    if status["mps_available"] and "mps" in [str(x).lower() for x in device_pref]:
        try:
            x = torch.ones(4, device="mps")
            y = (x * 2).sum().item()
            print(f"- MPS 텐서 연산 확인: 성공 (검증값={y})")
            print("\n결론: 현재 설정에서 Mac GPU(MPS)를 사용할 수 있습니다.")
        except Exception as exc:
            print(f"- MPS 텐서 연산 확인: 실패 ({exc})")
            print("\n결론: MPS가 감지되었지만 연산 테스트에 실패했습니다.")
        return

    print("\n결론: 현재 환경에서는 MPS를 사용할 수 없습니다. CPU로 동작합니다.")
    if status["probe_error"]:
        print(f"원인 힌트: {status['probe_error']}")
    if status["diagnosis_hint"]:
        print(f"추가 힌트: {status['diagnosis_hint']}")


if __name__ == "__main__":
    main()
