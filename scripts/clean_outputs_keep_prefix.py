from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="outputs 정리: 지정 prefix 외 파일/디렉토리 삭제")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument(
        "--keep-prefixes",
        type=str,
        default="llama3_seed10,flare_tradeoff,gh1_bias_check,manual_paraphrase_subset",
        help="콤마 구분 prefix 목록",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="확인 없이 실제 삭제")
    return parser.parse_args()


def _parse_prefixes(text: str) -> List[str]:
    out = [x.strip() for x in str(text).split(",") if x.strip()]
    return sorted(set(out))


def _should_keep(name: str, keep_prefixes: List[str]) -> bool:
    if name in {".gitkeep", ".DS_Store"}:
        return True
    if name == "logs":
        return True
    return any(name.startswith(p) for p in keep_prefixes)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    keep_prefixes = _parse_prefixes(args.keep_prefixes)
    if not outputs_dir.exists():
        print(f"[정보] outputs 디렉토리가 없어 정리할 항목이 없습니다: {outputs_dir}")
        return

    targets = []
    for p in sorted(outputs_dir.iterdir()):
        if _should_keep(p.name, keep_prefixes):
            continue
        targets.append(p)

    print(f"[정리대상] 총 {len(targets)}개")
    for p in targets[:30]:
        print(f"- {p}")
    if len(targets) > 30:
        print(f"... (추가 {len(targets)-30}개)")

    if args.dry_run:
        print("[dry-run] 삭제는 수행하지 않았습니다.")
        return

    if not args.yes:
        print("[중단] --yes 옵션이 없어 실제 삭제를 수행하지 않았습니다.")
        return

    deleted = 0
    for p in targets:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink(missing_ok=True)
            except TypeError:
                if p.exists():
                    p.unlink()
        deleted += 1

    print(f"[완료] {deleted}개 항목 삭제 완료")


if __name__ == "__main__":
    main()
