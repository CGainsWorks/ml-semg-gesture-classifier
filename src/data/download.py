"""Fetch Ninapro DB5 and record what was fetched.

DB5 is served without an authentication wall, so this runs unattended. The
licence is not waived by that: it is research-use-only and the terms are
reproduced in ``data/DATASET_CARD.md``. Read them before using any of this
commercially.

Every download is checksummed and the digests are written to
``data/DB5_CHECKSUMS.txt`` so a later run can prove it read the same bytes.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.config import resolve
from src.data.ninapro import DB5_SUBJECTS, sha256, subject_zip_path, subject_zip_url

CHECKSUM_FILE = "data/DB5_CHECKSUMS.txt"
_CHUNK = 1 << 20


def _download(url: str, dest: Path, timeout: float = 120.0) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length", 0))
            written = 0
            with open(tmp, "wb") as fh:
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    written += len(block)
                    if total:
                        pct = 100.0 * written / total
                        print(
                            f"\r  {dest.name}: {written / 1e6:6.1f} / {total / 1e6:6.1f} MB "
                            f"({pct:5.1f}%)",
                            end="",
                            flush=True,
                        )
            print()
            if total and written != total:
                raise OSError(
                    f"{url}: expected {total} bytes, received {written}"
                )
    except (urllib.error.URLError, OSError):
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)


def read_checksums(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        out[name.strip()] = digest
    return out


def write_checksums(path: Path, digests: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# sha256 digests of the Ninapro DB5 subject archives as downloaded from",
        "# https://ninapro.hevs.ch/instructions/DB5.html",
        "# Regenerate with: python -m src.data.download --verify",
    ]
    lines += [f"{digests[name]}  {name}" for name in sorted(digests)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch(
    raw_dir: Path,
    subjects: list[int],
    force: bool = False,
    verify: bool = True,
) -> dict[str, str]:
    """Download the requested subjects and return their digests."""
    checksum_path = resolve(CHECKSUM_FILE)
    known = read_checksums(checksum_path)
    digests: dict[str, str] = dict(known)
    mismatches: list[str] = []

    for subject in subjects:
        dest = subject_zip_path(raw_dir, subject)
        if dest.exists() and not force:
            print(f"present: {dest.name}")
        else:
            url = subject_zip_url(subject)
            print(f"fetching {url}")
            _download(url, dest)

        if verify:
            digest = sha256(dest)
            digests[dest.name] = digest
            if dest.name in known and known[dest.name] != digest:
                mismatches.append(
                    f"{dest.name}: expected {known[dest.name]}, got {digest}"
                )

    if mismatches:
        raise RuntimeError(
            "downloaded archives do not match the recorded checksums:\n  "
            + "\n  ".join(mismatches)
        )

    if verify:
        write_checksums(checksum_path, digests)
        print(f"checksums written to {checksum_path}")
    return digests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/DB5")
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=DB5_SUBJECTS,
        help="subject numbers to fetch (default: all ten)",
    )
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="checksum every archive and refresh data/DB5_CHECKSUMS.txt",
    )
    args = parser.parse_args(argv)

    try:
        fetch(
            resolve(args.raw_dir),
            subjects=args.subjects,
            force=args.force,
            verify=args.verify or True,
        )
    except (RuntimeError, OSError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
