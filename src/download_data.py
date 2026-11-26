"""Reproducibly download and verify the MIMIC-IV Clinical Database Demo (v2.2).

The demo is open access (ODbL) on PhysioNet:
    https://physionet.org/content/mimic-iv-demo/2.2/

Every file is verified against the SHA-256 checksums that PhysioNet publishes for
release 2.2 (pinned below), so a silently changed or truncated file fails loudly.

Usage:
    python src/download_data.py [--out data/raw] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://physionet.org/files/mimic-iv-demo/2.2/"

# SHA-256 checksums from https://physionet.org/files/mimic-iv-demo/2.2/SHA256SUMS.txt
TABLES: dict[str, str] = {
    "hosp/patients.csv.gz": "59dcc6cd8450ddf03ad32cc7d76e41caa4c9633541715987f590c74835cf6fae",
    "hosp/admissions.csv.gz": "910b9f160ffdf1e08ea673585393f347c773ccc87d66875c627584a903ae8493",
    "hosp/diagnoses_icd.csv.gz": "958565670c0b3903c0c12825366dc2bae8561d2d55f16feac0c949dd630ed3e4",
    "hosp/labevents.csv.gz": "d51a4cf1ea63245abe7d544b0d50a928e9ee2ec5eb9a5ce85500aed0bd6c6dfc",
    "hosp/d_labitems.csv.gz": "72ea5f020469fd24543bd98ab3bd0a4f645a5a4b7802d9d52af20d736f9d76db",
    "icu/icustays.csv.gz": "e05e81aa52a3022e522b6832a898101a69b84e64c17bb344d819e458d5bc21b3",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "raw"

log = logging.getLogger("download_data")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
            return
        except requests.RequestException as exc:
            wait = 2**attempt
            log.warning("Attempt %d/%d failed for %s (%s); retrying in %ds",
                        attempt, retries, url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def fetch_all(out_dir: Path = DEFAULT_OUT, force: bool = False) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for rel, expected in TABLES.items():
        dest = out_dir / rel
        if dest.exists() and not force and sha256(dest) == expected:
            log.info("OK (cached)  %s", rel)
        else:
            log.info("Downloading  %s", rel)
            download(BASE_URL + rel, dest)
            actual = sha256(dest)
            if actual != expected:
                dest.unlink(missing_ok=True)
                raise ValueError(f"Checksum mismatch for {rel}: expected {expected}, got {actual}")
            log.info("Verified     %s (%.1f KB)", rel, dest.stat().st_size / 1024)
        paths[rel] = dest
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory (default: data/raw)")
    parser.add_argument("--force", action="store_true", help="Re-download even if verified files exist")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_all(args.out, args.force)
    log.info("All %d tables downloaded and SHA-256 verified in %s", len(TABLES), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
