#!/usr/bin/env python
"""Mirror a Drive results folder locally, small files only.

A finished batch is gigabytes of delta stores and rendered frames, and none of
the paper's tables are filled from either. This walks a Drive folder with the
v3 API and downloads only what the evidence collector reads: the decision,
aggregate, calibration, H3, provenance and nomination artefacts, plus the CSVs
that back them. Anything above ``--max-bytes`` or matching a skipped extension
is listed and not fetched, so the transfer is kilobytes rather than gigabytes.

Authentication is the caller's: it uses the active gcloud account's access
token, which needs the Drive scope. Without it the API returns 403 and this
says so rather than producing an empty tree.

    gcloud auth login --enable-gdrive-access
    python scripts/fetch_drive_evidence.py <folder-id-or-url> --out outputs/drive
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.googleapis.com/drive/v3"
FOLDER_MIME = "application/vnd.google-apps.folder"

# Everything the evidence collector reads, plus the tables behind them.
WANTED_SUFFIXES = (".json", ".csv", ".log", ".txt", ".md")
# Bulk artefacts: the delta store, frames, checkpoints, the traced graph.
SKIP_SUFFIXES = (".npz", ".npy", ".pt", ".pth", ".png", ".jpg", ".jpeg", ".mp4",
                 ".svg", ".html", ".zip", ".safetensors")
DEFAULT_MAX_BYTES = 8 * 1024 * 1024


def folder_id(value: str) -> str:
    """Accept a bare id or any of the Drive URL shapes."""
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", value) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", value)
    return match.group(1) if match else value.strip()


def access_token() -> str:
    try:
        out = subprocess.run(["gcloud", "auth", "print-access-token"],
                             capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise SystemExit("gcloud is not on PATH; this uses the active gcloud account's token.")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"gcloud could not mint a token: {exc.stderr.strip()}")
    return out.stdout.strip()


def api_get(path: str, token: str, **params):
    url = f"{API}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        if exc.code == 403 and "insufficient authentication scopes" in body:
            raise SystemExit(
                "The active gcloud token has no Drive scope.\n"
                "Grant it once, in your own shell:\n"
                "    gcloud auth login --enable-gdrive-access\n"
                "then re-run this command."
            )
        if exc.code == 404:
            raise SystemExit(
                f"Drive returned 404 for {path}. Either the id is wrong, or the active "
                "account cannot see that folder. Check `gcloud auth list`."
            )
        raise SystemExit(f"Drive API {exc.code} on {path}: {body[:400]}")


def walk(fid: str, token: str, prefix: Path = Path(".")) -> list[dict]:
    """Every file under a folder, depth first, with its relative path."""
    found, page = [], None
    while True:
        result = api_get("files", token,
                         q=f"'{fid}' in parents and trashed = false",
                         fields="nextPageToken, files(id, name, mimeType, size)",
                         pageSize=1000, supportsAllDrives="true",
                         includeItemsFromAllDrives="true",
                         **({"pageToken": page} if page else {}))
        for entry in result.get("files", []):
            rel = prefix / entry["name"]
            if entry["mimeType"] == FOLDER_MIME:
                found.extend(walk(entry["id"], token, rel))
            else:
                found.append({"id": entry["id"], "rel": rel,
                              "size": int(entry.get("size") or 0)})
        page = result.get("nextPageToken")
        if not page:
            return found


def wanted(entry: dict, max_bytes: int) -> tuple[bool, str]:
    suffix = entry["rel"].suffix.lower()
    if suffix in SKIP_SUFFIXES:
        return False, "bulk artefact"
    if suffix not in WANTED_SUFFIXES:
        return False, "not an evidence file"
    if entry["size"] > max_bytes:
        return False, f"{entry['size'] / 1024**2:.1f} MB over the cap"
    return True, ""


def download(entry: dict, token: str, out_root: Path) -> int:
    target = out_root / entry["rel"]
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API}/files/{entry['id']}?alt=media&supportsAllDrives=true"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request) as response, target.open("wb") as handle:
        payload = response.read()
        handle.write(payload)
    return len(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Drive folder id, or any Drive URL containing one.")
    parser.add_argument("--out", type=Path, required=True, help="Local directory to mirror into.")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--list-only", action="store_true",
                        help="Show what would be fetched, and what would be skipped, without fetching.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = access_token()
    fid = folder_id(args.folder)
    root = api_get(f"files/{fid}", token, fields="id,name,mimeType", supportsAllDrives="true")
    if root["mimeType"] != FOLDER_MIME:
        raise SystemExit(f"{root['name']} is not a folder ({root['mimeType']}).")
    print(f"Walking Drive folder {root['name']!r} ({fid})", flush=True)

    entries = walk(fid, token, Path(root["name"]))
    take, skip = [], []
    for entry in entries:
        ok, why = wanted(entry, args.max_bytes)
        (take if ok else skip).append((entry, why))
    skipped_bytes = sum(e["size"] for e, _ in skip)
    print(f"{len(entries)} files: fetching {len(take)}, skipping {len(skip)} "
          f"({skipped_bytes / 1024**3:.2f} GB of bulk artefacts)")

    if args.list_only:
        for entry, _ in take:
            print(f"  take  {entry['size']:>9,}  {entry['rel']}")
        for entry, why in skip[:20]:
            print(f"  skip  {entry['size']:>9,}  {entry['rel']}  ({why})")
        if len(skip) > 20:
            print(f"  ... and {len(skip) - 20} more skipped")
        return

    total = 0
    for n, (entry, _) in enumerate(take, 1):
        total += download(entry, token, args.out)
        if n % 25 == 0 or n == len(take):
            print(f"  {n}/{len(take)}  {total / 1024:.0f} KB", flush=True)
    print(f"\nMirrored {len(take)} files ({total / 1024:.0f} KB) into {args.out / root['name']}")
    print("Now run:  python scripts/collect_evidence.py "
          f"{args.out / root['name']}")


if __name__ == "__main__":
    main()
