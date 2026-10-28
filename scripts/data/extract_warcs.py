"""Unpack ProdCrawl WARC files into the directory tree the mirror web server serves.

Every resource record becomes the file <out_dir>/<host>/<path>, where its
WARC-Target-URI is http://<host>/<path>, and keeps WARC-Date as its modification
time. This is the layout the experiments crawled: the server publishes <out_dir> as
its document root, so the page http://<host>/<path> is served at
<MIRROR_URL><host>/<path>.

Usage: python scripts/data/extract_warcs.py --out-dir /srv/prodcrawl <file.warc.gz> [...]
"""
import argparse
import calendar
import os
import sys
import time

from warcio.archiveiterator import ArchiveIterator


def target_path(uri: str, out_dir: str) -> str:
    if not uri.startswith("http://"):
        raise ValueError(f"unexpected WARC-Target-URI {uri!r}")
    rel = uri[len("http://"):]
    path = os.path.normpath(os.path.join(out_dir, rel))
    if not path.startswith(os.path.abspath(out_dir) + os.sep):
        raise ValueError(f"WARC-Target-URI escapes the output directory: {uri!r}")
    return path


def extract(warc_path: str, out_dir: str) -> int:
    n = 0
    with open(warc_path, "rb") as fh:
        for record in ArchiveIterator(fh):
            if record.rec_type != "resource":
                continue
            uri = record.rec_headers.get_header("WARC-Target-URI")
            path = target_path(uri, out_dir)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as out:
                stream = record.content_stream()
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    out.write(chunk)
            date = record.rec_headers.get_header("WARC-Date")
            ts = calendar.timegm(time.strptime(date[:19], "%Y-%m-%dT%H:%M:%S"))
            os.utime(path, (ts, ts))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("warcs", nargs="+")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for i, w in enumerate(args.warcs, 1):
        n = extract(w, out_dir)
        total += n
        print(f"[{i}/{len(args.warcs)}] {os.path.basename(w)}: {n} files", flush=True)
    print(f"{total} files written under {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
