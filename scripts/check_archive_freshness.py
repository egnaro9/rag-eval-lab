#!/usr/bin/env python3
"""Fail if this suite's newest published run in eval-history has gone stale.

Why this exists. From 2026-08-30 to 2026-10-10 nothing was recorded, and every component
behaved correctly the whole time. The writer skipped because DATABASE_URL was empty, which
is right for a fork. eval-history's daily export connected, read the database, published,
and verified the served bytes by sha256, all green. The step here printed the newest date on
every run and asserted nothing, so a frozen pipeline and a working one looked identical.

Nobody's contract said "a row should have appeared recently." That is the gap, and freshness
is the only assertion that closes it: every other check can pass over a dead pipeline.

Exits 1 when the newest run for the suite is older than the limit, or when the archive cannot
be read. An unreachable archive is not a fresh one. The caller is continue-on-error, so a red
step here annotates the run without failing the build.

    python scripts/check_archive_freshness.py --suite rag-eval-lab --max-age-days 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_API = "https://erikhill.dev/eval-history"


def loud(msg: str) -> None:
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error title=eval-history archive is stale::{msg}", file=sys.stderr)
    else:
        print(f"ERROR: {msg}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--api", default=os.environ.get("EVAL_HISTORY_API", DEFAULT_API))
    ap.add_argument("--max-age-days", type=float, default=3.0,
                    help="the export is daily; GitHub's daily cron lands 4 to 7 hours late, "
                         "so anything beyond a couple of days means writes stopped")
    args = ap.parse_args()

    url = args.api.rstrip("/") + "/runs.json"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        # Deliberately not a pass. The old step ended in `|| echo ...`, so an unreachable
        # archive printed a reassuring sentence and returned echo's exit status.
        loud(f"could not read {url}: {type(e).__name__}: {e}")
        return 1

    rows = payload if isinstance(payload, list) else payload.get("runs", [])
    mine = [r for r in rows if r.get("name") == args.suite]
    if not mine:
        loud(f"the archive has no runs at all for suite {args.suite!r} "
             f"({len(rows)} runs published across every suite)")
        return 1

    newest = max(mine, key=lambda r: r.get("created_at", ""))
    stamp = newest.get("created_at", "")
    try:
        when = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        loud(f"newest {args.suite} run has an unparseable created_at: {stamp!r}")
        return 1

    age = dt.datetime.now(dt.timezone.utc) - when
    days = age.total_seconds() / 86400
    print(f"{len(mine)} {args.suite} runs published, newest {stamp} ({days:.1f} days old)")

    if days > args.max_age_days:
        loud(f"the newest published {args.suite} run is {days:.1f} days old, over the "
             f"{args.max_age_days} day limit. The export is daily and green, so the rows are "
             "not arriving: check that DATABASE_URL is set on this repository.")
        return 1

    print(f"ok: within the {args.max_age_days} day limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
