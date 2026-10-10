#!/usr/bin/env python3
"""Record an eval_run.json in eval-history.

A CI utility, deliberately not part of the package: `ragevallab` produces a
run and knows nothing about where it gets stored. Keeping the coupling here
means the eval harness stays usable by anyone who has never heard of
eval-history, which is the whole reason eval_run.json is a plain file.

TWO MODES, because the hosted API went away.

  --url   POST to a running eval-history service. stdlib only. This is the
          mode to use if you host one.

  --db    Write straight to the Postgres eval-history reads, importing its own
          `evalhistory.app.ingest` so the schema mapping is not restated here.
          Needs `pip install evalhistory` and DATABASE_URL in the environment.

Render dropped its free tier on 2026-08-30 and took the hosted API with it.
This script kept POSTing to a dead host for 13 days; the job is
continue-on-error, so CI stayed green while every run went nowhere and the
archive quietly stopped gaining rag-eval-lab rows. --db exists so recording
does not depend on a process being up.

DATABASE_URL is read from the environment and never logged, never printed,
never passed as an argument. Do not add a --database-url flag: a connection
string on a command line lands in shell history and in CI logs.

Exits 0 and does nothing when neither a key nor a DATABASE_URL is set: forks
and pull requests don't get secrets, and that's a skip, not a failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# The free tier sleeps after ~15 minutes idle and takes ~30-50s to wake, so the
# first attempt is expected to be slow or to fail outright. Retrying is not
# papering over flakiness — it's the documented behaviour of the host.
TIMEOUT = 90
ATTEMPTS = 3
BACKOFF = 20


def post(url: str, key: str, payload: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/runs",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        # A 4xx is the server saying no. Retrying it would just be rude.
        return e.code, e.read().decode()


def _loud(msg: str) -> None:
    """Report a failure where someone will actually see it.

    This job is continue-on-error, deliberately: a recording outage should not
    block a library's CI. The cost showed up on 2026-08-30, when the host went
    away and every run failed for 13 days inside a green workflow. A GitHub
    annotation surfaces at the top of the run page instead of only in the job
    list, which is the difference between a failure being visible and being
    technically reported.
    """
    print(f"::error title=eval-history recording failed::{msg}"
          if os.environ.get("GITHUB_ACTIONS") else f"ERROR: {msg}", file=sys.stderr)


def write_direct(payload: dict) -> int:
    """Insert via eval-history's own ingest(), against DATABASE_URL.

    Imports are local so the --url mode stays stdlib-only and this script still
    runs on a machine with neither evalhistory nor a database driver installed.
    """
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        # This used to print "skipping (forks and PRs have no secrets)" and return 0.
        # That named the one cause that cannot apply here: the only caller is gated to
        # github.ref == refs/heads/main on push or dispatch, so a fork or a PR never
        # reaches it. The real cause was that the secret did not exist, and the message
        # explained it away for 41 days while nothing was recorded. A skip that supplies
        # its own excuse is worse than a silent one, because the excuse stops the reader.
        _loud("DATABASE_URL is empty, so this run was NOT recorded.")
        print("    Nothing here can tell you why it is empty. The two causes are:\n"
              "      - the secret is not set on this repository (Settings > Secrets and\n"
              "        variables > Actions, DATABASE_URL), or\n"
              "      - this context has no access to secrets, which for this job would\n"
              "        mean the workflow guard changed, because it only runs on main.\n"
              "    Returning nonzero so the step goes red. The job is continue-on-error,\n"
              "    so CI stays green and the annotation still reaches the summary.",
              file=sys.stderr)
        return 1
    try:
        from evalhistory.app import ingest
        from evalhistory.db import make_engine
        from evalhistory.schemas import RunIn
        from sqlalchemy.orm import Session
    except ImportError as e:
        _loud(f"--db needs eval-history installed: {e}")
        print("    pip install 'evalhistory @ git+https://github.com/egnaro9/eval-history'",
              file=sys.stderr)
        return 1

    # Do NOT build the engine by hand. evalhistory.db.make_engine already
    # rewrites the scheme to postgresql+psycopg (SQLAlchemy 2 defaults to
    # psycopg2, which is not what CI installs) and handles SQLite pooling. A
    # hand-rolled create_engine here failed with a bare ModuleNotFoundError for
    # psycopg2, which names the wrong problem entirely.
    try:
        engine = make_engine(url)
        with Session(engine) as db:
            run = ingest(db, RunIn.model_validate(payload))
            print(f"stored: {run.id}  ({payload.get('label')})")
    except Exception as e:                       # noqa: BLE001 - report loudly, then fail
        _loud(f"{type(e).__name__}: {e}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="base URL of a running eval-history service")
    ap.add_argument("--db", action="store_true",
                    help="write straight to DATABASE_URL instead of POSTing")
    ap.add_argument("--file", default="eval_run.json")
    ap.add_argument("--label", default=None, help="what made this run different")
    ap.add_argument("--git-sha", default=None)
    args = ap.parse_args()

    if not args.db and not args.url:
        print("need --db or --url", file=sys.stderr)
        return 2

    payload = json.load(open(args.file))
    if args.label:
        payload["label"] = args.label[:200]
    if args.git_sha:
        payload["git_sha"] = args.git_sha[:40]

    if args.db:
        return write_direct(payload)

    key = os.environ.get("EVAL_HISTORY_WRITE_KEY", "").strip()
    if not key:
        # Still a skip: this path genuinely does run where secrets are absent. Only the
        # cause claim is gone, because an empty variable does not say why it is empty.
        print("EVAL_HISTORY_WRITE_KEY is empty, so nothing was posted. Expected in a fork "
              "or a PR, which have no secrets; otherwise the secret is not set.")
        return 0

    for attempt in range(1, ATTEMPTS + 1):
        try:
            status, body = post(args.url, key, payload)
        except Exception as e:                       # noqa: BLE001 - report, then retry
            status, body = 0, f"{type(e).__name__}: {e}"

        if status == 201:
            print(f"stored: {json.loads(body)['id']}  ({payload.get('label')})")
            return 0
        if 400 <= status < 500:
            print(f"refused with {status}: {body}", file=sys.stderr)
            return 1                                 # our fault; retrying won't fix it

        print(f"attempt {attempt}/{ATTEMPTS} failed ({status or 'no response'}): {body[:120]}",
              file=sys.stderr)
        if attempt < ATTEMPTS:
            print(f"  waiting {BACKOFF}s — the free tier is probably still waking up",
                  file=sys.stderr)
            time.sleep(BACKOFF)

    print("could not reach eval-history", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
