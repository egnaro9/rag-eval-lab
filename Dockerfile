# Minimal image — the lab core is stdlib-only, so no pip install is required
# to run the eval. The default command runs the suite and prints the report.
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Install only the dev extra (pytest) so `docker run ... pytest` works too.
RUN python -m pip install --no-cache-dir -e ".[dev]"

# Default: run the eval and write the report to /app/eval_run.json
CMD ["python", "-m", "ragevallab.cli", "eval", "--out", "/app/eval_run.json"]
