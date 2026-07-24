# The lab core is stdlib-only; this image adds the optional HTTP service (api)
# and the dev extra (pytest) so `docker run ... pytest` works too.
FROM python:3.12-slim

# git is needed to install the shared grounding grader from gradecore, which is a
# git dependency (the "one engine" is shared, not vendored). slim has no git.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python -m pip install --no-cache-dir -e ".[api,dev]"

EXPOSE 8000
# Default stays the eval (so CI's `docker run <image>` still checks the harness).
# To serve the API instead — what compose and Render do:
#   docker run -p 8000:8000 <image> uvicorn ragevallab.api:app --host 0.0.0.0 --port 8000
CMD ["python", "-m", "ragevallab.cli", "eval", "--out", "/app/eval_run.json"]
