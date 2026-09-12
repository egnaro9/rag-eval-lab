"""Cloudflare Worker entrypoint for the rag-eval-lab API.

The app object is ragevallab.api:app unchanged. This port was cheap because the
service was already shaped for it: no database, no data files, no entropy at
import, and uvicorn already an optional extra rather than a runtime dependency.

The only code change was making the four route handlers `async def`. Starlette
offloads a sync handler to a threadpool, and an isolate has no threads, so every
request would 500 with "can't start new thread".

There is no static assets binding here, unlike crashkit. This service has no
frontend: "/" redirects to /docs, which FastAPI renders itself.
"""
from workers import WorkerEntrypoint

from ragevallab.api import app


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        import asgi
        return await asgi.fetch(app, request, self.env)
