"""Flask app entry point.

Wires the two blueprints that the runtime needs:
  - gateway.web.bp        — signup SPA + admin JSON API
  - gateway.bluebubbles.bp — /bluebubbles/webhook for inbound iMessage
"""
from __future__ import annotations

import logging

from flask import Flask, Response

from core.article_fetcher import init_cache
from core.user_profile import init_db
from gateway.bluebubbles import bp as bluebubbles_bp
from gateway.web import bp as web_bp

log = logging.getLogger(__name__)


app = Flask(__name__)
app.register_blueprint(web_bp)
app.register_blueprint(bluebubbles_bp)


@app.get("/healthz")
def healthz() -> Response:
    return Response("ok", mimetype="text/plain")


def main() -> None:
    import os
    logging.basicConfig(level=logging.INFO)
    init_db()
    init_cache()
    port = int(os.getenv("PORT", "5000"))
    log.info("serving on http://localhost:%d (signup at /, BlueBubbles webhook at /bluebubbles/webhook)", port)
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
