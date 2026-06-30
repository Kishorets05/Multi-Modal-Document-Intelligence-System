"""EntityRuler patterns for technical skills and tools.

All patterns are registered here under the ``TECH_SKILL`` label.
No extractor should hardcode skill names — use this module as the
single source of truth.

The patterns are intentionally case-insensitive by using both the
canonical form and lowercase equivalents, so that "react", "React",
and "REACT" all match.
"""
from __future__ import annotations

from typing import Final

# Each item is {"label": "TECH_SKILL", "pattern": ...}.
# Patterns can be:
#   str       → matched as a single token (exact, case-sensitive by spaCy)
#   list[dict] → matched as a token sequence (more flexible)
#
# We use the "lower" attribute to achieve case-insensitive matching on
# individual tokens without requiring a lowercased pipeline.

def _tok(text: str) -> dict:
    """Return a single-token pattern dict that matches case-insensitively."""
    return {"lower": text.lower()}


def _phrase(*words: str) -> list[dict]:
    """Return a multi-token pattern list for a phrase."""
    return [{"lower": w.lower()} for w in words]


# ─────────────────────────────────────────────────────────────────────────── #
#  Pattern definitions                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

_PATTERNS: Final[list[dict]] = [

    # ── Programming Languages ──────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("python")]},
    {"label": "TECH_SKILL", "pattern": [_tok("java")]},
    {"label": "TECH_SKILL", "pattern": [_tok("javascript")]},
    {"label": "TECH_SKILL", "pattern": [_tok("typescript")]},
    {"label": "TECH_SKILL", "pattern": [_tok("c++")]},
    {"label": "TECH_SKILL", "pattern": [_tok("c#")]},
    {"label": "TECH_SKILL", "pattern": [_tok("go")]},
    {"label": "TECH_SKILL", "pattern": [_tok("golang")]},
    {"label": "TECH_SKILL", "pattern": [_tok("rust")]},
    {"label": "TECH_SKILL", "pattern": [_tok("kotlin")]},
    {"label": "TECH_SKILL", "pattern": [_tok("swift")]},
    {"label": "TECH_SKILL", "pattern": [_tok("r")]},
    {"label": "TECH_SKILL", "pattern": [_tok("scala")]},
    {"label": "TECH_SKILL", "pattern": [_tok("php")]},
    {"label": "TECH_SKILL", "pattern": [_tok("ruby")]},
    {"label": "TECH_SKILL", "pattern": [_tok("dart")]},
    {"label": "TECH_SKILL", "pattern": [_tok("perl")]},
    {"label": "TECH_SKILL", "pattern": [_tok("haskell")]},
    {"label": "TECH_SKILL", "pattern": [_tok("elixir")]},
    {"label": "TECH_SKILL", "pattern": [_tok("clojure")]},
    {"label": "TECH_SKILL", "pattern": [_tok("matlab")]},
    {"label": "TECH_SKILL", "pattern": [_tok("bash")]},
    {"label": "TECH_SKILL", "pattern": [_tok("shell")]},
    {"label": "TECH_SKILL", "pattern": [_tok("powershell")]},
    {"label": "TECH_SKILL", "pattern": [_tok("sql")]},
    {"label": "TECH_SKILL", "pattern": [_tok("c")]},

    # ── Frontend Frameworks ────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("react")]},
    {"label": "TECH_SKILL", "pattern": [_tok("react.js")]},
    {"label": "TECH_SKILL", "pattern": _phrase("react", "js")},
    {"label": "TECH_SKILL", "pattern": [_tok("angular")]},
    {"label": "TECH_SKILL", "pattern": [_tok("vue.js")]},
    {"label": "TECH_SKILL", "pattern": _phrase("vue", "js")},
    {"label": "TECH_SKILL", "pattern": [_tok("vue")]},
    {"label": "TECH_SKILL", "pattern": [_tok("next.js")]},
    {"label": "TECH_SKILL", "pattern": _phrase("next", "js")},
    {"label": "TECH_SKILL", "pattern": [_tok("svelte")]},
    {"label": "TECH_SKILL", "pattern": [_tok("nuxt.js")]},
    {"label": "TECH_SKILL", "pattern": _phrase("nuxt", "js")},
    {"label": "TECH_SKILL", "pattern": [_tok("gatsby")]},
    {"label": "TECH_SKILL", "pattern": [_tok("jquery")]},
    {"label": "TECH_SKILL", "pattern": [_tok("ember.js")]},
    {"label": "TECH_SKILL", "pattern": [_tok("backbone.js")]},

    # ── Backend Frameworks ─────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("node.js")]},
    {"label": "TECH_SKILL", "pattern": _phrase("node", "js")},
    {"label": "TECH_SKILL", "pattern": [_tok("express.js")]},
    {"label": "TECH_SKILL", "pattern": _phrase("express", "js")},
    {"label": "TECH_SKILL", "pattern": [_tok("django")]},
    {"label": "TECH_SKILL", "pattern": [_tok("flask")]},
    {"label": "TECH_SKILL", "pattern": [_tok("fastapi")]},
    {"label": "TECH_SKILL", "pattern": _phrase("spring", "boot")},
    {"label": "TECH_SKILL", "pattern": [_tok("spring")]},
    {"label": "TECH_SKILL", "pattern": [_tok("nestjs")]},
    {"label": "TECH_SKILL", "pattern": [_tok("nest.js")]},
    {"label": "TECH_SKILL", "pattern": [_tok("laravel")]},
    {"label": "TECH_SKILL", "pattern": [_tok("rails")]},
    {"label": "TECH_SKILL", "pattern": _phrase("ruby", "on", "rails")},
    {"label": "TECH_SKILL", "pattern": [_tok("gin")]},
    {"label": "TECH_SKILL", "pattern": [_tok("fiber")]},
    {"label": "TECH_SKILL", "pattern": [_tok("fastify")]},
    {"label": "TECH_SKILL", "pattern": [_tok("asp.net")]},
    {"label": "TECH_SKILL", "pattern": _phrase("asp", ".net")},

    # ── Databases ──────────────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("postgresql")]},
    {"label": "TECH_SKILL", "pattern": [_tok("postgres")]},
    {"label": "TECH_SKILL", "pattern": [_tok("mysql")]},
    {"label": "TECH_SKILL", "pattern": [_tok("mongodb")]},
    {"label": "TECH_SKILL", "pattern": [_tok("redis")]},
    {"label": "TECH_SKILL", "pattern": [_tok("sqlite")]},
    {"label": "TECH_SKILL", "pattern": [_tok("cassandra")]},
    {"label": "TECH_SKILL", "pattern": [_tok("dynamodb")]},
    {"label": "TECH_SKILL", "pattern": [_tok("elasticsearch")]},
    {"label": "TECH_SKILL", "pattern": [_tok("firebase")]},
    {"label": "TECH_SKILL", "pattern": [_tok("neo4j")]},
    {"label": "TECH_SKILL", "pattern": [_tok("couchdb")]},
    {"label": "TECH_SKILL", "pattern": [_tok("mariadb")]},
    {"label": "TECH_SKILL", "pattern": [_tok("oracle")]},
    {"label": "TECH_SKILL", "pattern": [_tok("mssql")]},
    {"label": "TECH_SKILL", "pattern": _phrase("sql", "server")},
    {"label": "TECH_SKILL", "pattern": [_tok("supabase")]},
    {"label": "TECH_SKILL", "pattern": [_tok("planetscale")]},

    # ── Cloud Platforms ────────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("aws")]},
    {"label": "TECH_SKILL", "pattern": [_tok("azure")]},
    {"label": "TECH_SKILL", "pattern": [_tok("gcp")]},
    {"label": "TECH_SKILL", "pattern": _phrase("google", "cloud")},
    {"label": "TECH_SKILL", "pattern": [_tok("heroku")]},
    {"label": "TECH_SKILL", "pattern": [_tok("render")]},
    {"label": "TECH_SKILL", "pattern": [_tok("vercel")]},
    {"label": "TECH_SKILL", "pattern": [_tok("netlify")]},
    {"label": "TECH_SKILL", "pattern": [_tok("digitalocean")]},
    {"label": "TECH_SKILL", "pattern": [_tok("linode")]},
    {"label": "TECH_SKILL", "pattern": [_tok("cloudflare")]},
    {"label": "TECH_SKILL", "pattern": [_tok("lambda")]},
    {"label": "TECH_SKILL", "pattern": [_tok("ec2")]},
    {"label": "TECH_SKILL", "pattern": [_tok("s3")]},
    {"label": "TECH_SKILL", "pattern": [_tok("rds")]},

    # ── DevOps Tools ───────────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("docker")]},
    {"label": "TECH_SKILL", "pattern": [_tok("kubernetes")]},
    {"label": "TECH_SKILL", "pattern": [_tok("k8s")]},
    {"label": "TECH_SKILL", "pattern": [_tok("jenkins")]},
    {"label": "TECH_SKILL", "pattern": _phrase("github", "actions")},
    {"label": "TECH_SKILL", "pattern": _phrase("gitlab", "ci")},
    {"label": "TECH_SKILL", "pattern": _phrase("gitlab", "ci/cd")},
    {"label": "TECH_SKILL", "pattern": [_tok("terraform")]},
    {"label": "TECH_SKILL", "pattern": [_tok("ansible")]},
    {"label": "TECH_SKILL", "pattern": [_tok("prometheus")]},
    {"label": "TECH_SKILL", "pattern": [_tok("grafana")]},
    {"label": "TECH_SKILL", "pattern": [_tok("nginx")]},
    {"label": "TECH_SKILL", "pattern": [_tok("apache")]},
    {"label": "TECH_SKILL", "pattern": [_tok("git")]},
    {"label": "TECH_SKILL", "pattern": [_tok("github")]},
    {"label": "TECH_SKILL", "pattern": [_tok("gitlab")]},
    {"label": "TECH_SKILL", "pattern": [_tok("bitbucket")]},
    {"label": "TECH_SKILL", "pattern": [_tok("ci/cd")]},
    {"label": "TECH_SKILL", "pattern": [_tok("helm")]},
    {"label": "TECH_SKILL", "pattern": [_tok("vagrant")]},
    {"label": "TECH_SKILL", "pattern": [_tok("packer")]},

    # ── Web Technologies ───────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("html")]},
    {"label": "TECH_SKILL", "pattern": [_tok("css")]},
    {"label": "TECH_SKILL", "pattern": _phrase("tailwind", "css")},
    {"label": "TECH_SKILL", "pattern": [_tok("tailwind")]},
    {"label": "TECH_SKILL", "pattern": [_tok("bootstrap")]},
    {"label": "TECH_SKILL", "pattern": [_tok("rest")]},
    {"label": "TECH_SKILL", "pattern": [_tok("graphql")]},
    {"label": "TECH_SKILL", "pattern": [_tok("websocket")]},
    {"label": "TECH_SKILL", "pattern": [_tok("oauth")]},
    {"label": "TECH_SKILL", "pattern": [_tok("jwt")]},
    {"label": "TECH_SKILL", "pattern": [_tok("sass")]},
    {"label": "TECH_SKILL", "pattern": [_tok("less")]},
    {"label": "TECH_SKILL", "pattern": [_tok("webpack")]},
    {"label": "TECH_SKILL", "pattern": [_tok("vite")]},
    {"label": "TECH_SKILL", "pattern": [_tok("babel")]},
    {"label": "TECH_SKILL", "pattern": [_tok("eslint")]},
    {"label": "TECH_SKILL", "pattern": [_tok("redux")]},
    {"label": "TECH_SKILL", "pattern": [_tok("mobx")]},
    {"label": "TECH_SKILL", "pattern": [_tok("axios")]},
    {"label": "TECH_SKILL", "pattern": [_tok("fetch")]},

    # ── AI / ML Frameworks ─────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("tensorflow")]},
    {"label": "TECH_SKILL", "pattern": [_tok("pytorch")]},
    {"label": "TECH_SKILL", "pattern": [_tok("keras")]},
    {"label": "TECH_SKILL", "pattern": [_tok("scikit-learn")]},
    {"label": "TECH_SKILL", "pattern": _phrase("scikit", "learn")},
    {"label": "TECH_SKILL", "pattern": _phrase("hugging", "face")},
    {"label": "TECH_SKILL", "pattern": [_tok("huggingface")]},
    {"label": "TECH_SKILL", "pattern": [_tok("langchain")]},
    {"label": "TECH_SKILL", "pattern": [_tok("opencv")]},
    {"label": "TECH_SKILL", "pattern": [_tok("nltk")]},
    {"label": "TECH_SKILL", "pattern": [_tok("spacy")]},
    {"label": "TECH_SKILL", "pattern": [_tok("transformers")]},
    {"label": "TECH_SKILL", "pattern": [_tok("diffusers")]},
    {"label": "TECH_SKILL", "pattern": [_tok("llamaindex")]},

    # ── ML / Data Libraries ────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("pandas")]},
    {"label": "TECH_SKILL", "pattern": [_tok("numpy")]},
    {"label": "TECH_SKILL", "pattern": [_tok("matplotlib")]},
    {"label": "TECH_SKILL", "pattern": [_tok("seaborn")]},
    {"label": "TECH_SKILL", "pattern": [_tok("xgboost")]},
    {"label": "TECH_SKILL", "pattern": [_tok("lightgbm")]},
    {"label": "TECH_SKILL", "pattern": [_tok("faiss")]},
    {"label": "TECH_SKILL", "pattern": [_tok("plotly")]},
    {"label": "TECH_SKILL", "pattern": [_tok("scipy")]},
    {"label": "TECH_SKILL", "pattern": [_tok("statsmodels")]},

    # ── Testing ────────────────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("pytest")]},
    {"label": "TECH_SKILL", "pattern": [_tok("jest")]},
    {"label": "TECH_SKILL", "pattern": [_tok("mocha")]},
    {"label": "TECH_SKILL", "pattern": [_tok("selenium")]},
    {"label": "TECH_SKILL", "pattern": [_tok("cypress")]},
    {"label": "TECH_SKILL", "pattern": [_tok("playwright")]},
    {"label": "TECH_SKILL", "pattern": [_tok("postman")]},

    # ── Mobile ─────────────────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("react native")]},
    {"label": "TECH_SKILL", "pattern": _phrase("react", "native")},
    {"label": "TECH_SKILL", "pattern": [_tok("flutter")]},
    {"label": "TECH_SKILL", "pattern": [_tok("ionic")]},
    {"label": "TECH_SKILL", "pattern": [_tok("xamarin")]},

    # ── Other Tools ────────────────────────────────────────────────────── #
    {"label": "TECH_SKILL", "pattern": [_tok("linux")]},
    {"label": "TECH_SKILL", "pattern": [_tok("unix")]},
    {"label": "TECH_SKILL", "pattern": [_tok("macos")]},
    {"label": "TECH_SKILL", "pattern": [_tok("windows")]},
    {"label": "TECH_SKILL", "pattern": [_tok("arduino")]},
    {"label": "TECH_SKILL", "pattern": [_tok("raspberry")]},
    {"label": "TECH_SKILL", "pattern": [_tok("mqtt")]},
    {"label": "TECH_SKILL", "pattern": [_tok("kafka")]},
    {"label": "TECH_SKILL", "pattern": [_tok("rabbitmq")]},
    {"label": "TECH_SKILL", "pattern": [_tok("celery")]},
    {"label": "TECH_SKILL", "pattern": [_tok("airflow")]},
    {"label": "TECH_SKILL", "pattern": [_tok("spark")]},
    {"label": "TECH_SKILL", "pattern": [_tok("hadoop")]},
    {"label": "TECH_SKILL", "pattern": [_tok("tableau")]},
    {"label": "TECH_SKILL", "pattern": [_tok("power bi")]},
    {"label": "TECH_SKILL", "pattern": _phrase("power", "bi")},
    {"label": "TECH_SKILL", "pattern": [_tok("excel")]},
    {"label": "TECH_SKILL", "pattern": [_tok("figma")]},
    {"label": "TECH_SKILL", "pattern": [_tok("jira")]},
    {"label": "TECH_SKILL", "pattern": [_tok("confluence")]},
    {"label": "TECH_SKILL", "pattern": [_tok("notion")]},
    {"label": "TECH_SKILL", "pattern": [_tok("rfid")]},
    {"label": "TECH_SKILL", "pattern": [_tok("iot")]},
]


def get_patterns() -> list[dict]:
    """Return all EntityRuler patterns for the TECH_SKILL label."""
    return list(_PATTERNS)
