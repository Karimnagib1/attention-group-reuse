#!/usr/bin/env bash
set -euo pipefail

# Activate the thesis environment before invoking this wrapper.
REVIEW_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REVIEW_REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
exec python -m review.latent_country.run "$@"
