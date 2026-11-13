#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
exec uv run --locked python inference.py
