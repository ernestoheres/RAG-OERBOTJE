"""Small DiskCache wrapper for notebook results and readable JSON exports."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from diskcache import Cache


class ExperimentCache:
    """Persistent JSON values, keyed by content and experiment settings."""

    schema_version = 1

    def __init__(self, root):
        self.root = Path(root)

    def key(self, namespace, identity):
        serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"v{self.schema_version}:{namespace}:{digest}"

    def _open(self):
        return Cache(
            str(self.root / "artifacts"), timeout=30,
            eviction_policy="none", sqlite_synchronous=2,
        )

    def get(self, namespace, identity):
        with self._open() as cache:
            value = cache.get(self.key(namespace, identity))
        return json.loads(value) if value is not None else None

    def put(self, namespace, identity, value):
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
        with self._open() as cache:
            cache.set(self.key(namespace, identity), serialized, tag=namespace, retry=True)

    def get_or_compute(self, namespace, identity, compute):
        value = self.get(namespace, identity)
        if value is not None:
            return value
        # Exceptions are deliberately not cached; the next run can retry.
        value = compute()
        self.put(namespace, identity, value)
        return value

    def save_checkpoint(self, namespace, identity, value, export_path):
        # Commit the authoritative checkpoint before updating its readable mirror.
        self.put(namespace, identity, value)
        self.export_json(export_path, value)

    @staticmethod
    def export_json(path, value):
        """Atomically replace an export; interruption cannot leave half a JSON file."""
        path = Path(path)
        serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
