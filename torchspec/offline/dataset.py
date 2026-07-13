"""Small on-disk dataset used by offline replay."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

OFFLINE_SCHEMA_VERSION = 1
_TENSOR_NAMES = ("input_ids", "hidden_states", "target", "last_hidden_states")


class OfflineDataset:
    """Read and append replay records.

    The format intentionally has only three parts: ``dataset.json``, one
    ``manifest.jsonl``, and self-describing ``samples/*.pt`` files.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        create: bool = False,
        last_hidden_states_prenorm: bool | None = None,
        overwrite: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if overwrite and self.root.exists():
            shutil.rmtree(self.root)

        metadata_path = self.root / "dataset.json"
        if create and not metadata_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        "version": OFFLINE_SCHEMA_VERSION,
                        "last_hidden_states_prenorm": last_hidden_states_prenorm,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Offline dataset not found: {metadata_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("version") != OFFLINE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported offline dataset version {self.metadata.get('version')!r}"
            )
        if (
            create
            and last_hidden_states_prenorm is not None
            and self.metadata.get("last_hidden_states_prenorm") != last_hidden_states_prenorm
        ):
            raise ValueError("Offline dataset uses a different hidden-state representation")

        self._rows: dict[str, list[dict[str, str]]] = {"train": [], "eval": []}
        self._by_id: dict[str, dict[str, str]] = {}
        manifest = self.root / "manifest.jsonl"
        if manifest.exists():
            with manifest.open(encoding="utf-8") as stream:
                for lineno, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    split = row.get("split")
                    data_id = str(row.get("data_id"))
                    if split not in self._rows:
                        raise ValueError(f"Invalid split in {manifest}:{lineno}")
                    if data_id in self._by_id:
                        raise ValueError(f"Duplicate data_id {data_id!r} in {manifest}")
                    path = (self.root / row["file"]).resolve()
                    if self.root not in path.parents or not path.is_file():
                        raise FileNotFoundError(f"Offline sample not found: {path}")
                    item = {"split": split, "data_id": data_id, "file": row["file"]}
                    self._rows[split].append(item)
                    self._by_id[data_id] = item

    def rows(self, split: str) -> list[dict[str, str]]:
        return list(self._rows[split])

    def ids(self, split: str) -> set[str]:
        return {row["data_id"] for row in self._rows[split]}

    def count(self, split: str) -> int:
        return len(self._rows[split])

    def load(self, row_or_id: dict[str, str] | str) -> dict[str, Any]:
        row = self._by_id[str(row_or_id)] if isinstance(row_or_id, str) else row_or_id
        path = self.root / row["file"]
        record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(record, dict):
            raise ValueError(f"Offline sample must contain a dict: {path}")
        if str(record.get("data_id")) != row["data_id"]:
            raise ValueError(f"Offline sample data_id does not match manifest: {path}")
        if not all(
            isinstance(record.get(name), torch.Tensor) for name in ("input_ids", "hidden_states")
        ):
            raise ValueError(f"Offline sample has missing tensors: {path}")
        return record

    def append(
        self,
        split: str,
        *,
        data_id: str,
        tensors: dict[str, torch.Tensor | None],
        packed_loss_mask: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if split not in self._rows:
            raise ValueError(f"Unknown offline split: {split!r}")
        data_id = str(data_id)
        if data_id in self._by_id:
            if self._by_id[data_id]["split"] != split:
                raise ValueError(f"data_id {data_id!r} already exists in another split")
            return False

        saved = {
            name: value.detach().cpu().contiguous()
            for name, value in tensors.items()
            if name in _TENSOR_NAMES and isinstance(value, torch.Tensor)
        }
        if not {"input_ids", "hidden_states"}.issubset(saved):
            raise ValueError(f"Offline sample {data_id!r} has missing tensors")

        relative = Path("samples") / f"{hashlib.sha1(data_id.encode()).hexdigest()}.pt"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            **saved,
            "data_id": data_id,
            "packed_loss_mask": packed_loss_mask,
            "metadata": dict(metadata or {}),
        }
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        os.close(fd)
        try:
            torch.save(record, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        row = {"split": split, "data_id": data_id, "file": relative.as_posix()}
        with (self.root / "manifest.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._rows[split].append(row)
        self._by_id[data_id] = row
        return True


def configure_offline_args(dataset: OfflineDataset, args) -> None:
    """Use the representation detail recorded with the tensors."""
    value = dataset.metadata.get("last_hidden_states_prenorm")
    if value is None:
        raise ValueError("Offline dataset does not declare last_hidden_states_prenorm")
    args.last_hidden_states_prenorm = bool(value)
