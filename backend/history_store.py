from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class HistoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mongo = None
        self.backend = "json"
        uri = os.getenv("ECHOSAFE_MONGODB_URI")
        if uri:
            try:
                from pymongo import MongoClient

                self._mongo = MongoClient(uri, serverSelectionTimeoutMS=1500)
                self._mongo.admin.command("ping")
                self.collection = self._mongo[os.getenv("ECHOSAFE_MONGO_DB", "echosafe")]["history"]
                self.backend = "mongodb"
            except Exception:
                self._mongo = None

        if self._mongo is None:
            self.path = Path(os.getenv("ECHOSAFE_HISTORY_FILE", "backend/data/history.json"))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.write_text("[]", encoding="utf-8")

    @property
    def status(self) -> str:
        return self.backend

    def list(self) -> List[Dict[str, Any]]:
        if self._mongo is not None:
            rows = list(self.collection.find({}, {"_id": 0}).sort("created_at", -1))
            for row in rows:
                if "_id" in row:
                    row.pop("_id")
                row["id"] = str(row.get("id", ""))
            return rows
        with self._lock:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            return sorted(rows, key=lambda row: row.get("created_at", ""), reverse=True)

    def add(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(entry)
        row["id"] = str(uuid.uuid4())
        row["created_at"] = datetime.now(timezone.utc).isoformat()
        if self._mongo is not None:
            self.collection.insert_one(dict(row))
            row.pop("_id", None)
        else:
            with self._lock:
                rows = json.loads(self.path.read_text(encoding="utf-8"))
                rows.insert(0, row)
                self.path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return row

    def delete(self, entry_id: str) -> bool:
        if self._mongo is not None:
            result = self.collection.delete_one({"id": entry_id})
            return result.deleted_count > 0
        with self._lock:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            kept = [row for row in rows if str(row.get("id")) != entry_id]
            if len(kept) == len(rows):
                return False
            self.path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
            return True

    def clear(self) -> int:
        if self._mongo is not None:
            return int(self.collection.delete_many({}).deleted_count)
        with self._lock:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            count = len(rows)
            self.path.write_text("[]", encoding="utf-8")
            return count
