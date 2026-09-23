"""Read-only integrity check of this edition's wheel and packaged Python source."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import sys
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parent
    wheel = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "metabaw-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        broken = archive.testzip()
        if broken:
            raise ValueError(f"ZIP CRC failed: {broken}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate wheel member paths")
        record = "metabaw-0.1.0.dist-info/RECORD"
        rows = list(csv.reader(io.StringIO(archive.read(record).decode("utf-8"))))
        if {row[0] for row in rows} != set(names) or len(rows) != len(names):
            raise ValueError("RECORD membership does not match ZIP contents")
        for name, encoded_hash, size in rows:
            if name == record:
                if encoded_hash or size:
                    raise ValueError("RECORD must not hash itself")
                continue
            content = archive.read(name)
            actual = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
            if encoded_hash != f"sha256={actual}" or int(size) != len(content):
                raise ValueError(f"RECORD integrity mismatch: {name}")
        sources = list((root / "src" / "metabaw").rglob("*.py"))
        for path in sources:
            member = path.relative_to(root / "src").as_posix()
            if archive.read(member) != path.read_bytes():
                raise ValueError(f"Packaged source differs from source tree: {member}")
        metadata = archive.read("metabaw-0.1.0.dist-info/METADATA").decode("utf-8")
        for required in ("Version: 0.1.0", "named-input edition", "Requires-Python: <3.12,>=3.11"):
            if required not in metadata:
                raise ValueError(f"Missing metadata: {required}")
    print(f"OK: ZIP CRC, all {len(rows)} RECORD entries, {len(sources)} Python sources, metadata")
    print(f"Bytes: {wheel.stat().st_size}")
    print(f"SHA256: {hashlib.sha256(wheel.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
