import hashlib
import zipfile

import pytest

from download_rdd2022 import human_size, md5sum, safe_extract


def test_human_size_boundaries():
    assert human_size(0) == "0.00 B"
    assert human_size(1024) == "1.00 KiB"
    assert human_size(1024**3) == "1.00 GiB"


def test_md5sum(tmp_path):
    payload = b"rdd2022\n"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    assert md5sum(path) == hashlib.md5(payload, usedforsecurity=False).hexdigest()


def test_safe_extract_accepts_relative_members(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("RDD2022/domain.zip", b"content")
    destination = tmp_path / "destination"
    safe_extract(archive, destination)
    assert (destination / "RDD2022" / "domain.zip").read_bytes() == b"content"


def test_safe_extract_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape.txt", b"no")
    with pytest.raises(RuntimeError, match="Caminho inseguro"):
        safe_extract(archive, tmp_path / "destination")
