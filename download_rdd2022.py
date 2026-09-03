#!/usr/bin/env python3
"""Baixa e valida a versao 1 do RDD2022 a partir do Figshare.

O script usa apenas a biblioteca padrao do Python, retoma downloads interrompidos
e compara o MD5 de cada arquivo com o checksum publicado pelo Figshare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ARTICLE_ID = 21_431_547
VERSION = 1
API_URL = (
    f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{VERSION}"
)
CHUNK_SIZE = 8 * 1024 * 1024
USER_AGENT = "RDD2022-downloader/1.0"


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def request(url: str, *, start: int | None = None) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.Request(url, headers=headers)


def fetch_metadata() -> dict:
    with urllib.request.urlopen(request(API_URL), timeout=60) as response:
        return json.load(response)


def md5sum(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def show_progress(
    name: str, done: int, total: int, started_at: float, initial: int = 0
) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = (done - initial) / elapsed
    percent = 100 * done / total if total else 0
    remaining = (total - done) / rate if rate else 0
    print(
        f"\r{name}: {percent:6.2f}%  {human_size(done)}/{human_size(total)}  "
        f"{human_size(int(rate))}/s  ETA {remaining / 60:.1f} min",
        end="",
        flush=True,
    )


def download_file(file_info: dict, output_dir: Path, retries: int = 5) -> Path:
    name = Path(file_info["name"]).name
    destination = output_dir / name
    partial = destination.with_name(destination.name + ".part")
    expected_size = int(file_info["size"])
    expected_md5 = file_info.get("computed_md5") or file_info.get("supplied_md5")

    if destination.exists():
        if destination.stat().st_size == expected_size and (
            not expected_md5 or md5sum(destination) == expected_md5
        ):
            print(f"OK (ja existe): {destination}")
            return destination
        raise RuntimeError(
            f"{destination} ja existe, mas seu tamanho ou MD5 nao confere. "
            "Remova ou renomeie o arquivo antes de tentar novamente."
        )

    for attempt in range(1, retries + 1):
        start = partial.stat().st_size if partial.exists() else 0
        try:
            with urllib.request.urlopen(
                request(file_info["download_url"], start=start), timeout=120
            ) as response:
                resumed = start > 0 and response.status == 206
                if start > 0 and not resumed:
                    print(f"\nO servidor nao aceitou retomada para {name}; reiniciando.")
                    start = 0
                mode = "ab" if resumed else "wb"
                started_at = time.monotonic()
                downloaded = start
                with partial.open(mode) as stream:
                    while chunk := response.read(CHUNK_SIZE):
                        stream.write(chunk)
                        downloaded += len(chunk)
                        show_progress(
                            name, downloaded, expected_size, started_at, initial=start
                        )
            print()

            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"tamanho invalido para {name}: {actual_size} != {expected_size}"
                )
            if expected_md5:
                print(f"Validando MD5 de {name}...")
                actual_md5 = md5sum(partial)
                if actual_md5 != expected_md5:
                    raise RuntimeError(
                        f"MD5 invalido para {name}: {actual_md5} != {expected_md5}"
                    )
            partial.replace(destination)
            print(f"OK: {destination}")
            return destination
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            if attempt == retries:
                raise
            delay = 2**attempt
            print(f"\nFalha ({attempt}/{retries}): {error}. Nova tentativa em {delay}s.")
            time.sleep(delay)

    raise AssertionError("unreachable")


def safe_extract(archive: Path, destination: Path) -> None:
    """Extrai ZIP sem permitir caminhos absolutos ou `..`."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Caminho inseguro no ZIP: {member.filename}")
            target = (destination / Path(*member_path.parts)).resolve()
            if not target.is_relative_to(destination_root):
                raise RuntimeError(f"Caminho inseguro no ZIP: {member.filename}")
        print(f"Extraindo {archive.name} em {destination}...")
        zipped.extractall(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rdd2022/raw"),
        help="diretorio dos arquivos originais (padrao: data/rdd2022/raw)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="extrai o ZIP apos download e validacao",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Consultando metadados: {API_URL}")
    metadata = fetch_metadata()
    metadata_path = args.output / "figshare_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    total = sum(int(item["size"]) for item in metadata["files"])
    free = shutil.disk_usage(args.output).free
    print(f"Dataset: {metadata['title']}")
    print(f"Arquivos: {len(metadata['files'])}; total: {human_size(total)}")
    print(f"Espaco livre: {human_size(free)}")
    if free < total:
        raise RuntimeError("espaco livre insuficiente para o download")

    downloaded = [download_file(item, args.output) for item in metadata["files"]]
    if args.extract:
        archives = [path for path in downloaded if path.suffix.lower() == ".zip"]
        for archive in archives:
            safe_extract(archive, args.output.parent / "extracted")

    print("Download e validacao concluidos.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompido; execute novamente para retomar o download.", file=sys.stderr)
        raise SystemExit(130)
