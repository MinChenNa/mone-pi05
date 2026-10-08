"""Repair corrupt files in a public GCS checkpoint using CRC32C verification."""

import argparse
import base64
import pathlib
import urllib.parse

import gcsfs
import google_crc32c
import requests


def crc32c(path: pathlib.Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    checksum = google_crc32c.Checksum()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("remote", help="GCS path without gs://")
    parser.add_argument("local", type=pathlib.Path)
    args = parser.parse_args()

    fs = gcsfs.GCSFileSystem(token="anon")
    objects = fs.find(args.remote, detail=True)
    repaired = 0
    for remote_path, metadata in objects.items():
        relative = pathlib.PurePosixPath(remote_path).relative_to(args.remote)
        local_path = args.local / relative
        expected_crc = metadata["crc32c"]
        if local_path.exists() and local_path.stat().st_size == metadata["size"] and crc32c(local_path) == expected_crc:
            print(f"OK {relative}", flush=True)
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = local_path.with_name(f"{local_path.name}.part")
        url = "https://storage.googleapis.com/" + urllib.parse.quote(remote_path, safe="/")
        print(f"DOWNLOAD {relative} ({metadata['size']} bytes)", flush=True)
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            with part_path.open("wb") as output:
                while chunk := response.raw.read(8 * 1024 * 1024):
                    output.write(chunk)
        actual_crc = crc32c(part_path)
        if part_path.stat().st_size != metadata["size"] or actual_crc != expected_crc:
            raise RuntimeError(
                f"verification failed for {relative}: "
                f"size={part_path.stat().st_size}/{metadata['size']} crc={actual_crc}/{expected_crc}"
            )
        part_path.replace(local_path)
        repaired += 1
        print(f"REPAIRED {relative}", flush=True)

    print(f"complete: repaired={repaired}, files={len(objects)}", flush=True)


if __name__ == "__main__":
    main()
