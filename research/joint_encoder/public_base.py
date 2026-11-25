"""Fetch only the five pinned public multilingual DistilBERT files."""

import hashlib
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from equip_triage.joint.model import PUBLIC_ID, PUBLIC_REV

FILES = {
    "config.json": "67ab80c9a476e7c2d189cd03f95d297b04bd1d90e99765323af155ab6308960e",
    "tokenizer_config.json": "0f6d13e6f4da6f9e24f22ada6bc3be571123d858d7c0c05a8a7cd55a9c23c2e8",
    "tokenizer.json": "f4a4d5bf7301717e261fafbe26e1eb967f6ba4cb3ae0ab7a29f4642ec229f386",
    "vocab.txt": "fe0fda7c425b48c516fc8f160d594c8022a0808447475c1a7c6d6479763f310c",
    "model.safetensors": "fcf002be901b9ad708e0df430b8d18deaa4b3fe3519e24e9017a5ee17ca2c228",
}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assets(root, download=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = root / name
        if path.is_file() and file_sha(path) == expected:
            continue
        if not download:
            raise ValueError(
                f"Missing or changed pinned public file: {name}. Run fetch-public."
            )
        url = f"https://huggingface.co/{PUBLIC_ID}/resolve/{PUBLIC_REV}/{name}?download=true"
        temp = None
        try:
            with NamedTemporaryFile(dir=root, delete=False) as out:
                temp = Path(out.name)
                with urlopen(
                    Request(url, headers={"User-Agent": "EquipTriage/1"}), timeout=120
                ) as incoming:
                    shutil.copyfileobj(incoming, out)
            if file_sha(temp) != expected:
                raise ValueError(f"Public file checksum mismatch: {name}")
            temp.replace(path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    return root
