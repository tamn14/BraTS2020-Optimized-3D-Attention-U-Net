import zipfile
from pathlib import Path
"""Utility function to extract a zip file to a specified directory."""
def extract_zip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)