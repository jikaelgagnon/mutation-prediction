import argparse
import tarfile
from pathlib import Path

from zenodo_get import download

DATA_DIR = Path.cwd() / "data" / "raw"
MEGASCALE_DIR = DATA_DIR / "megascale"


def get_megascale():
    if MEGASCALE_DIR.exists() and any(MEGASCALE_DIR.iterdir()):
        return MEGASCALE_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    archive = DATA_DIR / "Megascale.tar.gz"

    # downloads: https://zenodo.org/records/13345274/files/Megascale.tar.gz?download=1
    download(
        record_or_doi="13345274",
        output_dir=DATA_DIR,
        file_glob="Megascale.tar.gz",
        verbosity=3,  # Show byte-level progress within each file.
    )

    MEGASCALE_DIR.mkdir(exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.startswith("Megascale/"):
                member.name = member.name.removeprefix("Megascale/")
                tar.extract(member, MEGASCALE_DIR)

    archive.unlink()

    return MEGASCALE_DIR


def main():
    """Download MegaScale into data/raw under the working directory."""
    argparse.ArgumentParser(description=main.__doc__).parse_args()
    print(f"MegaScale data: {get_megascale()}")


if __name__ == '__main__':
    main()
