from pathlib import Path
import tarfile
from zenodo_get import download

DATA_DIR = Path("../data/raw")
MEGASCALE_DIR = DATA_DIR / "megascale"


def get_megascale():
    if MEGASCALE_DIR.exists() and any(MEGASCALE_DIR.iterdir()):
        return MEGASCALE_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    archive = DATA_DIR / "Megascale.tar.gz"

    download(
        record_or_doi="13345274",
        output_dir=DATA_DIR,
        file_glob="Megascale.tar.gz",
    )

    MEGASCALE_DIR.mkdir(exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.startswith("Megascale/"):
                member.name = member.name.removeprefix("Megascale/")
                tar.extract(member, MEGASCALE_DIR)

    archive.unlink()

    return MEGASCALE_DIR


megascale_dir = get_megascale()