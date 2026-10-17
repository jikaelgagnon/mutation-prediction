from pathlib import Path
import tarfile
from zenodo_get import download

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MEGASCALE_DIR = DATA_DIR / "megascale"
ROSETTA_DIR = DATA_DIR / "megascale-rosetta"


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


def get_rosetta():
    complete = ROSETTA_DIR / ".download-complete"
    if complete.exists():
        return ROSETTA_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "Megascale-Rosetta-Sweeps.tar.gz"
    # downloads: https://zenodo.org/records/13345274/files/Megascale-Rosetta-Sweeps.tar.gz?download=1
    download(
        record_or_doi="13345274",
        output_dir=DATA_DIR,
        file_glob=archive.name,
        verbosity=3,  # Show byte-level progress within each file.
    )

    ROSETTA_DIR.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        # Remove a shared enclosing folder, if present, so the protein folders
        # live directly under ROSETTA_DIR regardless of the archive's label.
        roots = {Path(member.name).parts[0] for member in members if Path(member.name).parts}
        prefix = next(iter(roots)) + "/" if len(roots) == 1 else ""
        for member in members:
            if prefix:
                if member.name.rstrip("/") == prefix.rstrip("/"):
                    continue
                member.name = member.name.removeprefix(prefix)
            tar.extract(member, ROSETTA_DIR, filter="data")

    complete.touch()
    archive.unlink()
    return ROSETTA_DIR


if __name__ == "__main__":
    megascale_dir = get_megascale()
    rosetta_dir = get_rosetta()
    print(f"Megascale data: {megascale_dir}")
    print(f"Rosetta data: {rosetta_dir}")
