"""Retrieval of the project's Hugging Face image datasets."""

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

WATERHEMP_GREENHOUSE = "punlalit19/Waterhemp_GreenHouse_Images"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def fetch_dataset(
    repo_id: str = WATERHEMP_GREENHOUSE,
    data_dir: Path | str = DATA_DIR,
    revision: str | None = None,
) -> Path:
    """Download a dataset repo into `<data_dir>/<dataset name>/` and return that path."""
    local_dir = Path(data_dir) / repo_id.split("/")[-1]
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )
    return local_dir


def dataset_revision(repo_id: str = WATERHEMP_GREENHOUSE) -> str:
    """Commit sha of the dataset on the Hub."""
    return HfApi().dataset_info(repo_id).sha
