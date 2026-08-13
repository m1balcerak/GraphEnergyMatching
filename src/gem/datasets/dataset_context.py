"""Dataset construction shared by GEM training and diagnostics."""

from __future__ import annotations

from omegaconf import DictConfig


def build_dataset_context(cfg: DictConfig):
    """Return datamodule, dataset metadata, and optional molecular SMILES."""
    dataset_name = str(getattr(cfg.dataset, "name", "")).lower()
    if not dataset_name:
        raise ValueError("cfg.dataset.name must be set for GEM training.")

    if dataset_name == "moses":
        from gem.datasets import moses_dataset

        datamodule = moses_dataset.MosesDataModule(cfg)
        dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
        dataset_smiles = moses_dataset.get_smiles(
            raw_dir=datamodule.train_dataset.raw_dir,
            filter_dataset=getattr(cfg.dataset, "filter", False),
        )
        return datamodule, dataset_infos, dataset_smiles

    raise ValueError(f"Dataset {dataset_name!r} is not supported; expected 'moses'.")
