import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import torch

logger = logging.getLogger(__name__)


def _detect_workspace_root() -> str | None:
    """Search upwards from CWD for common repo markers."""
    markers = ("pyproject.toml", "setup.cfg", ".git")
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        if any((directory / marker).exists() for marker in markers):
            return str(directory)
    return None


_WORKSPACE_ROOT = _detect_workspace_root()


class Dataset(torch.utils.data.Dataset):
    """A class representing a dataset."""

    def __init__(self, data: list[dict[str, Any]], name: str | None = None, split: str | None = None):
        """Initialize a Dataset.

        Args:
            data: List of dictionaries containing the dataset examples
            name: Optional name for the dataset
            split: Optional split name (e.g., 'train', 'test')
        """
        super().__init__()
        self.data = data
        self.name = name
        self.split = split

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get an item by index."""
        return self.data[idx]

    def get_data(self) -> list[dict[str, Any]]:
        """Get the dataset data."""
        return self.data

    def repeat(self, n: int) -> "Dataset":
        """Repeat the dataset n times, keeping repeated entries adjacent.

        Args:
            n: Number of times to repeat the dataset

        Returns:
            Dataset: A new dataset with repeated entries
        """
        if n <= 0:
            raise ValueError("Repeat count must be positive")

        # Create repeated data with adjacent copies
        repeated_data = []
        for item in self.data:
            # Add n copies of this item consecutively
            repeated_data.extend([item.copy() for _ in range(n)])

        return Dataset(data=repeated_data, name=self.name, split=self.split)

    def get_data_path(self) -> str | None:
        """Get the absolute path of the dataset file.

        Returns:
            Optional[str]: The absolute path of the dataset file, or None if the dataset is not registered
        """
        if self.name is None or self.split is None:
            return None

        registry = DatasetRegistry._load_registry()
        if self.name not in registry or self.split not in registry[self.name]:
            return None

        stored_path = registry[self.name][self.split]
        return DatasetRegistry._resolve_dataset_path(stored_path)

    def get_verl_data_path(self) -> str | None:
        """Get the absolute path of the Verl-processed dataset file.

        Returns:
            Optional[str]: The absolute path of the Verl-processed dataset file, or None if not found
        """
        data_path = self.get_data_path()
        if data_path is None:
            return None

        verl_path = data_path.replace(".parquet", "_verl.parquet")
        return verl_path if os.path.exists(verl_path) else None

    @classmethod
    def load_data(cls, path: str) -> "Dataset":
        """Load dataset directly from a file path.

        Args:
            path: Path to the dataset file

        Returns:
            Dataset: The loaded dataset

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}")

        file_ext = os.path.splitext(path)[1].lower()

        if file_ext == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        elif file_ext == ".jsonl":
            data = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        elif file_ext == ".csv":
            data = pd.read_csv(path).to_dict("records")
        elif file_ext == ".parquet":
            data = pd.read_parquet(path).to_dict("records")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return cls(data=data)


class DatasetRegistry:
    """A registry for datasets that manages storage and retrieval."""

    _DATA_HOME_OVERRIDE = os.environ.get("RLLM_DATA_HOME")
    if _DATA_HOME_OVERRIDE:
        _DATA_HOME_OVERRIDE = os.path.abspath(os.path.expanduser(_DATA_HOME_OVERRIDE))

    _WORKSPACE_DATA_HOME = os.path.join(_WORKSPACE_ROOT, "rllm_data") if _WORKSPACE_ROOT else None

    _BASE_DATA_HOME = _DATA_HOME_OVERRIDE or _WORKSPACE_DATA_HOME

    if _BASE_DATA_HOME:
        _REGISTRY_DIR = os.path.join(_BASE_DATA_HOME, "registry")
        _DATASET_DIR = os.path.join(_BASE_DATA_HOME, "datasets")
    else:
        _REGISTRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "registry")
        _DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")

    _REGISTRY_FILE = os.path.join(_REGISTRY_DIR, "dataset_registry.json")

    @classmethod
    def _path_relative_to_base(cls, abs_path: str) -> str:
        """Return a path relative to the data home if possible."""
        abs_path = os.path.abspath(abs_path)
        if not cls._BASE_DATA_HOME:
            return abs_path

        base_path = os.path.abspath(cls._BASE_DATA_HOME)
        try:
            if os.path.commonpath([abs_path, base_path]) == base_path:
                rel_path = os.path.relpath(abs_path, base_path)
                return rel_path.replace("\\", "/")
        except ValueError:
            # Paths on different drives (Windows)
            pass

        return abs_path

    @classmethod
    def _rewrite_with_current_base(cls, abs_path: str) -> str | None:
        """Replace the portion up to 'rllm_data' with the current base directory."""
        if not cls._BASE_DATA_HOME:
            return None

        parts = Path(abs_path).parts
        try:
            idx = next(i for i, part in enumerate(parts) if part == "rllm_data")
        except StopIteration:
            return None

        suffix_parts = parts[idx + 1 :]
        candidate = Path(cls._BASE_DATA_HOME).joinpath(*suffix_parts)
        return str(candidate)

    @classmethod
    def _resolve_dataset_path(cls, stored_path: str) -> str:
        """Convert a registry path into an absolute path."""
        expanded = os.path.expanduser(stored_path)
        if os.path.isabs(expanded):
            if os.path.exists(expanded):
                return expanded
            rewritten = cls._rewrite_with_current_base(expanded)
            return rewritten or expanded

        if cls._BASE_DATA_HOME:
            return str(Path(cls._BASE_DATA_HOME).joinpath(Path(expanded)))

        return os.path.abspath(expanded)

    @classmethod
    def _ensure_directories(cls) -> None:
        """Ensure the registry and dataset directories exist."""
        os.makedirs(cls._REGISTRY_DIR, exist_ok=True)
        os.makedirs(cls._DATASET_DIR, exist_ok=True)

    @classmethod
    def _load_registry(cls) -> dict[str, dict[str, str]]:
        """Load the dataset registry from the registry file."""
        cls._ensure_directories()
        if not os.path.exists(cls._REGISTRY_FILE):
            return {}

        try:
            with open(cls._REGISTRY_FILE, encoding="utf-8") as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON format in registry file. Creating a new registry.")
            return {}
        return registry

    @classmethod
    def _save_registry(cls, registry: dict[str, dict[str, str]]) -> None:
        """Save the dataset registry to the registry file."""
        cls._ensure_directories()
        with open(cls._REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    @classmethod
    def register_dataset(cls, name: str, data: list[dict[str, Any]] | Any, split: str = "default") -> Dataset:
        """Register a dataset by saving it to disk and updating the registry.

        Args:
            name: Name of the dataset
            data: List of dictionaries containing the dataset examples or a Hugging Face dataset
            split: Split name (e.g., 'train', 'test', 'default')

        Returns:
            Dataset: The registered dataset
        """
        cls._ensure_directories()

        # Create dataset directory if it doesn't exist
        dataset_dir = os.path.join(cls._DATASET_DIR, name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Convert HuggingFace dataset to list of dictionaries if needed
        if hasattr(data, "to_pandas") and callable(data.to_pandas):
            # This is likely a HuggingFace dataset
            data_df = data.to_pandas()
            data_list = data_df.to_dict("records")
        else:
            # Assume it's already a list of dictionaries
            data_list = data
            data_df = pd.DataFrame(data_list)

        # Save original data
        dataset_path = os.path.join(dataset_dir, f"{split}.parquet")
        data_df.to_parquet(dataset_path)

        # Apply Verl postprocessing and save
        verl_data = cls.apply_verl_postprocessing(data_list)
        verl_dataset_path = os.path.join(dataset_dir, f"{split}_verl.parquet")
        verl_data_df = pd.DataFrame(verl_data)
        verl_data_df.to_parquet(verl_dataset_path)

        # Update registry
        registry = cls._load_registry()

        # Initialize dataset entry if it doesn't exist
        if name not in registry:
            registry[name] = {}

        # Add the split to the dataset
        registry[name][split] = cls._path_relative_to_base(dataset_path)
        cls._save_registry(registry)

        logger.info(f"Registered dataset '{name}' split '{split}' with {len(data_list)} examples. Verl-processed version saved at {verl_dataset_path}.")

        return Dataset(data=data_list, name=name, split=split)

    @classmethod
    def load_dataset(cls, name: str, split: str = "default") -> Dataset | None:
        """Load a dataset from the registry.

        Args:
            name: Name of the dataset to load
            split: Split name to load (e.g., 'train', 'test', 'default')

        Returns:
            Dataset: The loaded dataset or None if not found
        """
        registry = cls._load_registry()
        if name not in registry:
            logger.warning(f"Dataset '{name}' not found in registry.")
            return None

        dataset_info = registry[name]

        if split not in dataset_info:
            logger.warning(f"Split '{split}' not found in dataset '{name}'.")
            return None

        # Load data
        dataset_path = cls._resolve_dataset_path(dataset_info[split])
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found: {dataset_path}")
            return None

        data = pl.read_parquet(dataset_path).to_dicts()

        logger.info(f"Loaded dataset '{name}' split '{split}' with {len(data)} examples.")

        return Dataset(data=data, name=name, split=split)

    @classmethod
    def get_dataset_names(cls) -> list[str]:
        """Get the names of all registered datasets.

        Returns:
            List[str]: List of dataset names
        """
        return list(cls._load_registry().keys())

    @classmethod
    def get_dataset_splits(cls, name: str) -> list[str]:
        """Get the available splits for a dataset.

        Args:
            name: Name of the dataset

        Returns:
            List[str]: List of available splits
        """
        registry = cls._load_registry()
        if name not in registry:
            return []
        return list(registry[name].keys())

    @classmethod
    def dataset_exists(cls, name: str, split: str | None = None) -> bool:
        """Check if a dataset exists in the registry.

        Args:
            name: Name of the dataset to check
            split: Optional split to check

        Returns:
            bool: True if the dataset exists, False otherwise
        """
        registry = cls._load_registry()
        if name not in registry:
            return False

        if split is not None:
            return split in registry[name]

        return True

    @classmethod
    def remove_dataset_split(cls, name: str, split: str) -> bool:
        """Remove a specific split from a dataset in the registry.

        Args:
            name: Name of the dataset
            split: Split to remove

        Returns:
            bool: True if the split was removed, False otherwise
        """
        registry = cls._load_registry()
        if name not in registry or split not in registry[name]:
            logger.warning(f"Dataset '{name}' split '{split}' not found in registry.")
            return False

        # Get dataset path
        dataset_path = cls._resolve_dataset_path(registry[name][split])

        # Remove file if it exists
        if dataset_path and os.path.exists(dataset_path):
            os.remove(dataset_path)

        # Also remove the Verl-processed file if it exists
        verl_path = dataset_path.replace(".parquet", "_verl.parquet") if dataset_path else None
        if verl_path and os.path.exists(verl_path):
            os.remove(verl_path)

        # Remove split from registry
        del registry[name][split]

        # If no splits left, remove the dataset directory
        if not registry[name]:
            del registry[name]
            dataset_dir = os.path.join(cls._DATASET_DIR, name)
            if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
                os.rmdir(dataset_dir)

        # Update registry
        cls._save_registry(registry)

        logger.info(f"Removed dataset '{name}' split '{split}' from registry.")
        return True

    @classmethod
    def remove_dataset(cls, name: str) -> bool:
        """Remove an entire dataset from the registry and delete its files.

        Args:
            name: Name of the dataset to remove

        Returns:
            bool: True if the dataset was removed, False otherwise
        """
        registry = cls._load_registry()
        if name not in registry:
            logger.warning(f"Dataset '{name}' not found in registry.")
            return False

        # Get dataset paths
        dataset_info = registry[name]

        # Remove files for all splits
        for split, path in dataset_info.items():
            abs_path = cls._resolve_dataset_path(path)
            if abs_path and os.path.exists(abs_path):
                os.remove(abs_path)

            # Also check for and remove verl-processed file if it exists
            verl_path = abs_path.replace(".parquet", "_verl.parquet") if abs_path else None
            if verl_path and os.path.exists(verl_path):
                os.remove(verl_path)

        # Remove dataset directory if it's empty
        dataset_dir = os.path.join(cls._DATASET_DIR, name)
        if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
            os.rmdir(dataset_dir)

        # Update registry
        del registry[name]
        cls._save_registry(registry)

        logger.info(f"Removed dataset '{name}' from registry.")
        return True

    @classmethod
    def apply_verl_postprocessing(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply Verl postprocessing to the dataset.

        Args:
            data: List of dictionaries containing the dataset examples

        Returns:
            List of dictionaries with Verl-compatible format
        """
        processed_data = []
        for entry in data:
            processed_entry = {
                "prompt": [{"role": "user", "content": "placeholder"}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None,
                },
                "extra_info": entry,
            }
            processed_data.append(processed_entry)
        return processed_data
