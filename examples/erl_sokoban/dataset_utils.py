from __future__ import annotations

from typing import Any

from rllm.data.dataset import Dataset, DatasetRegistry

DEFAULT_LAYOUTS: list[dict[str, Any]] = [
    {
        "layout_name": "basic_push",
        "layout": [
            "#######",
            "#@ $. #",
            "#  #  #",
            "#     #",
            "#     #",
            "#######",
        ],
        "max_steps": 48,
    },
    {
        "layout_name": "double_box_corridor",
        "layout": [
            "########",
            "#  .  .#",
            "# $$#  #",
            "#  @   #",
            "#   #  #",
            "########",
        ],
        "max_steps": 72,
    },
]


def build_default_tasks() -> list[dict[str, Any]]:
    """Return a small set of hand-crafted Sokoban tasks for quick-start experiments."""
    tasks: list[dict[str, Any]] = []
    for entry in DEFAULT_LAYOUTS:
        task = {
            "layout": entry["layout"],
            "layout_name": entry.get("layout_name", "default"),
            "max_steps": entry.get("max_steps"),
        }
        tasks.append(task)
    return tasks


def ensure_sokoban_dataset(split: str = "train") -> Dataset:
    """Load a registered Sokoban dataset, or create a small default one if missing."""
    dataset = DatasetRegistry.load_dataset("sokoban", split)
    if dataset is not None:
        return dataset

    tasks = build_default_tasks()
    return DatasetRegistry.register_dataset("sokoban", tasks, split=split)
