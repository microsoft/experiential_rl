from collections import deque
import numpy as np

from rllm.data.dataset import DatasetRegistry


def _neighbors(pos: tuple[int, int]):
    r, c = pos
    return [(r, c - 1), (r + 1, c), (r, c + 1), (r - 1, c)]


def _parse_layout(layout: list[str]):
    walls: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()
    player = None
    boxes: set[tuple[int, int]] = set()
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((r, c))
            elif ch == ".":
                goals.add((r, c))
            elif ch == "$":
                boxes.add((r, c))
            elif ch == "@":
                player = (r, c)
            elif ch == "+":
                player = (r, c)
                goals.add((r, c))
            elif ch == "*":
                boxes.add((r, c))
                goals.add((r, c))
    return walls, goals, boxes, player


def _shortest_solution_steps(layout: list[str], max_depth: int) -> int | None:
    """Return shortest solution length (moves) or None if unsolved within max_depth."""
    walls, goals, boxes, player = _parse_layout(layout)
    if not goals or len(boxes) != 1 or player is None:
        return None
    goal = next(iter(goals))
    start_box = next(iter(boxes))
    height, width = len(layout), len(layout[0])

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < height and 0 <= c < width

    start = (player, start_box)
    queue: deque[tuple[tuple[int, int], tuple[int, int], int]] = deque([(player, start_box, 0)])
    visited = {start}

    def free(cell: tuple[int, int], box_pos: tuple[int, int]) -> bool:
        return in_bounds(cell) and cell not in walls and cell != box_pos

    while queue:
        p_pos, b_pos, steps = queue.popleft()
        if b_pos == goal:
            return steps
        if steps >= max_depth:
            continue

        for nbr in _neighbors(p_pos):
            # Walk without pushing
            if free(nbr, b_pos):
                state = (nbr, b_pos)
                if state not in visited:
                    visited.add(state)
                    queue.append((nbr, b_pos, steps + 1))
            # Push
            if nbr == b_pos:
                push_to = (b_pos[0] + (b_pos[0] - p_pos[0]), b_pos[1] + (b_pos[1] - p_pos[1]))
                if free(push_to, b_pos):
                    state = (nbr, push_to)
                    if state not in visited:
                        visited.add(state)
                        queue.append((nbr, push_to, steps + 1))
    return None


def _generate_layout(rng: np.random.Generator, size_range: tuple[int, int], max_solution_steps: int, max_attempts: int = 5000) -> dict:
    min_size, max_size = size_range
    attempts = 0
    seen_layouts: set[tuple[str, ...]] = set()
    while attempts < max_attempts:
        attempts += 1
        size = int(rng.integers(min_size, max_size + 1))
        # Build empty grid with border walls.
        grid = [["#" if r in (0, size - 1) or c in (0, size - 1) else " " for c in range(size)] for r in range(size)]

        # Sample goal, box, player (single box puzzle).
        inner_cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
        goal = inner_cells[rng.integers(len(inner_cells))]
        box = goal
        while box == goal:
            box = inner_cells[rng.integers(len(inner_cells))]
        player = box
        while player in (goal, box):
            player = inner_cells[rng.integers(len(inner_cells))]

        grid[goal[0]][goal[1]] = "."
        grid[box[0]][box[1]] = "$"
        grid[player[0]][player[1]] = "@"

        layout = ["".join(row) for row in grid]
        layout_key = tuple(layout)
        if layout_key in seen_layouts:
            continue

        steps = _shortest_solution_steps(layout, max_solution_steps)
        if steps is not None and steps <= max_solution_steps:
            layout_name = f"sokoban_{size}x{size}_g{goal[0]}{goal[1]}_b{box[0]}{box[1]}_p{player[0]}{player[1]}"
            return {
                "layout": layout,
                "layout_name": layout_name,
                "max_steps": max_solution_steps,
            }
        seen_layouts.add(layout_key)

    raise RuntimeError(f"Failed to generate a solvable layout within {max_solution_steps} steps after {max_attempts} attempts.")


def prepare_sokoban_data(train_size: int = 10000, test_size: int = 100, size_range: tuple[int, int] = (6, 8), max_solution_steps: int = 8):
    """
    Prepare and register Sokoban datasets with variable map sizes and guaranteed ≤8-step solutions.

    Args:
        train_size: Number of training examples to generate.
        test_size: Number of test examples to generate.
        size_range: Inclusive (min, max) grid size, e.g., (6, 8).
        max_solution_steps: Maximum allowed optimal solution length; also used as max_steps for the env.
    """
    rng = np.random.default_rng(seed=42)

    global_seen: set[tuple[str, ...]] = set()

    def build_split(count: int, offset: int) -> list[dict]:
        layouts: list[dict] = []
        attempts = 0
        while len(layouts) < count:
            attempts += 1
            task = _generate_layout(rng, size_range, max_solution_steps, max_attempts=10000)
            key = tuple(task["layout"])
            if key in global_seen:
                continue
            global_seen.add(key)
            task["uid"] = f"{task['layout_name']}_{len(layouts)+offset}"
            task["index"] = len(layouts) + offset
            layouts.append(task)
        return layouts

    train_data = build_split(train_size, offset=0)
    test_data = build_split(test_size, offset=train_size)

    train_dataset = DatasetRegistry.register_dataset("sokoban", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("sokoban", test_data, "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_sokoban_data()
    print(f"Train dataset: {len(train_dataset.get_data())} examples")
    print(f"Test dataset: {len(test_dataset.get_data())} examples")
    print("Sample train example:", train_dataset.get_data()[0])
    print("Sample test example:", test_dataset.get_data()[0])
