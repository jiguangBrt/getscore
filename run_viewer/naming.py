"""Shared experiment names and discovery of unique benchmark cases."""

from pathlib import Path
import re


def experiment_name(simulation, experiment, language_model, action_model):
    """Keep the task number before the two model identifiers."""
    parts = (str(simulation), str(experiment), str(language_model), str(action_model))
    if any(not part or part in {".", ".."} or "/" in part or "\\" in part for part in parts):
        raise ValueError("实验名称各部分不能为空或包含路径分隔符。")
    if not parts[1].isascii() or not parts[1].isdigit():
        raise ValueError("试验编号必须是非负整数。")
    return "-".join(parts)


def experiment_parts(name):
    """Read canonical names and the previous model-model-number order."""
    if re.search(r"-[0-9]{8}-[0-9]{6}$", name):
        return None
    parts = name.rsplit("-", 3)
    if len(parts) != 4 or not all(parts):
        return None
    simulation, first, second, third = parts
    if first.isascii() and first.isdigit():
        experiment, language_model, action_model = first, second, third
    elif third.isascii() and third.isdigit():
        experiment, language_model, action_model = third, first, second
    else:
        return None
    return dict(simulation=simulation, experiment=experiment,
                language_model=language_model, action_model=action_model)


def canonical_name(name):
    parts = experiment_parts(name)
    return experiment_name(**parts) if parts else name


def case_directories(batch):
    """Count each recording once when historical and current aliases coexist."""
    seen = set()
    paths = sorted((Path(batch) / "cases").iterdir(),
                   key=lambda path: (canonical_name(path.name) != path.name, path.name))
    for path in paths:
        if not (path / "configuration.json").is_file():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield path
