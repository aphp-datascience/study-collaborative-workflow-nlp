import re
from pathlib import Path
from typing import Union


def ckpt_from_config(
    config_name: str, ckpt_dir: Union[str, Path], version: str = "last"
):
    ckpt_dir = Path(ckpt_dir)

    config_path = config_name + "{version}.ckpt"
    versions = sorted(
        [
            re.search(config_path.format(version="(.*)"), str(f)).groups()[0]
            for f in ckpt_dir.glob(config_path.format(version="*"))
        ]
    )

    if version == "first":
        ckpt_path = ckpt_dir / config_path.format(version="")
    elif version == "last":
        ckpt_path = ckpt_dir / config_path.format(version=versions[-1])
    else:
        versions = [v.lstrip("-") for v in versions if v.startswith("-")]
        if version not in versions:
            raise ValueError(
                f"No matching version found (given: {version}, available: {versions})"
            )

        ckpt_path = ckpt_dir / config_path.format(version=f"-{version}")

    print(f"Retrieved checkpoint: {ckpt_path}")

    return ckpt_path
