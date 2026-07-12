from pathlib import Path

from src.utils.industrial import python_executable, run_command


def run_camera_traps_pipeline(source: str, output: str, cfg) -> None:
    wildlife_cfg = cfg["pytorch_wildlife"]
    repo = Path(wildlife_cfg["repo"])
    if not repo.exists():
        raise RuntimeError(f"未找到 CameraTraps/Pytorch-Wildlife 仓库: {repo}。请先 clone 对应仓库。")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_executable(),
        wildlife_cfg["entry_script"],
        "--input",
        source,
        "--output",
        output,
        "--detector",
        wildlife_cfg["detector_weights"],
        "--classifier",
        wildlife_cfg["classifier_weights"],
    ]
    run_command(cmd, cwd=str(repo))
