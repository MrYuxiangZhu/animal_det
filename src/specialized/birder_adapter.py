from pathlib import Path

from src.utils.industrial import python_executable, run_command


def run_birder_inference(source: str, output: str, cfg) -> None:
    birder_cfg = cfg["birder"]
    repo = Path(birder_cfg["repo"])
    if not repo.exists():
        raise RuntimeError(f"未找到 Birder 仓库: {repo}。请先 clone https://github.com/birder-project/birder")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_executable(),
        birder_cfg["entry_script"],
        "--network",
        birder_cfg["model_name"],
        "--input",
        source,
        "--output",
        output,
        "--top-k",
        str(birder_cfg["topk"]),
    ]
    if birder_cfg.get("checkpoint"):
        cmd.extend(["--checkpoint", birder_cfg["checkpoint"]])
    run_command(cmd, cwd=str(repo))
