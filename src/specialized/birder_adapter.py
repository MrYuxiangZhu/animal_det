from pathlib import Path

from src.utils.industrial import python_executable, run_command


def run_birder_inference(source: str, output: str, cfg) -> None:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        source: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        output: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        cfg: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
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
