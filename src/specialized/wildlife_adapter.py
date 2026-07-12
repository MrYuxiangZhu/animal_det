from pathlib import Path

from src.utils.industrial import python_executable, run_command


def run_camera_traps_pipeline(source: str, output: str, cfg) -> None:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        source: 输入图片、视频、目录或数据源路径，由推理入口传入。
        output: 推理结果输出路径，可以是图片、视频、文本或 JSON 文件。
        cfg: 已解析的 YAML 配置字典，包含 project/data/model/train/infer 等运行参数。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
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
