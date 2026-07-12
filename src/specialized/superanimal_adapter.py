from pathlib import Path

from src.utils.industrial import require_package


def run_superanimal_inference(source: str, output_dir: str, cfg) -> None:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        source: 输入图片、视频、目录或数据源路径，由推理入口传入。
        output_dir: 输出目录路径，用于保存模型结果、日志或第三方框架产物。
        cfg: 已解析的 YAML 配置字典，包含 project/data/model/train/infer 等运行参数。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    require_package("deeplabcut", "pip install 'deeplabcut[gui]' 或参考 DeepLabCut 官方安装文档")
    import deeplabcut

    super_cfg = cfg["superanimal"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    deeplabcut.video_inference_superanimal(
        [source],
        superanimal_name=super_cfg["superanimal_name"],
        model_name=super_cfg["model_name"],
        detector_name=super_cfg["detector_name"],
        video_adapt=super_cfg["video_adapt"],
        destfolder=output_dir,
    )
