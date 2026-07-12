from pathlib import Path

from src.utils.industrial import require_package


def run_superanimal_inference(source: str, output_dir: str, cfg) -> None:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        source: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        output_dir: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        cfg: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
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
