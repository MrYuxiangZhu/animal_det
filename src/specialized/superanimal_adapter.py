from pathlib import Path

from src.utils.industrial import require_package


def run_superanimal_inference(source: str, output_dir: str, cfg) -> None:
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
