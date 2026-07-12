import importlib
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENCLIP_LOCAL_SRC = PROJECT_ROOT / "third_party" / "open_clip" / "src"


def prefer_local_open_clip() -> bool:
    """把工程内 OpenCLIP 官方源码目录插入 Python 导入路径最前面。

    Returns:
        如果 ``third_party/open_clip/src/open_clip`` 存在并已加入 ``sys.path``，返回 True；否则返回 False。
    """
    package_dir = OPENCLIP_LOCAL_SRC / "open_clip"
    if not package_dir.exists():
        return False
    src = str(OPENCLIP_LOCAL_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return True


def import_open_clip() -> ModuleType:
    """优先导入工程内官方 OpenCLIP 源码，失败时回退到环境安装的 open_clip 包。

    Returns:
        已导入的 ``open_clip`` 模块。调用方可以继续使用 ``create_model_and_transforms``、``get_tokenizer`` 等官方接口。
    """
    prefer_local_open_clip()
    return importlib.import_module("open_clip")
