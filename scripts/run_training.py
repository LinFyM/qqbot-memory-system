"""
离线运行记忆训练（不经过HTTP接口）
python scripts/run_training.py
"""
from pathlib import Path
import sys


def _prepare_sys_path():
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    for path in (root, src):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def main():
    _prepare_sys_path()
    from server import api_server_qwen3vl as api  # noqa: WPS433
    from server.memory.training_service import MemoryTrainingService  # noqa: WPS433

    config = api.load_config(None)
    service = MemoryTrainingService(config)
    trained_dir = service.run_training()
    if trained_dir:
        print(f"✅ 训练完成，模型保存于: {trained_dir}")
    else:
        print("⚠️ 训练未完成或被跳过")


if __name__ == "__main__":
    main()
