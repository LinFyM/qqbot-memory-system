# -*- coding: utf-8 -*-
"""
Flask应用装配入口
创建并配置Flask应用，注册路由
"""
import os
import sys
from pathlib import Path
import yaml
import logging

project_root = Path(__file__).resolve().parents[2]
src_dir = project_root / "src"

# ⚠️ 关键：在导入torch之前设置CUDA_VISIBLE_DEVICES
# 先加载配置，检查是否需要设置CUDA_VISIBLE_DEVICES
try:
    config_path = project_root / "configs" / "config_qwen3vl.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            early_config = yaml.safe_load(f)
        device_config = early_config.get("model", {}).get("device", "cuda:0")
        if isinstance(device_config, list):
            # 多GPU配置，提取GPU索引并设置CUDA_VISIBLE_DEVICES
            gpu_indices = []
            for device in device_config:
                if device.startswith("cuda:"):
                    try:
                        gpu_idx = int(device.split(":")[1])
                        gpu_indices.append(str(gpu_idx))
                    except (ValueError, IndexError):
                        pass
            if gpu_indices:
                cuda_visible_devices = ",".join(gpu_indices)
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                print(f"🔧 在导入torch之前设置CUDA_VISIBLE_DEVICES={cuda_visible_devices}（对应实际GPU {device_config}）")
        elif isinstance(device_config, str) and device_config.startswith("cuda:"):
            # 单GPU配置，也需要设置CUDA_VISIBLE_DEVICES
            try:
                gpu_idx = int(device_config.split(":")[1])
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                print(f"🔧 在导入torch之前设置CUDA_VISIBLE_DEVICES={gpu_idx}（对应实际GPU {device_config}）")
            except (ValueError, IndexError):
                print(f"⚠️ 无法解析单GPU配置: {device_config}")
except Exception as e:
    print(f"⚠️ 预加载配置失败，将在模型初始化时设置CUDA_VISIBLE_DEVICES: {e}")

# 设置模块搜索路径
for path in (src_dir, project_root):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

# 配置基础日志（确保所有模块的日志都能输出）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 导入新模块（使用绝对导入，因为src在sys.path中）
from flask import Flask, send_from_directory
import api.server_state as server_state
from api.routes import register_blueprints
from utils.common import get_project_root

_log = logging.getLogger(__name__)


def _resolve_entry_script_path() -> str:
    """确定服务器入口脚本路径，优先使用环境变量"""
    env_path = os.environ.get("SERVER_SCRIPT_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    if hasattr(sys, "argv") and sys.argv and sys.argv[0]:
        candidate = os.path.abspath(sys.argv[0])
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(__file__)


def create_app():
    """
    创建并配置Flask应用
    
    Returns:
        配置好的Flask app实例
    """
    app = Flask(__name__)
    
    # 加载配置
    server_state.load_config()
    config = server_state.config
    
    # 设置上传目录
    server_state.setup_upload_directories()
    
    # 设置server_base_url
    host_for_url = config["server"].get("public_host") or config["server"].get("host", "127.0.0.1")
    if host_for_url in ("0.0.0.0", "::"):
        host_for_url = "127.0.0.1"
    server_state.server_base_url = f"http://{host_for_url}:{config['server']['port']}"
    
    # 静态文件路由
    @app.route("/static/images/<path:filename>")
    def serve_uploaded_image(filename: str):
        return send_from_directory(server_state.IMAGE_UPLOAD_DIR, filename)
    
    @app.route("/static/videos/<path:filename>")
    def serve_uploaded_video(filename: str):
        return send_from_directory(server_state.VIDEO_UPLOAD_DIR, filename)
    
    @app.route("/static/audios/<path:filename>")
    def serve_uploaded_audio(filename: str):
        return send_from_directory(server_state.AUDIO_UPLOAD_DIR, filename)
    
    @app.route("/static/files/<path:filename>")
    def serve_uploaded_file(filename: str):
        return send_from_directory(server_state.FILE_UPLOAD_DIR, filename)
    
    # 注册所有蓝图（使用新的路由系统）
    register_blueprints(app)
    
    # 初始化模型
    try:
        _log.info("=" * 60)
        _log.info("开始查找和加载模型...")
        _log.info("=" * 60)
        
        device = config.get("model", {}).get("device", "cuda:0")
        _log.info(f"📍 配置的目标设备: {device}")
        
        model_path = server_state.find_latest_model(config)
        _log.info(f"📁 选定的模型路径: {model_path}")
        
        server_state.initialize_model(model_path, device)
        _log.info("=" * 60)
        _log.info("✅ 模型初始化流程完成")
        _log.info("=" * 60)
    except Exception as e:
        _log.error("=" * 60)
        _log.error(f"❌ 模型初始化失败: {e}")
        _log.error("=" * 60)
        import traceback
        traceback.print_exc()
    
    # 初始化训练调度器（如果启用）
    try:
        script_path = _resolve_entry_script_path()
        script_args = sys.argv[1:] if hasattr(sys, 'argv') else []
        server_state.server_script_path = script_path
        server_state.server_script_args = script_args
        memory_config = config.get("memory", {}).get("training", {})
        training_enabled = memory_config.get("enabled", False)
        if training_enabled:
            from memory.training_scheduler import MemoryTrainingScheduler
            server_state.training_scheduler = MemoryTrainingScheduler(
                config, script_path, script_args
            )
            server_state.training_scheduler.start()
            _log.info("✅ 训练调度器已启动")
    except Exception as e:
        _log.error(f"❌ 训练调度器初始化失败: {e}", exc_info=True)
    
    return app


# 创建全局app实例
app = create_app()

if __name__ == "__main__":
    config = server_state.config
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 9999)
    app.run(host=host, port=port, debug=False, threaded=True)
