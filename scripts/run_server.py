#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服务器启动脚本
使用方法: python scripts/run_server.py
"""
from pathlib import Path
import sys
import os
import logging


def _setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def _prepare_sys_path():
    """准备Python模块搜索路径"""
    # 项目根目录
    root = Path(__file__).resolve().parents[1]
    # src目录
    src = root / "src"
    
    # 确保这些路径在sys.path中
    for path in (src, root):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    
    # 设置环境变量，便于模块查找
    os.environ.setdefault("PROJECT_ROOT", str(root))
    os.environ.setdefault("SERVER_SCRIPT_PATH", str(Path(__file__).resolve()))


def main():
    """主函数"""
    # 准备路径（在导入前）
    _prepare_sys_path()
    
    # 配置日志（在导入模块前）
    _setup_logging()
    
    # 导入并运行应用
    try:
        from api.app import app, server_state
        
        # 获取配置
        config = server_state.config
        host = config.get("server", {}).get("host", "0.0.0.0")
        port = config.get("server", {}).get("port", 9999)
        
        print("=" * 60)
        print("🚀 萝卜子QQ机器人服务器")
        print("=" * 60)
        print(f"📡 监听地址: {host}:{port}")
        print(f"🌐 访问地址: http://127.0.0.1:{port}")
        print("=" * 60)
        print("按 Ctrl+C 停止服务器")
        print("=" * 60)
        
        # 启动服务器
        app.run(host=host, port=port, debug=False, threaded=True)
        
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
