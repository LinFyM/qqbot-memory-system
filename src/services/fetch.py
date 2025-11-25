# -*- coding: utf-8 -*-
from typing import Optional, Literal
import os

Mode = Literal["full", "text_only", "summary"]

def _respect_interrupt(interrupt_event) -> bool:
    try:
        return bool(interrupt_event and interrupt_event.is_set())
    except Exception:
        return False

def _truncate_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if max_chars is None or max_chars <= 0:
        return text
    return text[:max_chars] + ("…" if len(text) > max_chars else "")

def _slice_text(text: str, offset: int, chunk_chars: int) -> str:
    if not isinstance(offset, int) or offset < 0:
        offset = 0
    if not isinstance(chunk_chars, int) or chunk_chars <= 0:
        return text[offset:]
    end = offset + chunk_chars
    snippet = text[offset:end]
    suffix = "…(more)" if end < len(text) else ""
    prefix = "(prev)…" if offset > 0 else ""
    return (prefix + snippet + suffix).strip()

def fetch_url_content(
    url: str,
    mode: Mode = "text_only",
    max_chars: int = 12000,
    offset: int = 0,
    chunk_chars: int = 0,
    interrupt_event=None,
    _metrics_add=None,
    _log=None,
) -> dict:
    """
    按需抓取URL内容：
    - text_only: 抽取正文文本（依赖 services.extractors.download_and_extract_webpage）
    - full:     与 text_only 等价（当前阶段不返回原HTML，避免窗口膨胀）
    - summary:  先抽取正文，再进行简单截断（模型侧可继续要求更精细摘要）
    """
    try:
        if _respect_interrupt(interrupt_event):
            return {"status": "interrupted"}
        from .extractors import download_and_extract_webpage as svc_download_and_extract_webpage
        # 针对常见站点设置Headers，降低403/412概率
        default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        if "bilibili.com" in url:
            default_headers["Referer"] = "https://www.bilibili.com/"
        if "zhihu.com" in url:
            default_headers["Referer"] = "https://www.zhihu.com/"
        text = svc_download_and_extract_webpage(url, _metrics_add, _log, headers=default_headers)
        if _respect_interrupt(interrupt_event):
            return {"status": "interrupted"}
        if not text:
            return {"status": "empty", "url": url, "content": ""}
        if mode in ("text_only", "full"):
            content = _truncate_text(text, max_chars or 12000)
            if chunk_chars or offset:
                content = _slice_text(content, offset or 0, chunk_chars or 0)
            return {"status": "ok", "mode": mode, "url": url, "content": content}
        elif mode == "summary":
            # 简摘要：直接更短的截断（更复杂的摘要交给模型）
            content = _truncate_text(text, min(max_chars or 6000, 6000))
            if chunk_chars or offset:
                content = _slice_text(content, offset or 0, chunk_chars or 0)
            return {"status": "ok", "mode": mode, "url": url, "content": content}
        else:
            content = _truncate_text(text, max_chars or 12000)
            if chunk_chars or offset:
                content = _slice_text(content, offset or 0, chunk_chars or 0)
            return {"status": "ok", "mode": "text_only", "url": url, "content": content}
    except Exception as e:
        if _log:
            _log.info(f"fetch_url_content 失败: {e}")
        return {"status": "error", "url": url, "error": str(e)}


def fetch_file_content(
    file_ref: str,
    mode: Mode = "text_only",
    base_static_url: Optional[str] = None,
    max_chars: int = 12000,
    offset: int = 0,
    chunk_chars: int = 0,
    only: Optional[str] = None,
    interrupt_event=None,
    _metrics_add=None,
    _log=None,
) -> dict:
    """
    拉取文件内容：
    - file_ref: 可以是
      1) 本地静态URL（如 http://host:port/static/files/<name>）
      2) 绝对本地路径（位于 FILE_UPLOAD_DIR）
      3) 外部URL（会先缓存到本地，再读取）
    - mode:
      text_only/full: 调用 services.extractors.extract_text_from_file
      summary:        简单截断作为轻摘要
    """
    try:
        # 解析本地路径
        from urllib.parse import urlparse
        from .media import download_file_to_storage as svc_download_file_to_storage
        from .extractors import extract_text_from_file as svc_extract_text_from_file
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_dir = script_dir  # server/
        FILE_UPLOAD_DIR = os.path.join(server_dir, "uploaded_files")
        os.makedirs(FILE_UPLOAD_DIR, exist_ok=True)

        local_path = None
        parsed = urlparse(file_ref)
        if parsed.scheme in ("http", "https"):
            # 本地静态URL
            if base_static_url and file_ref.startswith(base_static_url.rstrip("/") + "/static/files/"):
                filename = file_ref.rsplit("/", 1)[-1]
                local_path = os.path.join(FILE_UPLOAD_DIR, filename)
            else:
                # 外部URL -> 先缓存
                local_url = svc_download_file_to_storage(file_ref, FILE_UPLOAD_DIR, base_static_url, _metrics_add, _log)
                if _respect_interrupt(interrupt_event):
                    return {"status": "interrupted"}
                if local_url and base_static_url and local_url.startswith(base_static_url.rstrip("/") + "/static/files/"):
                    filename = local_url.rsplit("/", 1)[-1]
                    local_path = os.path.join(FILE_UPLOAD_DIR, filename)
        else:
            # 认为是本地路径
            if os.path.isabs(file_ref):
                local_path = file_ref

        if not local_path or not os.path.exists(local_path):
            return {"status": "not_found", "ref": file_ref}

        if _respect_interrupt(interrupt_event):
            return {"status": "interrupted"}
        text = svc_extract_text_from_file(local_path, _metrics_add, _log)
        if _respect_interrupt(interrupt_event):
            return {"status": "interrupted"}
        if not text:
            return {"status": "empty", "ref": file_ref, "content": ""}
        # 轻过滤：tables/code/images_ocr（仅粗粒度）
        if isinstance(only, str):
            only_l = only.strip().lower()
            if only_l == "tables":
                # 提取包含 | 或 \t 的行
                table_lines = [ln for ln in text.splitlines() if ("|" in ln or "\t" in ln)]
                text = "\n".join(table_lines) or text
            elif only_l == "code":
                # 提取看起来像代码的行（缩进或括号/分号密集）
                code_like = []
                for ln in text.splitlines():
                    s = ln.strip()
                    if (ln.startswith("    ") or ln.startswith("\t") or
                        any(tok in s for tok in [";", "{", "}", "def ", "class ", "func ", "=>"])):
                        code_like.append(ln)
                text = "\n".join(code_like) or text
            elif only_l == "images_ocr":
                # 已经在 extract_text_from_file 中做了图片OCR，这里不再加工
                pass
        if mode in ("text_only", "full"):
            content = _truncate_text(text, max_chars or 12000)
            if chunk_chars or offset:
                content = _slice_text(content, offset or 0, chunk_chars or 0)
            return {"status": "ok", "mode": mode, "ref": file_ref, "content": content}
        elif mode == "summary":
            content = _truncate_text(text, min(max_chars or 6000, 6000))
            if chunk_chars or offset:
                content = _slice_text(content, offset or 0, chunk_chars or 0)
            return {"status": "ok", "mode": mode, "ref": file_ref, "content": content}
        else:
            content = _truncate_text(text, max_chars or 12000)
            if chunk_chars or offset:
                content = _slice_text(content, offset or 0, chunk_chars or 0)
            return {"status": "ok", "mode": "text_only", "ref": file_ref, "content": content}
    except Exception as e:
        if _log:
            _log.info(f"fetch_file_content 失败: {e}")
        return {"status": "error", "ref": file_ref, "error": str(e)}


