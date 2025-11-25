# -*- coding: utf-8 -*-
from typing import Optional, Callable
import os

Logger = object

_asr_inited = False
_asr_backend = None  # "faster-whisper" | "openai-whisper"
_asr_model = None

def _lazy_init_asr(preferred_backend: str = "faster-whisper",
                   model_size: str = "base",
                   device: str = "cpu",
                   logger: Optional[Logger] = None) -> None:
    global _asr_inited, _asr_backend, _asr_model
    if _asr_inited:
        return
    backend = None
    model = None
    try:
        if preferred_backend == "faster-whisper":
            from faster_whisper import WhisperModel
            backend = "faster-whisper"
            model = WhisperModel(model_size, device=device, compute_type="int8" if device == "cpu" else "float16")
        else:
            import whisper
            backend = "openai-whisper"
            model = whisper.load_model(model_size, device=device)
    except Exception:
        try:
            import whisper
            backend = "openai-whisper"
            model = whisper.load_model(model_size, device=device)
        except Exception:
            backend = None
            model = None
    _asr_backend = backend
    _asr_model = model
    _asr_inited = True
    if logger: logger.info(f"ASR 初始化: backend={_asr_backend}, model={'ok' if _asr_model else 'none'}")

def transcribe_audio(local_audio_path: str,
                     metrics_add: Optional[Callable[[str, int], None]] = None,
                     logger: Optional[Logger] = None) -> str:
    _lazy_init_asr(logger=logger)
    if not _asr_model or not _asr_backend:
        if logger: logger.warning("ASR 后端不可用，跳过转写")
        if metrics_add: metrics_add("asr_unavailable", 1)
        return ""
    try:
        if _asr_backend == "faster-whisper":
            segments, info = _asr_model.transcribe(local_audio_path)
            text = " ".join([seg.text for seg in segments]).strip()
        else:
            result = _asr_model.transcribe(local_audio_path, task="transcribe", fp16=False)
            text = (result.get("text") or "").strip()
        if text:
            if metrics_add: metrics_add("asr_ok", 1)
        else:
            if metrics_add: metrics_add("asr_empty", 1)
        return text
    except Exception as e:
        if logger: logger.warning(f"ASR 转写失败: {e}")
        if metrics_add: metrics_add("asr_fail", 1)
        return ""


