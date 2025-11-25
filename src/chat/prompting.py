# -*- coding: utf-8 -*-
"""
Prompting 模块
处理提示词构建、回复解析、动作提取等
"""
import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from utils.cq import (
    extract_cq_appshare_cards,
    extract_cq_audio_urls,
    extract_cq_file_urls,
    extract_cq_image_urls,
    extract_cq_video_urls,
    extract_http_urls,
)

_log = logging.getLogger(__name__)


def format_multimodal_message(
    content: str,
    image_urls: List[str],
    video_urls: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    格式化多模态消息为Qwen3-VL格式
    
    Args:
        content: 文本内容
        image_urls: 图片URL列表
        video_urls: 视频URL列表
    
    Returns:
        格式化的消息内容列表
    """
    message_content: List[Dict[str, Any]] = []

    if content:
        message_content.append({"type": "text", "text": content})

    for img_url in image_urls:
        message_content.append({"type": "image", "image": img_url})

    if video_urls:
        for v_url in video_urls:
            message_content.append({"type": "video", "video": v_url})

    return message_content


def parse_action_commands(output_text: str) -> List[Dict[str, Any]]:
    """
    从模型输出中解析动作指令
    
    支持格式：
    1) <action>...</action> 内的 JSON
    2) ```json ... ``` 代码块
    3) ACTION: { ... } 或 ACTIONS: [ ... ]
    
    Args:
        output_text: 模型输出文本
    
    Returns:
        动作列表
    """
    candidates: List[str] = []
    
    # 提取 <action>...</action>
    for m in re.finditer(r"<action>([\s\S]+?)</action>", output_text, flags=re.IGNORECASE):
        candidates.append(m.group(1))
    
    # 提取 ```json ... ```
    for m in re.finditer(r"```json\s*([\s\S]+?)\s*```", output_text, flags=re.IGNORECASE):
        candidates.append(m.group(1))
    
    # 提取 ACTION: ... 或 ACTIONS: ...
    for m in re.finditer(
        r"ACTI(?:ON|ONS)\s*:\s*([\s\S]+)$", output_text, flags=re.IGNORECASE | re.MULTILINE
    ):
        candidates.append(m.group(1))

    # 提取 tool_call
    mcp_calls: List[Dict[str, Any]] = []
    for m in re.finditer(
        r"<tool[_-]?call\b[^>]*name=[\"']([^\"']+)[\"'][^>]*>([\s\S]+?)</tool[_-]?call>",
        output_text,
        flags=re.IGNORECASE,
    ):
        name = m.group(1).strip()
        payload_str = m.group(2).strip()
        try:
            payload = json.loads(payload_str)
        except Exception:
            payload = {"raw": payload_str}
        mcp_calls.append({"type": name, **payload})
    if mcp_calls:
        return mcp_calls

    # 解析JSON
    actions: List[Dict[str, Any]] = []
    for text in candidates:
        text = text.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "type" in parsed:
                actions.append(parsed)
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "type" in item:
                        actions.append(item)
        except Exception:
            continue
    return actions


def extract_final_reply(output_text: str) -> Tuple[str, bool, List[Dict[str, Any]]]:
    """
    提取模型输出中的最终回复部分，并判断是否需要回复
    
    Args:
        output_text: 模型输出文本
    
    Returns:
        (回复文本, 是否需要回复, 动作列表)
    """
    thinking_patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<THINKING>.*?</THINKING>",
        r"<THOUGHT>.*?</THOUGHT>",
        r"<analysis>.*?</analysis>",
        r"<reflect>.*?</reflect>",
    ]
    thinking_end_only = [
        r"</think>",
        r"</thinking>",
        r"</THINKING>",
        r"</analysis>",
        r"</reflect>",
    ]
    no_reply_patterns = [
        r"<no_reply\s*/?>",
        r"<no_reply>true</no_reply>",
        r"<no_reply>1</no_reply>",
        r"<no_reply>yes</no_reply>",
    ]

    # 查找最后一个thinking标签
    last_match = None
    last_pattern = None
    for pattern in thinking_patterns:
        matches = list(re.finditer(pattern, output_text, re.IGNORECASE))
        if matches:
            current_match = matches[-1]
            if last_match is None or current_match.end() > last_match.end():
                last_match = current_match
                last_pattern = pattern

    if last_match:
        final_reply = output_text[last_match.end():].strip()
        # 检查是否有no_reply标签
        for no_reply_pattern in no_reply_patterns:
            if re.search(no_reply_pattern, final_reply, re.IGNORECASE):
                _log.info("✅ 模型判断不需要回复")
                return "", False, parse_action_commands(output_text)

        # 清理回复中的tool_call和action标签
        final_reply = re.sub(
            r"<tool_call\b[^>]*>.*?</tool_call>", "", final_reply, flags=re.IGNORECASE | re.DOTALL
        ).strip()
        final_reply = re.sub(r"<action>[\s\S]*?</action>", "", final_reply, flags=re.IGNORECASE)
        final_reply = re.sub(r"```json[\s\S]*?```", "", final_reply, flags=re.IGNORECASE)
        final_reply = re.sub(r'\{[^{}]*"type"[^{}]*\}', "", final_reply, flags=re.IGNORECASE)
        final_reply = re.sub(r'\[[^\[\]]*"type"[^\[\]]*\]', "", final_reply, flags=re.IGNORECASE)
        return final_reply.strip(), True, parse_action_commands(output_text)

    # 如果没有成对的thinking块，但有结束标签
    for end_pat in thinking_end_only:
        m = list(re.finditer(end_pat, output_text, re.IGNORECASE))
        if m:
            last_end = m[-1]
            final_reply = output_text[last_end.end():].strip()
            for no_reply_pattern in no_reply_patterns:
                if re.search(no_reply_pattern, final_reply, re.IGNORECASE):
                    _log.info("✅ 模型判断不需要回复")
                    return "", False, parse_action_commands(output_text)
            final_reply = re.sub(r"<tool_call\b[^>]*>.*?</tool_call>", "", final_reply, flags=re.IGNORECASE | re.DOTALL).strip()
            final_reply = re.sub(r"<action>[\s\S]*?</action>", "", final_reply, flags=re.IGNORECASE)
            final_reply = re.sub(r"```json[\s\S]*?```", "", final_reply, flags=re.IGNORECASE)
            final_reply = re.sub(r'\{[^{}]*"type"[^{}]*\}', "", final_reply, flags=re.IGNORECASE)
            final_reply = re.sub(r'\[[^\[\]]*"type"[^\[\]]*\]', "", final_reply, flags=re.IGNORECASE)
            return final_reply.strip(), True, parse_action_commands(output_text)

    # 如果完全没有thinking标签，检查整个输出中的no_reply
    for pattern in no_reply_patterns:
        if re.search(pattern, output_text, re.IGNORECASE):
            _log.info("✅ 模型判断不需要回复（整个输出中包含no_reply标签）")
            return "", False, parse_action_commands(output_text)

    # 清理并返回整个输出
    cleaned = re.sub(r"<tool_call\b[^>]*>.*?</tool_call>", "", output_text, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"<action>[\s\S]*?</action>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```json[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\{[^{}]*"type"[^{}]*\}', "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[[^\[\]]*"type"[^\[\]]*\]', "", cleaned, flags=re.IGNORECASE)
    return cleaned, True, parse_action_commands(output_text)


def build_system_prompt(config: Dict[str, Any], chat_type: str = None, chat_context: Dict[str, str] = None) -> str:
    """
    构建系统提示词
    
    Args:
        config: 配置字典
        chat_type: "group" 或 "private"
        chat_context: 对话上下文信息（群名、用户昵称等）
    
    Returns:
        系统提示词
    """
    prompt_config = config.get("prompt", {})
    prompt_order = prompt_config.get(
        "prompt_order",
        [
            "context",
            "recall_instruction",
            "output_structure",
            "role_playing",
            "reply_actions",
        ],
    )
    
    # 获取各个组件
    output_structure = prompt_config.get("output_structure", "")
    recall_instruction = prompt_config.get("recall_instruction", "")
    role_playing = prompt_config.get("role_playing", "")
    reply_actions = prompt_config.get("reply_actions", "")
    context_template = prompt_config.get("context_template", {})
    
    # 构建上下文提示
    context_prompt = ""
    if chat_type and chat_context:
        if chat_type == "group":
            template = context_template.get("group", "")
            if template:
                context_prompt = template.format(**chat_context)
        elif chat_type == "private":
            template = context_template.get("private", "")
            if template:
                context_prompt = template.format(**chat_context)
    
    # 检查记忆功能是否启用
    memory_enabled = config.get("memory", {}).get("enabled", False)
    if not memory_enabled:
        recall_instruction = ""  # 如果记忆未启用，不添加回忆指令
    
    # 按照配置的顺序组合提示词
    components = {
        "output_structure": output_structure,
        "recall_instruction": recall_instruction,
        "role_playing": role_playing,
        "reply_actions": reply_actions,
        "context": context_prompt,
    }
    tool_guidance = prompt_config.get("tool_guidance", "")
    if tool_guidance:
        components["tool_guidance"] = tool_guidance
    
    part_labels = {
        "context": "【对话上下文】",
        "recall_instruction": "【回忆机制说明】",
        "output_structure": "【输出结构要求】",
        "role_playing": "【角色设定】",
        "tool_guidance": "【工具使用说明】",
        "reply_actions": "【多样化互动】",
    }
    
    parts: List[str] = []
    for key in prompt_order:
        component = components.get(key, "")
        if not component:
            continue
        segment = component.strip()
        if not segment:
            continue
        label = part_labels.get(key, f"【{key}】")
        if not segment.startswith(label):
            segment = f"{label}\n{segment}"
        parts.append(segment)
    
    if not parts:
        for key in ["context", "recall_instruction", "output_structure", "role_playing", "reply_actions"]:
            segment = components.get(key, "").strip()
            if segment:
                label = part_labels.get(key, f"【{key}】")
                if not segment.startswith(label):
                    segment = f"{label}\n{segment}"
                parts.append(segment)
    
    separator = "\n\n" + "=" * 60 + "\n\n"
    return separator.join(parts).strip()

