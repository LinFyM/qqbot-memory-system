# -*- coding: utf-8 -*-
from typing import Tuple, List

def extract_cq_image_urls(content: str) -> Tuple[str, List[str]]:
    import re
    from urllib.parse import unquote
    image_urls: List[str] = []
    pattern = r'\[CQ:image[^\]]*\]'
    matches = re.finditer(pattern, content)
    for match in matches:
        cq_code = match.group(0)
        url_match = re.search(r'url=([^,\]]+)', cq_code) or re.search(r'url=([^\]]+)', cq_code)
        if url_match:
            url = url_match.group(1).rstrip(']')
            url = url.replace('&amp;', '&')
            try:
                url = unquote(url)
            except Exception:
                pass
            image_urls.append(url)
    cleaned_content = re.sub(pattern, '', content).strip()
    return cleaned_content, image_urls

def extract_cq_video_urls(content: str) -> Tuple[str, List[str]]:
    import re
    from urllib.parse import unquote
    video_urls: List[str] = []
    pattern = r'\[CQ:video[^\]]*\]'
    matches = re.finditer(pattern, content)
    for match in matches:
        cq_code = match.group(0)
        # 提取所有可能包含视频URL的字段
        url_match = None
        # 按优先级提取：url > file > file_name
        for field in ['url', 'file', 'file_name']:
            url_match = (re.search(rf'{field}=([^,\]]+)', cq_code) or
                        re.search(rf'{field}=([^\]]+)', cq_code))
            if url_match:
                break

        if url_match:
            url = url_match.group(1).rstrip(']')
            url = url.replace('&amp;', '&')
            try:
                url = unquote(url)
            except Exception:
                pass
            video_urls.append(url)
    cleaned_content = re.sub(pattern, '', content).strip()
    return cleaned_content, video_urls

def extract_cq_audio_urls(content: str) -> Tuple[str, List[str]]:
    import re
    from urllib.parse import unquote
    audio_urls: List[str] = []
    pattern = r'\[CQ:record[^\]]*\]'
    matches = re.finditer(pattern, content)
    for match in matches:
        cq_code = match.group(0)
        url_match = re.search(r'url=([^,\]]+)', cq_code) or re.search(r'url=([^\]]+)', cq_code)
        if url_match:
            url = url_match.group(1).rstrip(']')
            url = url.replace('&amp;', '&')
            try:
                url = unquote(url)
            except Exception:
                pass
            audio_urls.append(url)
    cleaned_content = re.sub(pattern, '', content).strip()
    return cleaned_content, audio_urls

def extract_cq_file_urls(content: str) -> Tuple[str, List[str]]:
    import re
    from urllib.parse import unquote
    file_urls: List[str] = []
    pattern = r'\[CQ:file[^\]]*\]'
    matches = re.finditer(pattern, content)
    for match in matches:
        cq_code = match.group(0)
        url_match = re.search(r'url=([^,\]]+)', cq_code) or re.search(r'url=([^\]]+)', cq_code)
        if url_match:
            url = url_match.group(1).rstrip(']')
            url = url.replace('&amp;', '&')
            try:
                url = unquote(url)
            except Exception:
                pass
            file_urls.append(url)
    cleaned_content = re.sub(pattern, '', content).strip()
    return cleaned_content, file_urls

def extract_http_urls(text: str, max_urls: int = 5) -> List[str]:
    try:
        import re
        urls = re.findall(r'(https?://[^\s<>\"]+)', text or "")
        seen = set()
        uniq: List[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
            if len(uniq) >= max_urls:
                break
        return uniq
    except Exception:
        return []


def extract_cq_appshare_cards(content: str) -> Tuple[str, List[dict]]:
    """
    提取 QQ 小程序/应用卡片（[CQ:json] / [CQ:xml]）中的关键信息与可用URL
    返回:
      cleaned_content: 去除卡片CQ码后的文本
      cards: [{ 'url': str, 'title': str, 'app': str, 'raw': str }]
    兼容常见平台（如小红书/微信/哔哩哔哩等）的卡片JSON/XML，尽力从中提取URL与标题。
    """
    import re, json
    from urllib.parse import unquote

    cards: List[dict] = []

    # 1) 提取 JSON 卡片
    json_pattern = r'\[CQ:json[^\]]*data=([^\]]+)\]'
    # 2) 提取 XML 卡片
    xml_pattern = r'\[CQ:xml[^\]]*data=([^\]]+)\]'

    def decode_data(raw: str) -> str:
        s = raw
        s = s.rstrip(']')
        s = s.replace('&amp;', '&')
        try:
            s = unquote(s)
        except Exception:
            pass
        return s

    def pick_url_and_title_from_obj(obj: dict) -> tuple:
        # 广撒网找 URL
        text = json.dumps(obj, ensure_ascii=False)
        urls = re.findall(r'(https?://[^\s"<>\']+)', text)
        url = urls[0] if urls else ""
        # 常见字段取标题
        title = obj.get("title") or obj.get("prompt") or obj.get("desc") or ""
        # app 名称尝试从 app 字段或 meta 内部
        app = obj.get("app") or obj.get("appName") or ""
        if not app:
            meta = obj.get("meta") or {}
            if isinstance(meta, dict):
                app = meta.get("appName") or meta.get("tag") or ""
        return url, title, app

    # 处理 JSON
    def harvest_from_matches(pattern: str, content_text: str):
        results = []
        for m in re.finditer(pattern, content_text):
            raw = m.group(0)
            data_raw = m.group(1)
            data_text = decode_data(data_raw)
            # 数据可能再次包含转义的 JSON
            parsed = None
            try:
                parsed = json.loads(data_text)
            except Exception:
                # 兜底：尝试去除外层引号后再解析
                try:
                    if data_text and (data_text[0] in ['"', "'"]) and data_text[-1] == data_text[0]:
                        parsed = json.loads(data_text[1:-1])
                except Exception:
                    parsed = None
            if isinstance(parsed, dict):
                url, title, app = pick_url_and_title_from_obj(parsed)
                if url:
                    results.append({
                        "url": url,
                        "title": title or "",
                        "app": app or "",
                        "raw": raw
                    })
        return results

    cards.extend(harvest_from_matches(json_pattern, content))
    cards.extend(harvest_from_matches(xml_pattern, content))

    # 去除匹配到的 CQ 片段
    cleaned_content = re.sub(json_pattern, '', content)
    cleaned_content = re.sub(xml_pattern, '', cleaned_content).strip()
    return cleaned_content, cards

