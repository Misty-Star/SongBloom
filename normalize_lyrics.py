import re
import typing as tp


STRUCTURE_TAGS = ["intro", "outro", "verse", "chorus", "bridge", "inst", "silence"]
TAG_PATTERN = r"\[(" + "|".join(STRUCTURE_TAGS) + r")\]"
TIMESTAMP_PATTERN = re.compile(r"(?:\[\d{1,2}:\d{1,2}(?:[.:]\d{1,3})?\]|<\d{1,2}:\d{1,2}(?:[.:]\d{1,3})?>)")
LRC_METADATA_PATTERN = re.compile(
    r"^\[(?:ar|ti|al|by|hash|offset|kana|language|length|re|ve):.*\]$",
    flags=re.IGNORECASE,
)
PURE_MUSIC_PATTERNS = [
    re.compile(r"^(?:该歌曲为纯音乐.*请欣赏|纯音乐.*请欣赏)$", flags=re.IGNORECASE),
]
PLATFORM_NOISE_PHRASES = [
    "贡献歌词",
    "贡献滚动歌词",
    "期待您的精彩评论",
    "腾讯音乐",
    "qq音乐官方",
    "词曲版权由腾讯音乐提供",
]
COLON_CREDIT_KEYWORDS = [
    "作词", "词", "lyrics by", "lyricist", "text",
    "作曲", "曲", "composer", "composed by", "music by",
    "编曲", "arranger", "arrangement", "music arrangement",
    "制作人", "producer", "music producer",
    "监制", "executive producer", "producer management",
    "混音", "mixing", "mix", "mixing engineer",
    "母带", "mastering", "master", "mastered by",
    "录音", "recording", "recording engineer",
    "和声", "backing vocals", "harmony", "background vocals",
    "吉他", "guitar",
    "贝斯", "bass",
    "鼓", "drums",
    "弦乐", "弦乐编写", "钢琴",
    "音频编辑", "vocal editing", "pro tools",
    "翻译", "translator", "中文作词",
    "原唱", "original artist", "cover",
    "版权", "copyright", "provided by", "published by",
    "op", "sp",
    "artist", "title", "album", "lrc by",
]
NO_COLON_PREFIXES = [
    "lyrics by", "lyricist", "composed by", "composer", "music by",
    "arranged by", "arranger", "producer", "music producer",
    "executive producer", "mixing engineer", "mastered by",
    "recording engineer", "translator", "original artist", "provided by",
]


def _normalize_line_text(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _strip_timestamps(line: str) -> str:
    return _normalize_line_text(TIMESTAMP_PATTERN.sub(" ", line))


def _is_platform_noise(line: str) -> bool:
    lowered = line.lower()
    return any(phrase.lower() in lowered for phrase in PLATFORM_NOISE_PHRASES)


def _extract_credit_prefix(line: str) -> tp.Optional[str]:
    match = re.match(r"^\s*([^:：]{1,40})\s*[:：]\s*(.+)$", line)
    if not match:
        return None
    return match.group(1).strip(" []()（）【】").lower()


def _is_credit_line(line: str) -> bool:
    candidate = line.strip(" -–—•·")
    if not candidate:
        return False

    lowered = candidate.lower()
    prefix = _extract_credit_prefix(candidate)
    if prefix:
        normalized_prefix = re.sub(r"\s+", " ", prefix)
        if normalized_prefix in {item.lower() for item in COLON_CREDIT_KEYWORDS}:
            return True
        if any(keyword.lower() in normalized_prefix for keyword in COLON_CREDIT_KEYWORDS):
            return True

    for prefix in NO_COLON_PREFIXES:
        if lowered.startswith(prefix + " "):
            return True

    if re.match(r"^(?:作词|作曲|编曲|制作人|监制|混音|母带|录音|和声|吉他|贝斯|鼓|翻译|原唱|版权|提供)\s+\S+", candidate):
        return True
    if re.match(r"^(?:op|sp)\s+\S+", lowered):
        return True

    return False


def _remove_metadata_and_credits(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw_line in text.split("\n"):
        line = _strip_timestamps(raw_line)
        if not line:
            continue
        if LRC_METADATA_PATTERN.fullmatch(line):
            continue
        if any(pattern.fullmatch(line) for pattern in PURE_MUSIC_PATTERNS):
            continue
        if _is_platform_noise(line):
            continue
        if _is_credit_line(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def clean_lyrics(text):
    text = _remove_metadata_and_credits(text)

    # 替换省略号、其他标点 → .
    text = re.sub(r"[!?;,:…！？，]", ".", text)
    # 暂存结构标识后面的逗号（防止被替换掉）
    text = re.sub(rf"({TAG_PATTERN})\s*[,.]", r"\1 ", text, flags=re.IGNORECASE)

    # 按结构标识切分
    parts = re.split(r"(\[[^\]]+\])", text)
    cleaned_parts = []

    for part in parts:
        if not part.strip():
            continue

        if re.fullmatch(TAG_PATTERN, part.strip(), re.IGNORECASE):
            cleaned_parts.append(part.strip().lower() + " ")
        else:
            part = re.sub(r"[\"“”‘’«»〈〉《》【】\(\)\{\}]", "", part)
            part = re.sub(r"\s+", " ", part.strip())
            cleaned_parts.append(part.strip())

    result_parts = []
    for i, seg in enumerate(cleaned_parts):
        result_parts.append(seg)

        if i < len(cleaned_parts) - 1:
            curr = cleaned_parts[i].strip()
            nxt = cleaned_parts[i + 1].strip()

            is_curr_tag = re.fullmatch(TAG_PATTERN, curr.split()[0]) if curr else False
            is_next_tag = re.fullmatch(TAG_PATTERN, nxt.split()[0]) if nxt else False

            if is_curr_tag and is_next_tag:
                if curr.split()[0].lower() == nxt.split()[0].lower():
                    continue
                result_parts.append(" , ")
            elif is_curr_tag or is_next_tag:
                if curr not in ["[verse]", "[chorus]", "[bridge]"]:
                    result_parts.append(" , ")

    result = "".join(result_parts)
    result = re.sub(r"\s+,", " ,", result)
    result = re.sub(r",\s+", ", ", result)
    result = re.sub(r"\.{2,}", ".", result)
    result = re.sub(r"\.\s*,", ",", result)
    result = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1.\2", result)

    return result.strip()
