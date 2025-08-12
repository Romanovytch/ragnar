from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
from markdown_it import MarkdownIt
from ragnar.util import count_tokens


@dataclass
class Unit:
    kind: str                   # "para" | "code"
    text: str
    lang: Optional[str]
    heading_path: List[Tuple[int, str]]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


class MarkdownChunker:
    """
    Heading-aware chunker that preserves fenced code blocks.
    Splits on paragraph boundaries when possible; paragraph-only overlap.
    """
    def __init__(self, target_tokens=800, overlap_tokens=120, max_tokens=1200):
        self.target = target_tokens
        self.overlap = overlap_tokens
        self.max_tokens = max_tokens
        self.md = MarkdownIt("commonmark").enable("table").enable("strikethrough")

    def parse_units(self, text: str) -> List[Unit]:
        tokens = self.md.parse(text)
        units: List[Unit] = []
        heading_stack: List[Tuple[int, str]] = []
        buffer_lines: List[str] = []

        def flush_para():
            if buffer_lines:
                para_text = "\n".join(buffer_lines).strip()
                if para_text:
                    units.append(Unit(kind="para", text=para_text,
                                      lang=None, heading_path=heading_stack.copy()))
                buffer_lines.clear()

        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.type == "heading_open":
                flush_para()
                level = int(t.tag[1]) if t.tag and t.tag.startswith("h") else 1
                title = ""
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    title = tokens[i + 1].content.strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                i += 3
                continue

            if t.type == "fence":
                flush_para()
                lang = (t.info or "").strip() or None
                code = t.content.rstrip("\n")
                fenced = f"```{lang or ''}\n{code}\n```"
                units.append(Unit(kind="code", text=fenced,
                                  lang=lang, heading_path=heading_stack.copy()))
                i += 1
                continue

            if t.type == "paragraph_open":
                j = i + 1
                lines: List[str] = []
                while j < len(tokens) and tokens[j].type != "paragraph_close":
                    if tokens[j].type == "inline":
                        lines.append(tokens[j].content)
                    j += 1
                para = "\n".join(lines).strip()
                if para:
                    buffer_lines.append(para)
                i = j + 1
                flush_para()
                continue

            if t.type in {"bullet_list_open", "ordered_list_open", "blockquote_open", "table_open"}:
                depth = 1
                j = i + 1
                lines: List[str] = []
                while j < len(tokens) and depth > 0:
                    if tokens[j].type.endswith("_open"):
                        depth += 1
                    elif tokens[j].type.endswith("_close"):
                        depth -= 1
                    elif tokens[j].type == "inline":
                        lines.append(tokens[j].content)
                    j += 1
                flush_para()
                para = "\n".join(lines).strip()
                if para:
                    units.append(Unit(kind="para", text=para,
                                      lang=None, heading_path=heading_stack.copy()))
                i = j
                continue

            i += 1

        flush_para()
        return units

    def chunk(self, units: List[Unit]) -> List[tuple[str, List[tuple[int, str]]]]:
        chunks: List[tuple[str, List[tuple[int, str]]]] = []
        buf: List[str] = []
        buf_units: List[Unit] = []
        buf_tokens = 0
        last_para_for_overlap: str | None = None

        def close_chunk():
            nonlocal buf, buf_units, buf_tokens, last_para_for_overlap
            if not buf:
                return
            heading_path = buf_units[-1].heading_path if buf_units else []
            body = "\n\n".join(buf).strip()
            last_para_for_overlap = None
            for u in reversed(buf_units):
                if u.kind == "para":
                    last_para_for_overlap = u.text
                    break
            chunks.append((body, heading_path))
            buf, buf_units, buf_tokens = [], [], 0

        for u in units:
            u_tokens = count_tokens(u.text)

            if not buf:
                if last_para_for_overlap and u.kind == "para":
                    ov = count_tokens(last_para_for_overlap)
                    if ov <= self.overlap:
                        buf.append(last_para_for_overlap)
                        buf_units.append(Unit(kind="para", text=last_para_for_overlap,
                                              lang=None, heading_path=u.heading_path))
                        buf_tokens += ov
                buf.append(u.text)
                buf_units.append(u)
                buf_tokens += u_tokens
                continue

            if buf_tokens + u_tokens > self.max_tokens:
                close_chunk()
                if last_para_for_overlap and u.kind == "para":
                    ov = count_tokens(last_para_for_overlap)
                    if ov <= self.overlap:
                        buf.append(last_para_for_overlap)
                        buf_units.append(Unit(kind="para", text=last_para_for_overlap,
                                              lang=None, heading_path=u.heading_path))
                        buf_tokens += ov
                buf.append(u.text)
                buf_units.append(u)
                buf_tokens += u_tokens
                continue

            if buf_tokens + u_tokens > self.target and u.kind == "para":
                close_chunk()
                if last_para_for_overlap:
                    ov = count_tokens(last_para_for_overlap)
                    if ov <= self.overlap:
                        buf.append(last_para_for_overlap)
                        buf_units.append(Unit(kind="para", text=last_para_for_overlap,
                                              lang=None, heading_path=u.heading_path))
                        buf_tokens += ov
                buf.append(u.text)
                buf_units.append(u)
                buf_tokens += u_tokens
                continue

            buf.append(u.text)
            buf_units.append(u)
            buf_tokens += u_tokens

        close_chunk()
        return chunks
