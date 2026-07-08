from __future__ import annotations

from dataclasses import dataclass

from markdown_it import MarkdownIt

from agora.util import count_tokens


@dataclass
class Unit:
    kind: str  # "para" | "code"
    text: str
    lang: str | None
    heading_path: list[tuple[int, str]]  # breadcrumbs


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict  # titles, chapters, token_count...


@dataclass
class ChunkSpan:
    text: str
    heading_path: list[tuple[int, str]]
    start_unit: int
    end_unit: int


class MarkdownChunker:
    """Chunk Markdown into model-friendly pieces while preserving code fences.

    Chunks are assembled from paragraph and fenced-code units with a soft target
    size and a hard ceiling. Code fences are never split.

    Args:
        target_tokens: Soft target size for a chunk (tokens).
        overlap_tokens: Max paragraph-only overlap between consecutive chunks.
        max_tokens: Hard ceiling for a chunk size (tokens).
        split_heading_level: Optional exact heading level that defines section
            boundaries. When omitted, any heading path change closes the chunk.
    """

    def __init__(
        self,
        target_tokens: int = 800,
        overlap_tokens: int = 120,
        max_tokens: int = 1200,
        split_heading_level: int | None = None,
    ) -> None:
        self.target = target_tokens
        self.overlap = overlap_tokens  # paragraph-only
        self.max_tokens = max_tokens
        if split_heading_level is not None and not 1 <= split_heading_level <= 6:
            raise ValueError("split_heading_level must be between 1 and 6")
        self.split_heading_level = split_heading_level
        self.md = MarkdownIt("commonmark").enable("table").enable("strikethrough")

    def parse_units(self, text: str) -> list[Unit]:
        """Parse Markdown into atomic units (paragraphs and fenced code).

        The parser walks the Markdown-It token stream, producing `Unit` objects:

        - `kind="para"` for paragraphs and for flattened blocks (lists, blockquotes, tables).
        - `kind="code"` for fenced code blocks; fences are kept whole and never split.

        Each unit carries a `heading_path` (list of `(level, title)` tuples) reflecting the
        active H1/H2/... when the unit was encountered.

        Args:
            text: Raw Markdown source.

        Returns:
            A list of `Unit` objects in document order.

        Notes:
            - Lists and tables are flattened by concatenating their inline text.
            - Code fences are reconstructed as Markdown (triple backticks), and `lang`
            is taken from the fence info string.
        """
        tokens = self.md.parse(text)
        units: list[Unit] = []
        heading_stack: list[tuple[int, str]] = []
        buffer_lines: list[str] = []

        def flush_para():
            if buffer_lines:
                para_text = "\n".join(buffer_lines).strip()
                if para_text:
                    units.append(
                        Unit(
                            kind="para",
                            text=para_text,
                            lang=None,
                            heading_path=heading_stack.copy(),
                        )
                    )
                buffer_lines.clear()

        i = 0
        while i < len(tokens):
            t = tokens[i]

            # Handle the heading stack (titles, chapters...)
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

            # Handle pieces of code
            if t.type == "fence":
                flush_para()
                lang = (t.info or "").strip() or None
                code = t.content.rstrip("\n")
                fenced = f"```{lang or ''}\n{code}\n```"
                units.append(
                    Unit(kind="code", text=fenced, lang=lang, heading_path=heading_stack.copy())
                )
                i += 1
                continue

            # Handle paragraphs
            if t.type == "paragraph_open":
                j = i + 1
                lines: list[str] = []
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

            # Handle lists, quotes and tables (flatten)
            if t.type in {"bullet_list_open", "ordered_list_open", "blockquote_open", "table_open"}:
                depth = 1
                j = i + 1
                lines: list[str] = []
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
                    units.append(
                        Unit(kind="para", text=para, lang=None, heading_path=heading_stack.copy())
                    )
                i = j
                continue

            i += 1

        flush_para()
        return units

    def chunk_spans(self, units: list[Unit]) -> list[ChunkSpan]:
        """Pack units into chunks with soft/hard token budgets and paragraph-only overlap.

        Chunks are built greedily from `Unit`s (paragraphs and code fences). We never
        split inside a unit, so code fences remain intact. By default, heading changes
        close the current chunk so a chunk only owns source units from one Markdown
        section. If `split_heading_level` is set, only changes to that exact heading
        section close chunks; ancestor headings remain breadcrumb context. When
        starting a new chunk, we optionally prepend the last paragraph from the
        previous chunk in the same section (if it fits the `overlap_tokens` budget) to
        preserve continuity. Code is never overlapped.

        Cutting rules:

        1) If adding a unit would exceed `max_tokens`: close current chunk.
        2) If already >= `target_tokens` and next unit is a paragraph: close chunk (soft cut).
        3) If the next unit enters a different configured heading section: close chunk.
        4) Otherwise, keep appending.

        The chunk's `heading_path` is taken from the non-overlap source units it owns.

        Args:
            units: Units from `parse_units()`.

        Returns:
            A list of `ChunkSpan` objects in order. `start_unit` and `end_unit`
            cover the non-overlap source units owned by the chunk.
        """
        chunks: list[ChunkSpan] = []
        buf: list[str] = []
        buf_units: list[Unit] = []
        buf_unit_indices: list[int] = []
        buf_tokens = 0
        last_para_for_overlap: tuple[str, list[tuple[int, str]]] | None = None

        def section_key(heading_path: list[tuple[int, str]]):
            if self.split_heading_level is None:
                return tuple(heading_path)
            path: list[tuple[int, str]] = []
            for level, title in heading_path:
                path.append((level, title))
                if level == self.split_heading_level:
                    return tuple(path)
            return None

        def should_split(
            previous_heading_path: list[tuple[int, str]],
            next_heading_path: list[tuple[int, str]],
        ) -> bool:
            previous_key = section_key(previous_heading_path)
            next_key = section_key(next_heading_path)
            if previous_key == next_key:
                return False
            if previous_key is None and next_key is not None:
                return False
            return True

        def close_chunk():
            nonlocal buf, buf_units, buf_unit_indices, buf_tokens, last_para_for_overlap
            if not buf:
                return
            owned_units = [
                unit for unit, index in zip(buf_units, buf_unit_indices, strict=False) if index >= 0
            ]
            heading_path = owned_units[-1].heading_path if owned_units else []
            body = "\n\n".join(buf).strip()
            last_para_for_overlap = None
            for u in reversed(owned_units):
                if u.kind == "para":
                    last_para_for_overlap = (u.text, u.heading_path)
                    break
            if buf_unit_indices:
                owned_indices = [index for index in buf_unit_indices if index >= 0]
                start_unit = min(owned_indices)
                end_unit = max(owned_indices) + 1
            else:
                start_unit = end_unit = 0
            chunks.append(
                ChunkSpan(
                    text=body,
                    heading_path=heading_path,
                    start_unit=start_unit,
                    end_unit=end_unit,
                )
            )
            buf, buf_units, buf_unit_indices, buf_tokens = [], [], [], 0

        def add_overlap_if_allowed(heading_path: list[tuple[int, str]]) -> None:
            nonlocal buf_tokens
            if not last_para_for_overlap:
                return
            overlap_text, overlap_heading_path = last_para_for_overlap
            if section_key(overlap_heading_path) != section_key(heading_path):
                return
            ov = count_tokens(overlap_text)
            if ov <= self.overlap:
                buf.append(overlap_text)
                buf_units.append(
                    Unit(
                        kind="para",
                        text=overlap_text,
                        lang=None,
                        heading_path=overlap_heading_path,
                    )
                )
                buf_unit_indices.append(-1)
                buf_tokens += ov

        for unit_index, u in enumerate(units):
            u_tokens = count_tokens(u.text)

            # Prepend last para if next unit is para as well (paragraph-only overlap)
            if not buf:
                if u.kind == "para":
                    add_overlap_if_allowed(u.heading_path)
                buf.append(u.text)
                buf_units.append(u)
                buf_unit_indices.append(unit_index)
                buf_tokens += u_tokens
                continue

            if buf_units and should_split(buf_units[-1].heading_path, u.heading_path):
                close_chunk()
                if u.kind == "para":
                    add_overlap_if_allowed(u.heading_path)
                buf.append(u.text)
                buf_units.append(u)
                buf_unit_indices.append(unit_index)
                buf_tokens += u_tokens
                continue

            if buf_tokens + u_tokens > self.max_tokens:
                close_chunk()
                if u.kind == "para":
                    add_overlap_if_allowed(u.heading_path)
                buf.append(u.text)
                buf_units.append(u)
                buf_unit_indices.append(unit_index)
                buf_tokens += u_tokens
                continue

            if buf_tokens + u_tokens > self.target and u.kind == "para":
                close_chunk()
                add_overlap_if_allowed(u.heading_path)
                buf.append(u.text)
                buf_units.append(u)
                buf_unit_indices.append(unit_index)
                buf_tokens += u_tokens
                continue

            buf.append(u.text)
            buf_units.append(u)
            buf_unit_indices.append(unit_index)
            buf_tokens += u_tokens

        close_chunk()
        return chunks

    def chunk(self, units: list[Unit]) -> list[tuple[str, list[tuple[int, str]]]]:
        """Pack units into chunks and return the legacy tuple shape."""
        return [(chunk.text, chunk.heading_path) for chunk in self.chunk_spans(units)]
