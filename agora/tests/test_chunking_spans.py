from __future__ import annotations

from agora.chunking import Chunk, MarkdownChunker
from agora.cli.ingest import _chunk_metadata, _embedding_text


def test_chunk_spans_preserve_legacy_chunk_shape_and_unit_ranges():
    chunker = MarkdownChunker(target_tokens=8, overlap_tokens=4, max_tokens=12)
    units = chunker.parse_units(
        "# Title\n\n"
        "First paragraph has a few words.\n\n"
        "Second paragraph has a few words.\n\n"
        "Third paragraph has a few words.\n"
    )

    spans = chunker.chunk_spans(units)

    assert chunker.chunk(units) == [(span.text, span.heading_path) for span in spans]
    assert spans[0].start_unit == 0
    assert spans[0].end_unit > spans[0].start_unit
    assert spans[-1].end_unit == len(units)


def test_chunk_spans_do_not_cross_h2_or_h3_boundaries():
    chunker = MarkdownChunker(target_tokens=200, overlap_tokens=20, max_tokens=300)
    units = chunker.parse_units(
        "# Doc\n\n"
        "Intro paragraph.\n\n"
        "## First\n\n"
        "First section paragraph.\n\n"
        "### Detail\n\n"
        "Nested detail paragraph.\n\n"
        "## Second\n\n"
        "Second section paragraph.\n"
    )

    spans = chunker.chunk_spans(units)

    assert [span.heading_path[-1][1] for span in spans] == [
        "Doc",
        "First",
        "Detail",
        "Second",
    ]
    assert "Nested detail paragraph" not in spans[1].text
    assert "Second section paragraph" not in spans[2].text


def test_overlap_stays_inside_the_same_heading_path():
    chunker = MarkdownChunker(target_tokens=4, overlap_tokens=10, max_tokens=20)
    units = chunker.parse_units(
        "# Doc\n\n"
        "## First\n\n"
        "Alpha one two three.\n\n"
        "Alpha four five six.\n\n"
        "## Second\n\n"
        "Beta one two three.\n"
    )

    spans = chunker.chunk_spans(units)
    second_span = next(span for span in spans if span.heading_path[-1][1] == "Second")

    assert "Alpha" not in second_span.text
    assert second_span.start_unit == 2


def test_metadata_section_matches_owned_chunk_content_after_heading_boundary():
    chunker = MarkdownChunker(target_tokens=200, overlap_tokens=20, max_tokens=300)
    units = chunker.parse_units(
        "# Doc\n\n## First\n\nFirst owned paragraph.\n\n## Second\n\nSecond owned paragraph.\n"
    )

    spans = chunker.chunk_spans(units)
    metas = [
        _chunk_metadata(
            {"doc_title": "Doc", "source_url": "https://example.org/doc"},
            span.text,
            span.heading_path,
            idx,
        )
        for idx, span in enumerate(spans)
    ]

    assert metas[0]["section"] == "First"
    assert "Second owned paragraph" not in spans[0].text
    assert metas[1]["section"] == "Second"
    assert "First owned paragraph" not in spans[1].text


def test_split_heading_level_uses_exact_level_and_keeps_ancestors_as_context():
    chunker = MarkdownChunker(
        target_tokens=200,
        overlap_tokens=20,
        max_tokens=300,
        split_heading_level=3,
    )
    units = chunker.parse_units(
        "# H1 title\n\n"
        "Intro text.\n\n"
        "## H2 Chapter\n\n"
        "Chapter text.\n\n"
        "### H3 Section 1\n\n"
        "Section one text.\n\n"
        "### H3 Section 2\n\n"
        "Section two text.\n"
    )

    spans = chunker.chunk_spans(units)
    metas = [
        _chunk_metadata(
            {"doc_title": "Doc", "source_url": "https://example.org/doc"},
            span.text,
            span.heading_path,
            idx,
        )
        for idx, span in enumerate(spans)
    ]

    assert len(spans) == 2
    assert "Intro text" in spans[0].text
    assert "Chapter text" in spans[0].text
    assert "Section one text" in spans[0].text
    assert "Section two text" not in spans[0].text
    assert spans[0].heading_path == [
        (1, "H1 title"),
        (2, "H2 Chapter"),
        (3, "H3 Section 1"),
    ]
    assert spans[1].text == "Section two text."
    assert metas[1]["breadcrumbs"] == ["H1 title", "H2 Chapter", "H3 Section 2"]
    assert _embedding_text(Chunk(id="c2", text=spans[1].text, metadata=metas[1])).startswith(
        "H1 title > H2 Chapter > H3 Section 2\n\n"
    )


def test_split_heading_level_overlap_does_not_cross_exact_level_sections():
    chunker = MarkdownChunker(
        target_tokens=200,
        overlap_tokens=20,
        max_tokens=300,
        split_heading_level=3,
    )
    units = chunker.parse_units(
        "# Doc\n\n"
        "## Chapter\n\n"
        "### First\n\n"
        "Alpha one two three.\n\n"
        "### Second\n\n"
        "Beta one two three.\n"
    )

    spans = chunker.chunk_spans(units)

    assert len(spans) == 2
    assert "Alpha" not in spans[1].text


def test_chunker_can_disable_heading_boundaries_for_parent_chunks():
    chunker = MarkdownChunker(
        target_tokens=200,
        overlap_tokens=20,
        max_tokens=300,
        split_headings=False,
    )
    units = chunker.parse_units(
        "# Doc\n\nIntro.\n\n## First\n\nFirst text.\n\n## Second\n\nSecond text.\n"
    )

    spans = chunker.chunk_spans(units)

    assert len(spans) == 1
    assert "First text" in spans[0].text
    assert "Second text" in spans[0].text
    assert spans[0].heading_path == [(1, "Doc"), (2, "Second")]
