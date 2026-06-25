from __future__ import annotations

from agora.chunking import MarkdownChunker


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
