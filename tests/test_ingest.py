import pytest

import store
from ingest import (
    Chunk,
    Section,
    book_files,
    chunk_text,
    clean_text,
    ingest_book,
    make_chunks,
    read_book,
    split_sections,
)
from embeddings import count_tokens

SAMPLE = """[[_undoing]]
=== Undoing Things

Some text.

[NOTE]
====
A note.
====

==== Unstaging
More text.
"""

def test_book_files_finds_section_files_in_order(tmp_path):
    (tmp_path / "02-b" / "sections").mkdir(parents=True)
    (tmp_path / "01-a" / "sections").mkdir(parents=True)
    (tmp_path / "02-b" / "sections" / "x.asc").write_text("x")
    (tmp_path / "01-a" / "sections" / "y.asc").write_text("y")
    (tmp_path / "01-a" / "notes.txt").write_text("not a section")
    found = [p.relative_to(tmp_path).as_posix() for p in book_files(tmp_path)]
    assert found == ["01-a/sections/y.asc", "02-b/sections/x.asc"]

def test_split_sectionns_uses_headings_not_block_delimiters():
    sections = split_sections(SAMPLE, "x.asc")
    assert [s.heading for s in sections] == ["", "Undoing Things", "Unstaging"]

def test_split_sections_keeps_body_under_its_heading():
    sections = split_sections(SAMPLE, "x.asc")
    assert "A note." in sections[1].text
    assert sections[2].text == "More text."
    assert sections[2].source == "x.asc"

def test_clean_text_removes_markup_lines():
    text = "[[_undoing]]\n[NOTE]\n====\n(((git commands, commit)))A note.\n====\nimage::images/x.png[X]"
    assert clean_text(text) == "A note."


def test_clean_text_keeps_code_blocks_untouched():
    text = "Your config file:\n\n[source,ini]\n----\n[imap]\n  port = 993\n----\n\nDone."
    assert clean_text(text) == "Your config file:\n\n[imap]\n  port = 993\n\nDone."

def test_read_book_cleans_sections_and_drops_empty_ones(tmp_path):
    folder = tmp_path / "02-git-basics" / "sections"
    folder.mkdir(parents=True)
    (folder / "undoing.asc").write_text(SAMPLE, encoding="utf-8")
    assert read_book(tmp_path) == [
        Section("02-git-basics/sections/undoing.asc", "Undoing Things", "Some text.\n\nA note."),
        Section("02-git-basics/sections/undoing.asc", "Unstaging", "More text."),
    ]

def test_chunk_text_packs_paragraphs_up_to_the_limit():
    a, b, c = "git add stages changes.", "git commit records them.", "git push uploads them."
    limit = count_tokens(f"{a}\n\n{b}")
    assert chunk_text(f"{a}\n\n{b}\n\n{c}", max_tokens=limit, overlap_tokens=0) == [f"{a}\n\n{b}", c]

def test_chunk_text_splits_a_long_paragraph_by_lines():
    paragraph = "\n".join(f"line number {i}" for i in range(20))
    chunks = chunk_text(paragraph, max_tokens=30, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 30 for chunk in chunks)
    assert "\n".join(chunks) == paragraph

def test_chunk_text_repeats_a_short_last_paragraph_in_the_next_chunk():
    a, b, c = "git add stages changes.", "git commit records them.", "git push uploads them."
    limit = count_tokens(f"{b}\n\n{c}")
    chunks = chunk_text(f"{a}\n\n{b}\n\n{c}", max_tokens=limit, overlap_tokens=limit)
    assert chunks == [f"{a}\n\n{b}", f"{b}\n\n{c}"]  

def test_make_chunks_numbers_positions_per_file():
    sections = [
        Section("a.asc", "One", "first"),
        Section("a.asc", "Two", "second"),
        Section("b.asc", "Three", "third"),
    ]
    assert make_chunks(sections) == [
        Chunk("a.asc#0", "a.asc", "One", 0, "first"),
        Chunk("a.asc#1", "a.asc", "Two", 1, "second"),
        Chunk("b.asc#0", "b.asc", "Three", 0, "third"),
    ]

def test_ingest_book_stores_chunks_and_removes_stale_ones(tmp_path, isolated_store):
    folder = tmp_path / "02-git-basics" / "sections"
    folder.mkdir(parents=True)
    book_file = folder / "undoing.asc"
    book_file.write_text(SAMPLE, encoding="utf-8")

    assert ingest_book(tmp_path) == {"chunks": 2, "removed": 0}
    assert store.chunk_ids() == [
        "02-git-basics/sections/undoing.asc#0",
        "02-git-basics/sections/undoing.asc#1",
    ]
    assert ingest_book(tmp_path) == {"chunks": 2, "removed": 0}

    book_file.write_text("=== Undoing Things\n\nOnly this is left.", encoding="utf-8")
    assert ingest_book(tmp_path) == {"chunks": 1, "removed": 1}
    assert store.chunk_ids() == ["02-git-basics/sections/undoing.asc#0"]


def test_ingest_book_refuses_an_empty_book(tmp_path, isolated_store):
    with pytest.raises(ValueError):
        ingest_book(tmp_path)