from ingest import Section, book_files, split_sections, clean_text, read_book

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