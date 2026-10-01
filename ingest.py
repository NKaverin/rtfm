import re
from dataclasses import dataclass
from pathlib import Path

BOOK_DIR = Path("data/progit2/book")
INDEX_TERM = re.compile(r"\(\(\(.*?\)\)\)")

@dataclass
class Section:
    source: str
    heading: str
    text: str

def book_files(book_dir: Path) -> list[Path]:
    return sorted(book_dir.glob("*/sections/*.asc"))

def _heading(line: str) -> str | None:
    title = line.lstrip("=")
    if title == line or not title.startswith(" "):
        return None
    return title.strip()

def split_sections(text: str, source: str) -> list[Section]:
    sections = []
    heading = ""
    body = []
    for line in text.splitlines():
        title = _heading(line)
        if title is not None:
            sections.append(Section(source, heading, "\n".join(body)))
            heading = line.lstrip("=").strip()
            body = []
        else:
            body.append(line)

    sections.append(Section(source, heading, "\n".join(body)))
    return sections

def clean_text(text: str) -> str:
    kept = []
    in_code = False
    for line in text.splitlines():
        if line == "----":
            in_code = not in_code
            continue
        if in_code:
            kept.append(line)
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        if line.startswith("image::") or (line and line.strip("=") == ""):
            continue
        kept.append(INDEX_TERM.sub("", line))
    return "\n".join(kept).strip()

def read_book(book_dir: Path) -> list[Section]:
    sections = []
    for path in book_files(book_dir):
        source = path.relative_to(book_dir).as_posix()
        for section in split_sections(path.read_text(encoding="utf-8"), source):
            section.text = clean_text(section.text)
            if section.text:
                sections.append(section)
    return sections

def main():
    sections = read_book(BOOK_DIR)
    print(len(sections), "sections from", len({s.source for s in sections}), "files")
    words = sorted(len(s.text.split()) for s in sections)
    print("words per section: min", words[0], "| median", words[len(words) // 2], "| max", words[-1])
    print("sections over 400 words:", sum(1 for w in words if w > 400))
    print("sections with no heading:", sum(1 for s in sections if s.heading == ""))
    longest = max(sections, key=lambda s: len(s.text.split()))
    print("longest:", longest.source, "|", longest.heading, "|", len(longest.text.split()), "words")
    print(sections[0])

if __name__ == "__main__":
    main()