import re
import time
from dataclasses import dataclass
from pathlib import Path

import cache
import store
from embeddings import count_tokens, embed

BOOK_DIR = Path("data/progit2/book")
INDEX_TERM = re.compile(r"\(\(\(.*?\)\)\)")
MAX_TOKENS = 500
OVERLAP_TOKENS = 100

@dataclass
class Section:
    source: str
    heading: str
    text: str

@dataclass
class Chunk:
    chunk_id: str
    source: str
    heading: str
    position: int
    text:str

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

def _pack(pieces: list[str], separator:str, max_tokens: int, overlap_tokens: int) -> list[str]:
    chunks = []
    current = []
    for piece in pieces:
        candidate = current + [piece]
        if current and count_tokens(separator.join(candidate)) > max_tokens:
            chunks.append(separator.join(current))
            last = current[-1]
            with_overlap = [last, piece]
            if (
                count_tokens(last) <= overlap_tokens
                and count_tokens(separator.join(with_overlap)) <= max_tokens
            ):
                candidate = with_overlap
            else:
                candidate = [piece]
        current = candidate
    if current:
        chunks.append(separator.join(current))
    return chunks

def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap_tokens:int = OVERLAP_TOKENS) -> list[str]:
    units = []
    for paragraph in text.split("\n\n"):
        if count_tokens(paragraph) > max_tokens:
            units.extend(_pack(paragraph.split("\n"), "\n", max_tokens, overlap_tokens))
        else:
            units.append(paragraph)
    return _pack(units, "\n\n", max_tokens, overlap_tokens)

def make_chunks(sections: list[Section]) -> list[Chunk]:
    chunks = []
    next_position = {}
    for section in sections:
        for text in chunk_text(section.text):
            position = next_position.get(section.source, 0)
            next_position[section.source] = position + 1
            chunk_id = f"{section.source}#{position}"
            chunks.append(Chunk(chunk_id, section.source, section.heading, position, text))
    return chunks

def ingest_book(book_dir: Path) -> dict:
    chunks = make_chunks(read_book(book_dir))
    if not chunks:
        raise ValueError(f"no chunks found in {book_dir}")
    store.ensure_index()
    vectors = embed([f"{c.heading}\n\n{c.text}" for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        store.save_chunk(chunk.chunk_id, chunk.text, chunk.source, chunk.heading, chunk.position, vector)
    new_ids = {c.chunk_id for c in chunks}
    stale = [chunk_id for chunk_id in store.chunk_ids() if chunk_id not in new_ids]
    for chunk_id in stale:
        store.delete_chunk(chunk_id)
    cache.flush()
    return {"chunks": len(chunks), "removed": len(stale)}  

def main():
    start = time.perf_counter()
    stats = ingest_book(BOOK_DIR)
    seconds = time.perf_counter() - start
    print(f"{stats['chunks']} chunks stored, {stats['removed']} stale removed, {seconds:.1f}s")


if __name__ == "__main__":
    main()