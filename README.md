# Book Club Search Tool

A PyQt6 application for indexing and semantically searching through ebooks using Ollama embeddings and ChromaDB.

## Features

- Convert EPUB/MOBI books to searchable databases using Calibre
- Semantic search using natural language queries
- Results grouped by chapter with similarity scores
- Automatic chapter summaries when no search is active
- Persistent ChromaDB storage for fast repeated searches
- Streaming processing - never loads entire book into memory

## Requirements

- Python 3.13+
- Calibre (for `ebook-convert` command)
- Ollama installed and running locally
- Required Ollama models:
  - `nomic-embed-text` (for embeddings)
  - `qwen2.5:3b` (for summarization)

## Usage

### Index a new book

```bash
python main.py /path/to/book.epub
```

This will:
1. Convert the ebook to markdown using Calibre
2. Stream paragraphs from the markdown file
3. Create a database named `book_db` in the same directory as the ebook
4. Generate embeddings for each paragraph
5. Open the search interface when complete

### Search an existing database

```bash
python main.py /path/to/book_db
```

### Using the interface

- **Empty search bar**: Shows chapter summaries automatically
- **Enter a query**: Performs semantic search and shows relevant passages grouped by chapter
- Results show similarity scores as percentages
- Passages are truncated to 300 characters in the list view

## Example

```bash
# Index a book
python main.py ~/Books/pride_and_prejudice.epub

# Later, search the indexed book
python main.py ~/Books/pride_and_prejudice_db
```

## Notes

- Only EPUB is supported
- The database is stored persistently and can be reused
- Indexing time depends on book length (typically 1-5 minutes)
