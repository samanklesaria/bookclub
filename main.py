import sys
import os
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Iterator
import chromadb
import ollama
import itertools
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QListWidget, QListWidgetItem, QProgressBar,
    QLabel
)
from PyQt6.QtGui import QFont

def convert_to_markdown(book_path: str) -> str:
    """Convert ebook to markdown using Calibre's ebook-convert."""
    md_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
    md_path = md_file.name
    md_file.close()

    subprocess.run([
        'ebook-convert',
        book_path,
        md_path,
        '--markdown-extensions=extra'
    ], check=True, capture_output=True)

    return md_path

def extract_chapter_from_line(line: str) -> str:
    """Extract chapter title from markdown header or return None."""
    if line.startswith('#'):
        return line.lstrip('#').strip()
    return None

def stream_paragraphs(md_path: str) -> Iterator[Tuple[str, str]]:
    """Stream paragraphs from markdown file with their chapter."""
    current_chapter = "Introduction"
    current_para = []

    with open(md_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            chapter = extract_chapter_from_line(line)
            if chapter:
                current_chapter = chapter
            elif line:
                current_para.append(line)
            elif current_para:
                text = ' '.join(current_para).strip()
                if len(text) >= 20:
                    yield (current_chapter, text)
                current_para = []

        if current_para:
            text = ' '.join(current_para).strip()
            yield (current_chapter, text)

def embed_text(text: list[str]) -> list[list[float]]:
    """Generate embedding for text using Ollama."""
    return ollama.embed(model="nomic-embed-text", input=text)["embeddings"]

def index_book(book_path: str, progress_callback) -> str:
    """Index a book and return the database path."""
    progress_callback("Converting book to markdown...")
    md_path = convert_to_markdown(book_path)

    db_name = Path(book_path).stem + "_db"
    db_path = str(Path(book_path).parent / db_name)

    progress_callback(f"Creating database at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    collection_name = Path(book_path).stem.replace(" ", "_").replace("-", "_")
    collection = client.create_collection(name=collection_name)
    ids = itertools.count()

    for batch in itertools.batched(stream_paragraphs(md_path), 256):
        chapter, text = unzip(*batch)
        embeddings = embed_text(text)
        progress_callback(f"Processing: {chapter[0]}")

        collection.add(
            ids=map(str, itertools.islice(ids, len(chapter))),
            embeddings=embeddings,
            documents=text,
            metadatas=[{"chapter": c} for c in chapter])
    os.unlink(md_path)
    return db_path

def search_collection(collection, query: str, n_results: int = 10) -> List[Tuple[str, str, float]]:
    """Search collection and return (chapter, text, similarity) tuples."""
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results)
    return [
        (meta['chapter'], doc, dist)
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )
    ]

def get_chapter_docs(collection) -> dict[str, list[str]]:
    """Get all documents grouped by chapter."""
    all_docs = collection.get(include=['documents', 'metadatas'])
    chapters = {}

    for doc, meta in zip(all_docs['documents'], all_docs['metadatas']):
        chapter = meta.get('chapter', 'Unknown')
        chapters.setdefault(chapter, []).append(doc)

    return chapters

def summarize_text(text: str) -> str:
    """Generate summary using Ollama."""
    response = ollama.generate(
        model='qwen2.5:3b',
        prompt=f"Summarize the key points from this chapter in 3-4 concise bullet points:\n\n{text[:4000]}"
    )
    return response['response']

def create_search_window(db_path: str):
    """Create and return the main search window."""
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    if not collections:
        raise ValueError("No collections found in database")
    collection = collections[0]

    class SearchWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Book Search")
            self.setGeometry(100, 100, 900, 700)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            self.status = QLabel(f"Loaded: {collection.name} ({collection.count()} passages)")
            layout.addWidget(self.status)

            search_layout = QHBoxLayout()
            self.search_input = QLineEdit()
            self.search_input.setPlaceholderText("Enter search query")
            self.search_input.returnPressed.connect(self.perform_search)
            search_layout.addWidget(self.search_input)

            search_btn = QPushButton("Search")
            search_btn.clicked.connect(self.perform_search)
            search_layout.addWidget(search_btn)

            layout.addLayout(search_layout)

            self.results = QListWidget()
            self.results.setWordWrap(True)
            layout.addWidget(self.results)

            self.show_summaries()

        def show_summaries(self):
            self.results.clear()
            self.status.setText("Generating chapter summaries...")
            QApplication.processEvents()

            chapters = get_chapter_docs(collection)

            for chapter, docs in sorted(chapters.items()):
                combined = " ".join(docs[:10])
                summary = summarize_text(combined)

                item = QListWidgetItem(f"📖 {chapter}\n\n{summary}")
                item.setFont(QFont("Arial", 10))
                self.results.addItem(item)

            self.status.setText(f"Showing summaries for {len(chapters)} chapters")

        def perform_search(self):
            query = self.search_input.text().strip()

            if not query:
                self.show_summaries()
                return

            self.status.setText("Searching...")
            QApplication.processEvents()

            results = search_collection(collection, query)
            self.results.clear()

            grouped = {}
            for chapter, text, similarity in results:
                grouped.setdefault(chapter, []).append((text, similarity))

            for chapter, passages in sorted(grouped.items()):
                header = QListWidgetItem(f"📖 {chapter}")
                header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
                self.results.addItem(header)

                for text, similarity in passages:
                    truncated = text[:300] + ('...' if len(text) > 300 else '')
                    item = QListWidgetItem(f"  [{similarity:.2%}] {truncated}")
                    item.setFont(QFont("Arial", 9))
                    self.results.addItem(item)

            self.status.setText(f"Found {len(results)} results")

    return SearchWindow()

def create_indexing_window(book_path: str):
    """Create and return the indexing window."""
    class IndexingWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Indexing Book")
            self.setGeometry(100, 100, 600, 150)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            self.status = QLabel("Starting indexing...")
            layout.addWidget(self.status)

            self.show()
            QApplication.processEvents()

            try:
                db_path = index_book(book_path, self.update_status)
                self.status.setText("Complete! Opening search interface...")
                QApplication.processEvents()

                self.search_window = create_search_window(db_path)
                self.search_window.show()
                self.close()
            except Exception as e:
                self.status.setText(f"Error: {str(e)}")

        def update_status(self, msg: str):
            self.status.setText(msg)
            QApplication.processEvents()

    return IndexingWindow()

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <epub/mobi file or database directory>")
        sys.exit(1)

    path = Path(sys.argv[1])
    app = QApplication(sys.argv)

    if path.is_file() and path.suffix.lower() in ['.epub', '.mobi']:
        window = create_indexing_window(str(path))
    elif path.is_dir():
        window = create_search_window(str(path))
    else:
        print(f"Error: {path} is not a valid epub/mobi file or database directory")
        sys.exit(1)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
