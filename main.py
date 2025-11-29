import sys
import subprocess
from pathlib import Path
from typing import Iterator
import itertools

import chromadb
import ollama
import titles
from summarize import summarize
import json
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QListWidget, QListWidgetItem, QProgressBar,
    QLabel)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

def embed_text(text: list[str]) -> list[list[float]]:
    """Generate embedding for text using Ollama."""
    return ollama.embed(model="nomic-embed-text", input=text)["embeddings"]

def index_book(book_path: str, progress_callback) -> str:
    """Index a book and return the database path."""
    progress_callback("Reading book...")

    db_name = Path(book_path).stem + "_db"
    db_path = str(Path(book_path).parent / db_name)
    if os.path.exists(db_path):
        progress_callback(f"Database already exists at {db_path}")
        return db_path

    progress_callback(f"Creating database at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    collection_name = Path(book_path).stem.replace(" ", "_").replace("-", "_")
    collection = client.create_collection(name=collection_name, get_or_create=True)
    ids = itertools.count()

    for batch in itertools.batched(titles.iter_chapter_paragraphs(book_path), 256):
        chapters, texts = zip(*batch)
        embeddings = embed_text(texts)
        progress_callback(f"Indexing: {chapters[0]}")
        collection.add(
            ids=[str(next(ids)) for _ in chapters],
            embeddings=embeddings,
            documents=list(texts),
            metadatas=[{"chapter": c} for c in chapters])

    summaries = []
    for chapter, pairs in itertools.islice(itertools.groupby(titles.iter_chapter_paragraphs(book_path), lambda x: x[0]), 1, None):
        progress_callback(f"Summarizing {chapter}")
        summaries.append((chapter, summarize('\n'.join([p[1] for p in pairs]))))

    collection.modify(metadata={'summaries': json.dumps(summaries), 'book_path': book_path})
    return db_path

def search_collection(collection, query: str, n_results: int = 10) -> list[tuple[str, str, float]]:
    """Search collection and return (chapter, text, similarity) tuples."""
    query_embedding = embed_text([query])
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
            self.results.itemDoubleClicked.connect(self.open_in_viewer)
            layout.addWidget(self.results)
            self.book_path = collection.metadata.get('book_path', '')
            self.show_summaries()

        def show_summaries(self):
            self.results.clear()
            summaries = json.loads(collection.metadata['summaries'])
            for chapter_name, summary in summaries:
                header = QListWidgetItem(chapter_name)
                header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
                header.setFlags(header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self.results.addItem(header)
                for s in summary:
                    summary_item = QListWidgetItem(s.replace("\n", ""))
                    summary_item.setFont(QFont("Arial", 10))
                    summary_item.setFlags(summary_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                    self.results.addItem(summary_item)

        def perform_search(self):
            query = self.search_input.text().strip()

            if not query:
                self.show_summaries()
                return

            results = search_collection(collection, query)
            self.results.clear()

            grouped = {}
            for chapter, text, similarity in results:
                grouped.setdefault(chapter, []).append((text, similarity))

            for chapter, passages in sorted(grouped.items()):
                header = QListWidgetItem(f"📖 {chapter}")
                header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
                header.setFlags(header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self.results.addItem(header)

                for text, similarity in passages:
                    truncated = text[:300] + ('...' if len(text) > 300 else '')
                    item = QListWidgetItem(f"  [{similarity:.2%}] {truncated}")
                    item.setFont(QFont("Arial", 12))
                    item.setData(Qt.ItemDataRole.UserRole, text[:150])
                    self.results.addItem(item)

            self.status.setText(f"Found {len(results)} results")

        def open_in_viewer(self, item: QListWidgetItem):
            if self.search_input.text().strip():
                text = item.data(Qt.ItemDataRole.UserRole)
                subprocess.Popen(['/Applications/calibre.app/Contents/MacOS/ebook-viewer', self.book_path, '--open-at', f'search: {text}'])

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

            db_path = index_book(book_path, self.update_status)
            self.status.setText("Complete! Opening search interface...")
            QApplication.processEvents()

            self.search_window = create_search_window(db_path)
            self.search_window.show()
            self.close()

        def update_status(self, msg: str):
            self.status.setText(msg)
            QApplication.processEvents()

    return IndexingWindow()

def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python main.py <epub/mobi file or database directory>")
    #     sys.exit(1)

    path = Path("color_of_law.epub") # Path(sys.argv[1])
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
