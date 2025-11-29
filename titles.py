import zipfile
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import itertools

def iter_chapter_paragraphs(epub_path):
    return itertools.islice(iter_chapter_paragraphs_(epub_path), 300)

def iter_chapter_paragraphs_(epub_path):
    """
    Yields (chapter_name, paragraph_text) tuples from an EPUB file.
    Uses toc.ncx for chapter ordering and names.
    """
    with zipfile.ZipFile(epub_path, "r") as z:

        # 1. Find toc.ncx
        toc_path = None
        for name in z.namelist():
            if name.lower().endswith("toc.ncx"):
                toc_path = name
                break
        if toc_path is None:
            raise FileNotFoundError("toc.ncx not found in EPUB.")

        toc_data = z.read(toc_path)
        root = ET.fromstring(toc_data)

        ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}

        # 2. Extract chapter list from navMap
        chapters = []  # list of (title, href)
        toc_dir = toc_path.rpartition("/")[0]

        for navPoint in root.findall(".//ncx:navPoint", ns):
            label = navPoint.find("ncx:navLabel/ncx:text", ns)
            content = navPoint.find("ncx:content", ns)
            if label is None or content is None:
                continue

            title = label.text.strip()
            href = content.attrib["src"]

            # Resolve relative href to full path
            if toc_dir and not href.startswith(toc_dir):
                full_path = f"{toc_dir}/{href}"
            else:
                full_path = href

            chapters.append((title, full_path))

        # 3. For each chapter, extract <p> paragraphs
        for chapter_title, chapter_file in chapters:
            if chapter_file not in z.namelist():
                continue  # skip missing files

            html = z.read(chapter_file)
            soup = BeautifulSoup(html, "html.parser")

            # Get all paragraph tags
            for p in soup.find_all("p"):
                text = p.get_text(strip=True, separator=" ")
                if text:
                    yield (chapter_title, text)
