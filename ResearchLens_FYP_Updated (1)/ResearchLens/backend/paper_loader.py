"""Bounded document loading with stable source metadata."""
import hashlib
import ipaddress
import os
import re
import socket
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import MAX_DOCUMENT_CHARS, MAX_UPLOAD_BYTES

_splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=120, add_start_index=True)


def _chunks(docs, title, source):
    if sum(len(d.page_content) for d in docs) > MAX_DOCUMENT_CHARS:
        raise ValueError("Document is too large. Use a smaller document.")
    for d in docs:
        d.metadata.update(title=title, source=source)
    chunks = _splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No readable text found. Scanned PDFs require OCR.")
    return chunks


def load_document(source):
    path = Path(source)
    if source.startswith(("https://", "http://")):
        return load_webpage(source)
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Maximum upload size is 20 MB.")
    if path.suffix.lower() == '.pdf':
        try:
            with pymupdf.open(str(path)) as pdf:
                if pdf.needs_pass:
                    raise ValueError('This PDF is password protected. Upload an unlocked copy.')
                docs = [Document(page_content=page.get_text(), metadata={'page': i,'page_count':len(pdf)}) for i, page in enumerate(pdf)]
        except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
            raise ValueError('This PDF is empty or damaged. Open it in a PDF reader and export a readable copy.') from error
    elif path.suffix.lower() in {'.txt', '.md', '.markdown'}:
        docs = [Document(page_content=path.read_text(encoding='utf-8-sig'))]
    else:
        raise ValueError("Upload PDF, TXT or Markdown.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return _chunks(docs, path.stem, f"upload:{digest}")


def _public_https(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != 'https' or not p.hostname or p.username or p.password or p.port not in (None, 443):
        raise ValueError("Use a public HTTPS URL without credentials or a custom port.")
    addresses = socket.getaddrinfo(p.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("Private and local network URLs are blocked.")
    return url


def _fetch(url, max_bytes=MAX_UPLOAD_BYTES):
    # Validate every redirect. Keep this feature opt-in; use an egress firewall
    # for hostile multi-tenant hosting to also prevent DNS rebinding.
    with requests.Session() as client:
        client.trust_env = False
        for _ in range(5):
            _public_https(url)
            with client.get(url, stream=True, allow_redirects=False, timeout=(10, 30), headers={"User-Agent": "ResearchLens/1.0"}) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    url = urllib.parse.urljoin(url, response.headers['Location'])
                    continue
                response.raise_for_status()
                data = bytearray()
                for block in response.iter_content(65536):
                    data.extend(block)
                    if len(data) > max_bytes:
                        raise ValueError("Downloaded document exceeds the size limit.")
                return bytes(data), url, response.headers.get('Content-Type', '')
    raise ValueError("Too many redirects.")


def load_webpage(url):
    if os.getenv('ALLOW_WEB_IMPORT', 'false').lower() != 'true':
        raise ValueError("Web import is disabled. Set ALLOW_WEB_IMPORT=true on a trusted installation.")
    data, final_url, content_type = _fetch(url)
    if 'application/pdf' in content_type or data.startswith(b'%PDF'):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paper.pdf'
            path.write_bytes(data)
            docs = load_document(str(path))
        for d in docs:
            d.metadata.update(source=final_url, title=Path(urllib.parse.urlsplit(final_url).path).name or 'Web PDF')
        return docs
    soup = BeautifulSoup(data, 'html.parser')
    title = soup.title.get_text(' ', strip=True) if soup.title else final_url
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    return _chunks([Document(page_content=soup.get_text(' ', strip=True))], title, final_url)


def load_arxiv(query):
    match = re.search(r'(?<!\d)(\d{4}\.\d{4,5}(?:v\d+)?)(?!\d)', query)
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    api_query = 'id_list=' + match.group(1) if match else urllib.parse.urlencode({'search_query': f'ti:"{query.strip()}"', 'max_results': 1})
    xml, _, _ = _fetch('https://export.arxiv.org/api/query?' + api_query, 1_000_000)
    entry = ET.fromstring(xml).find('a:entry', ns)
    if entry is None:
        raise ValueError("No matching arXiv paper found.")
    entry_id = entry.findtext('a:id', '', ns)
    id_match = re.search(r'/(\d{4}\.\d{4,5}(?:v\d+)?)$', entry_id)
    if not id_match:
        raise ValueError("arXiv returned no valid paper ID. Try the exact ID.")
    arxiv_id = id_match.group(1)
    title = ' '.join(entry.findtext('a:title', arxiv_id, ns).split())
    data, _, _ = _fetch(f'https://arxiv.org/pdf/{arxiv_id}')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'paper.pdf'
        path.write_bytes(data)
        docs = load_document(str(path))
    for d in docs:
        d.metadata.update(title=title, source=f'https://arxiv.org/abs/{arxiv_id}')
    return docs
