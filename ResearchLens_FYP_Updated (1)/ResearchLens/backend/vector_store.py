"""Small local SQLite FTS index. No paid embeddings or model downloads."""
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from langchain_core.documents import Document
from backend.config import DATA_DIR

DB_PATH = DATA_DIR / 'papers_fts.db'
_STOP = {'a','an','and','are','as','at','be','by','can','did','do','does','explain','for','from','give','how','i','in','is','it','its','me','of','on','or','paper','please','report','research','show','summarize','summary','that','the','their','these','this','to','uploaded','us','was','were','what','which','with','you','your'}


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            conn.execute('CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, chunk_key TEXT UNIQUE, session TEXT, title TEXT, source TEXT, position INTEGER, text TEXT, metadata TEXT)')
            conn.execute('CREATE INDEX IF NOT EXISTS chunks_session ON chunks(session)')
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(title, text, content='chunks', content_rowid='id', tokenize='unicode61')")
            conn.executescript('''
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
              INSERT INTO chunks_fts(rowid,title,text) VALUES(new.id,new.title,new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
              INSERT INTO chunks_fts(chunks_fts,rowid,title,text) VALUES('delete',old.id,old.title,old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
              INSERT INTO chunks_fts(chunks_fts,rowid,title,text) VALUES('delete',old.id,old.title,old.text);
              INSERT INTO chunks_fts(rowid,title,text) VALUES(new.id,new.title,new.text);
            END;''')
            yield conn
    finally:
        conn.close()


def add_paper(docs, session_id):
    if not docs:
        raise ValueError('No readable text found. Scanned PDFs require OCR before upload.')
    with _connect() as c:
        for position, doc in enumerate(docs):
            m = doc.metadata
            source = m.get('url') or m.get('source', '')
            key = hashlib.sha256(f"{session_id}|{source}|{m.get('page', '')}|{m.get('start_index', '')}|{doc.page_content}".encode()).hexdigest()
            c.execute('''INSERT INTO chunks(chunk_key,session,title,source,position,text,metadata) VALUES(?,?,?,?,?,?,?)
                         ON CONFLICT(chunk_key) DO UPDATE SET title=excluded.title,metadata=excluded.metadata''',
                      (key, session_id, m.get('title', 'Uploaded paper'), source, position, doc.page_content, json.dumps(m)))


def list_papers(session_id):
    with _connect() as c:
        return [r[0] for r in c.execute('SELECT DISTINCT title FROM chunks WHERE session=? ORDER BY title', (session_id,))]


def _docs(rows):
    return [Document(page_content=r['text'], metadata=json.loads(r['metadata'])) for r in rows]


def first_passages(session_id, k=4):
    with _connect() as c:
        rows = c.execute('SELECT * FROM chunks WHERE session=? ORDER BY source,position,id LIMIT ?', (session_id, max(1, min(k, 10)))).fetchall()
    return _docs(rows)


def search(query, session_id, k=4, source=None):
    # Quote every token; user text can never become FTS operators or SQL.
    words = list(dict.fromkeys(w.lower() for w in re.findall(r'[^\W_]+', str(query), re.UNICODE) if w.lower() not in _STOP))[:32]
    if not words:
        return []
    expression = ' OR '.join('"' + w + '"' for w in words)
    with _connect() as c:
        rows = c.execute('''SELECT c.* FROM chunks_fts JOIN chunks c ON chunks_fts.rowid=c.id
                            WHERE chunks_fts MATCH ? AND c.session=? AND (? IS NULL OR c.source=?)
                            ORDER BY bm25(chunks_fts,2.0,1.0),c.id LIMIT ?''',
                         (expression, session_id, source, source, max(1, min(k, 10)))).fetchall()
    return _docs(rows)


def delete_index(session_id):
    """Internal evaluation cleanup; UI access must validate ownership first."""
    with _connect() as c:
        c.execute('DELETE FROM chunks WHERE session=?', (session_id,))


def representative_passages(session_id, k=6, source=None):
    """Sample opening, middle and closing sections, interleaved across sources."""
    with _connect() as c:
        rows = c.execute('SELECT * FROM chunks WHERE session=? AND (? IS NULL OR source=?) ORDER BY source,position,id', (session_id,source,source)).fetchall()
    grouped = {}
    for row in rows:
        grouped.setdefault(row['source'], []).append(row)
    choices = []
    for group in grouped.values():
        indices = list(dict.fromkeys([0, len(group)-1, 1, len(group)//3, 2*len(group)//3]))
        choices.append([group[i] for i in indices if 0 <= i < len(group)])
    selected = []
    for level in range(5):
        for group in choices:
            if level < len(group):
                selected.append(group[level])
                if len(selected) >= k:
                    return _docs(selected)
    return _docs(selected)


def document_catalog(session_id):
    """Stable source identities and physical PDF page counts, scoped to a chat."""
    with _connect() as c:
        rows = c.execute('SELECT source,title,metadata FROM chunks WHERE session=? ORDER BY title,source', (session_id,)).fetchall()
    catalog = {}
    for row in rows:
        meta = json.loads(row['metadata'])
        entry = catalog.setdefault(row['source'], dict(source=row['source'],title=row['title'],page_count=0))
        entry['page_count'] = max(entry['page_count'],meta.get('page_count',0),meta.get('page',-1)+1)
    return list(catalog.values())


def page_passages(session_id, pages, source=None, k=7):
    """Round-robin chunks so requested pages receive context before extra chunks."""
    with _connect() as c:
        rows = c.execute('SELECT * FROM chunks WHERE session=? AND (? IS NULL OR source=?) ORDER BY source,position,id', (session_id,source,source)).fetchall()
    groups = {}
    for row in rows:
        page = json.loads(row['metadata']).get('page',-1)+1
        if page in pages:
            groups.setdefault((row['source'],page),[]).append(row)
    selected=[]
    for level in range(k):
        for group in groups.values():
            if level < len(group):selected.append(group[level])
            if len(selected)>=k:return _docs(selected)
    return _docs(selected)
