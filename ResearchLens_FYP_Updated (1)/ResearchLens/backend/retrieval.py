"""Intent-aware local retrieval for conversational document questions."""
import re
from backend.vector_store import search, representative_passages, document_catalog, page_passages

_OVERVIEW = re.compile(r'\b(summar\w*|overview|abstract|what is this|what.*(?:paper|pdf|document|report).*about|what.*(?:paper|pdf|document|report)\s*(?:is|means)|(?:ye|yeh|is|iss)\s+(?:paper|pdf|document|report).*(?:kia|kya|kis)|kis.*(?:cheez|chees|bare|baar))\b', re.I)
_METHOD = re.compile(r'\b(method\w*|approach|pipeline|how.*(?:work|built|train)|algorithm|model.*(?:use|used)|technique)\b', re.I)
_LIMITS = re.compile(r'\b(limit\w*|weakness\w*|caveat\w*|drawback\w*|reliable|reliability|constraint\w*)\b', re.I)


def intent(query):
    if _METHOD.search(query):
        return 'methodology'
    if _LIMITS.search(query):
        return 'limitations'
    if _OVERVIEW.search(query.strip()) or query.strip().lower() in {'what is this?','what is this','explain this','explain it','ye paper kia ha?','ye paper kia ha ?'}:
        return 'overview'
    return 'specific'


def requested_pages(query, history=None):
    match = re.search(r'\bpages?\s+(\d+(?:\s*(?:-|–|to|and|,)\s*\d+)*)',query,re.I)
    if match:
        phrase=match.group(1)
        numbers=list(map(int,re.findall(r'\d+',phrase)))
        if re.search(r'-|–|\bto\b',phrase):
            if len(numbers)!=2 or numbers[1]<numbers[0]:
                raise ValueError('Use an ascending page range, such as pages 2–4.')
            if numbers[1]-numbers[0]>=7:
                raise ValueError('Please ask about at most seven pages at a time.')
            numbers=list(range(numbers[0],numbers[1]+1))
        if not all(n>0 for n in numbers) or len(set(numbers))>7:
            raise ValueError('Use PDF page numbers starting at 1, at most seven pages at a time.')
        return sorted(set(numbers))
    relative=re.search(r'\b(next|previous) page\b',query,re.I)
    if relative:
        for turn in reversed(history or []):
            previous=turn.get('requested_pages') or requested_pages(turn['question'])
            if previous:
                page=(max(previous)+1) if relative[1].lower()=='next' else min(previous)-1
                if page<1:raise ValueError('You are already at the first page.')
                return [page]
        raise ValueError('Specify a page first, for example: Explain page 2.')
    return []


def retrieve_for_chat(query, session_id, history=None, source=None):
    catalog=document_catalog(session_id)
    if source is not None:
        catalog=[item for item in catalog if item['source']==source]
        if not catalog:raise ValueError('The selected document is unavailable in this chat. Select an attached document.')
    pages=requested_pages(query,history)
    if pages and source is None and len(catalog)>1:
        raise ValueError('Select one document in Answer using before asking about page numbers.')
    kind = intent(query)
    if pages:
        docs=page_passages(session_id,pages,source)
        found={d.metadata.get('page',-1)+1 for d in docs}
        missing=sorted(set(pages)-found)
        notices=['Page numbers refer to physical PDF pages, starting at 1. Only selected page excerpts are provided.']
        if missing:notices.append('No readable indexed text was found for page(s): '+', '.join(map(str,missing))+'. Do not substitute other pages or invent their contents.')
        return dict(intent='page question',docs=docs,titles=[item['title'] for item in catalog],keyword_match=bool(docs),notices=notices,pages=pages)

    expansions = {
        'overview':'overview purpose objective title model analysis findings interpretation limitations',
        'methodology':'methodology method model classifier features trained training algorithm logistic regression threshold input audio visual',
        'limitations':'limitations limitation interpretation uncalibrated not established diagnostic confidence uncertainty scope',
    }
    search_query = expansions.get(kind, query)
    if kind == 'specific' and history and re.search(r'\b(it|its|this|that|they|their)\b', query, re.I):
        search_query += ' ' + history[-1]['question'][:250]
    matched = search(search_query, session_id, k=5, source=source)
    representatives = representative_passages(session_id, k=6, source=source)
    # Broad questions need document context even when the exact heading is absent.
    candidates = (representatives[:2] + matched + representatives[2:]) if kind == 'overview' else (matched + representatives)
    docs, seen = [], set()
    for doc in candidates:
        key = (doc.metadata.get('source'), doc.metadata.get('page'), doc.page_content)
        if key not in seen:
            seen.add(key)
            docs.append(doc)
        if len(docs) >= 7:
            break
    return {'intent':kind, 'docs':docs, 'titles':[item['title'] for item in catalog], 'keyword_match':bool(matched), 'notices':['Retrieved sections are a limited sample, not an exhaustive review of every document.']}
