"""Honest, completely keyless source-excerpt mode; no generated claims."""
import re
from backend.vector_store import first_passages, search


def answer_locally(query, session_id, history=None):
    # A short follow-up can reuse keywords from the last question.
    lookup = query
    if history and re.search(r'\b(its|this|that|it|they|their)\b', query, re.I):
        lookup += ' ' + history[-1]['question'][:300]
    docs = search(lookup, session_id, k=4)
    notices = ['Local mode shows original source excerpts, not an AI-written summary. Search uses keyword matching, so try specific terms from the paper.']
    if not docs and re.search(r'\b(summary|summarize|overview|abstract)\b', query, re.I):
        docs = first_passages(session_id, k=4)
        notices.append('For this broad request, the first passages are shown; these are not an automatically generated summary.')
    if not docs:
        return {'question': query, 'answer': 'No matching passages were found. Upload a paper or search for specific words used in its text.', 'sources': [], 'notices': notices}
    sources = []
    sections = ['**Local source excerpts**\n\nThese passages are copied from your indexed documents.']
    for i, d in enumerate(docs, 1):
        m = d.metadata
        raw_source = m.get('url') or m.get('source', '')
        page = m.get('page')
        sources.append({'id':i, 'title':m.get('title','Uploaded paper'), 'page':page + 1 if isinstance(page, int) else None,
                        'url':raw_source if raw_source.startswith(('http://','https://')) else '', 'kind':'original source excerpt'})
        # Markdown blockquotes separate source data from assistant statements.
        excerpt = '\n'.join('> ' + line for line in d.page_content.splitlines())
        sections.append(f'**Passage [{i}]**\n\n{excerpt}')
    if re.search(r'\b(verify|true|supersed|current|latest)\w*\b', query, re.I):
        notices.append('Local excerpts cannot verify a claim against current literature. No external verification was performed.')
    return {'question':query, 'answer':'\n\n'.join(sections), 'sources':sources, 'notices':notices}
