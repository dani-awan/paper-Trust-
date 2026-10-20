"""Local retrieval inspection; --ai explicitly enables configured cloud calls."""
import argparse
import json
from pathlib import Path
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ai', action='store_true', help='Allow configured Groq/Azure inference calls; service usage limits/charges apply')
    args = parser.parse_args()
    from backend.config import DATA_DIR, ai_enabled
    from backend.paper_loader import load_document
    from backend.chat_service import answer_question
    from backend.vector_store import add_paper, delete_index
    from backend.offline import answer_locally
    if args.ai and not ai_enabled():
        parser.error('Configure an AI provider first, or omit --ai for local excerpts.')
    pairs = json.loads(Path('goldens.json').read_text(encoding='utf-8'))
    docs = load_document('documents/Openclaw_Research_Report.pdf')
    sid = 'evaluation_' + str(uuid4())
    use_ai = args.ai
    results = []
    try:
        add_paper(docs, sid)
        for pair in pairs:
            query = pair['input']
            if use_ai:
                output = answer_question(query, sid)
            else:
                output = answer_locally(query, sid)
            results.append({'input':query,'mode':'cloud AI' if args.ai else 'local excerpts','answer':output['answer'], 'sources':output.get('sources',[]), 'notices':output.get('notices',[])})
        output = DATA_DIR / 'retrieval_inspection.json'
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Saved {len(results)} inspected questions to {output}. This is not an accuracy score.')
    finally:
        delete_index(sid)


if __name__ == '__main__':
    main()
