import pytest
import pymupdf
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from backend import vector_store as store, chat_service as chat
from backend.config import connection_scope
from backend.retrieval import requested_pages, retrieve_for_chat
from backend.paper_loader import load_document

@pytest.fixture
def indexed(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB_PATH',tmp_path/'pages.db')
    for source,title in [('a','First report'),('b','Second report')]:
        store.add_paper([Document(page_content=f'{title} page {p+1}: unique evidence',metadata={'source':source,'title':title,'page':p,'page_count':4}) for p in (0,1,3)],'mine')
    store.add_paper([Document(page_content='Private secret',metadata={'source':'private','title':'Other user','page':0})],'other')

@pytest.mark.parametrize('question,pages',[
    ('Explain page 2',[2]),('Summarize pages 2–4',[2,3,4]),
    ('Compare pages 2 and 4',[2,4]),('Explain pages 1, 2 and 4',[1,2,4]),
])
def test_page_numbers(question,pages):assert requested_pages(question)==pages

@pytest.mark.parametrize('question',['Explain pages 1-100000000','Explain page 0','Explain pages 4-2'])
def test_invalid_ranges_are_bounded(question):
    with pytest.raises(ValueError):requested_pages(question)

def test_page_questions_do_not_leak_other_pages_or_sources(indexed):
    result=retrieve_for_chat('Explain page 2','mine',source='a')
    assert len(result['docs'])==1
    assert all(d.metadata['source']=='a' and d.metadata['page']==1 for d in result['docs'])
    assert result['titles']==['First report']
    assert requested_pages('next page',[{'question':'Explain page 2'}])==[3]

def test_missing_pages_never_substitute_other_text(indexed):
    result=retrieve_for_chat('Explain page 3','mine',source='a')
    assert result['docs']==[]
    assert any('No readable indexed text' in n for n in result['notices'])

def test_scope_isolation_and_ambiguous_page_selection(indexed):
    with pytest.raises(ValueError,match='Select one document'):retrieve_for_chat('Explain page 1','mine')
    with pytest.raises(ValueError,match='unavailable'):retrieve_for_chat('Explain page 1','mine',source='private')
    result=retrieve_for_chat('Summarize this document','mine',source='b')
    assert {d.metadata['source'] for d in result['docs']}=={'b'}
    assert {d.metadata['source'] for d in retrieve_for_chat('Compare these reports','mine')['docs']}=={'a','b'}

def test_model_receives_missing_page_notice_not_unrelated_content(indexed,monkeypatch):
    seen=[]
    class Client:
        def invoke(self,messages):
            seen.extend(messages)
            return AIMessage(content='No readable indexed text is available for page 3.')
    monkeypatch.setattr(chat,'get_llm',Client)
    with connection_scope({'AI_PROVIDER':'groq','GROQ_API_KEY':'mock'}):
        result=chat.answer_question('Explain page 3','mine',source='a')
    assert result['sources']==[] and result['document_source']=='a'
    assert 'No readable indexed text' in seen[-1].content
    assert 'unique evidence' not in seen[-1].content

def test_pdf_physical_pages_blank_and_damaged_inputs(tmp_path):
    path=tmp_path/'sample.pdf'
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50,50),'First readable page')
        pdf.new_page()
        pdf.new_page().insert_text((50,50),'Third readable page')
        pdf.save(path)
    docs=load_document(str(path))
    assert {d.metadata['page'] for d in docs}=={0,2}
    assert all(d.metadata['page_count']==3 for d in docs)
    path.write_bytes(b'not a valid PDF')
    with pytest.raises(ValueError,match='damaged'):load_document(str(path))

def test_password_protected_pdf_has_actionable_error(tmp_path):
    path=tmp_path/'locked.pdf'
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50,50),'Protected text')
        pdf.save(path,encryption=pymupdf.PDF_ENCRYPT_AES_256,owner_pw='owner',user_pw='reader')
    with pytest.raises(ValueError,match='password protected'):load_document(str(path))


def test_relative_pages_use_resolved_previous_page():
    assert requested_pages('next page',[{'question':'Explain page 2','requested_pages':[2]},{'question':'next page','requested_pages':[3]}])==[4]
