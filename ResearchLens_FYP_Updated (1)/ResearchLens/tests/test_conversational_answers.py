from concurrent.futures import ThreadPoolExecutor
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from backend import vector_store as store, chat_service as chat
from backend.config import connection_scope, setting
from backend.retrieval import retrieve_for_chat


@pytest.fixture
def report(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB_PATH',tmp_path/'report.db')
    docs=[
        Document(page_content='Research analysis report. AV-ASD behavioral labels. Model: AV-ASD combined Logistic Regression. Frozen threshold: 0.30. Research output only; not an autism diagnosis.',metadata={'title':'selected_window_report','source':'upload:test','page':0,'start_index':0}),
        Document(page_content='Input quality: 10 sampled frames, duration 10 seconds. Features: 58. Audio and visual features were used. The classifier was trained on clips.',metadata={'title':'selected_window_report','source':'upload:test','page':1,'start_index':0}),
        Document(page_content='Interpretation and limitations: exploratory fixed-window model scores, not confirmed behaviors. Segment-level predictive accuracy has not been established. Scores are not clinical probabilities.',metadata={'title':'selected_window_report','source':'upload:test','page':6,'start_index':0}),
    ]
    store.add_paper(docs,'one')
    return docs


@pytest.mark.parametrize('question,expected',[
 ('Summarize this paper','overview'),
 ('What methodology does this paper use?','methodology'),
 ('What limitations are described?','limitations'),
 ('ye pdf kiss chees k ha ?','overview'),
 ('ye paper kia ha ?','overview'),
 ('What is this PDF about?','overview'),
])
def test_users_actual_questions_retrieve_evidence(report,question,expected):
    result=retrieve_for_chat(question,'one')
    assert result['intent']==expected
    assert result['docs']
    text=' '.join(doc.page_content for doc in result['docs'])
    if expected=='methodology':assert 'Logistic Regression' in text and '58' in text
    if expected=='limitations':assert 'not clinical probabilities' in text
    if expected=='overview':assert 'Research analysis report' in text


def test_one_request_natural_answer_and_english_prompt(report,monkeypatch):
    calls=[]
    class FakeClient:
        def invoke(self,messages):
            calls.append(messages)
            return AIMessage(content='This is an AV-ASD research analysis report using a Logistic Regression classifier [1]. Its scores are not a diagnosis [3].')
    monkeypatch.setattr(chat,'get_llm',FakeClient)
    with connection_scope({'AI_PROVIDER':'groq','GROQ_API_KEY':'mock'}):
        result=chat.answer_question('ye paper kia ha ?','one')
    assert len(calls)==1
    assert 'clear English' in calls[0][0].content
    assert 'not automatically a peer-reviewed' in calls[0][0].content
    assert 'No matching passages' not in result['answer']
    assert result['answer_kind']=='ai'
    assert result['sources'][0]['excerpt']


def test_session_keys_are_isolated_even_inside_graph_workers(report,monkeypatch):
    class ScopedClient:
        def invoke(self,messages):
            return AIMessage(content='Session '+setting('GROQ_API_KEY'))
    monkeypatch.setattr(chat,'get_llm',ScopedClient)
    def run(key):
        with connection_scope({'AI_PROVIDER':'groq','GROQ_API_KEY':key}):
            return chat.answer_question('Summarize this paper','one')['answer']
    with ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(run,['first-private-key','second-private-key']))
    assert results==['Session first-private-key','Session second-private-key']


def test_disconnected_chat_does_not_return_excerpt_as_answer(report):
    with connection_scope({'AI_PROVIDER':'local'}):
        with pytest.raises(chat.ConnectionRequiredError):
            chat.answer_question('What is this paper?','one')


def test_streamed_answer_is_finalized_and_invalid_citation_flagged(report,monkeypatch):
    class StreamingClient:
        def stream(self,messages):
            yield AIMessage(content='A short ')
            yield AIMessage(content='answer [999].')
    monkeypatch.setattr(chat,'get_llm',StreamingClient)
    tokens=[]
    with connection_scope({'AI_PROVIDER':'groq','GROQ_API_KEY':'mock'}):
        result=chat.answer_question('Summarize this paper','one',on_token=tokens.append)
    assert ''.join(tokens)=='A short answer [999].'
    assert '[999]' not in result['answer']
    assert 'unverified reference' in result['answer']
    assert result['notices']


def test_actual_client_factory_never_reuses_another_sessions_key(monkeypatch):
    from backend import llm
    captured=[]
    def factory(**options):
        captured.append(options)
        return object()
    monkeypatch.setattr(llm,'ChatOpenAI',factory)
    for key in ('first-key','second-key'):
        with connection_scope({'AI_PROVIDER':'groq','GROQ_API_KEY':key,'GROQ_MODEL':'openai/gpt-oss-20b'}):
            llm.get_llm()
    assert [item['api_key'] for item in captured]==['first-key','second-key']
    assert all(item['base_url']=='https://api.groq.com/openai/v1' for item in captured)
