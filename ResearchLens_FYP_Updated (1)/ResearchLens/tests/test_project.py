import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from backend import accounts, turn_store, rag_graph as rg, vector_store as vs
from backend.models import RelevancyDecision, RouterDecision, ClaimVerificationResult
from backend.paper_loader import load_document, _public_https, load_webpage
from backend.reports import pdf_report, markdown_report


@pytest.fixture
def users(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, 'DB_PATH', tmp_path / 'accounts.db')
    monkeypatch.setattr(turn_store, 'DB_PATH', tmp_path / 'turns.db')
    one = accounts.register('danial', 'a-long-password-123')
    two = accounts.register('ahmad', 'another-password-456')
    return one, two


def test_passwords_and_sessions_are_owner_scoped(users):
    one, two = users
    sid = accounts.create_session(one)
    assert accounts.authenticate('danial', 'a-long-password-123') == one
    assert accounts.authenticate('danial', 'wrong') is None
    assert len(accounts.list_sessions(one)) == 1
    assert accounts.list_sessions(two) == []
    for action in [lambda: accounts.require_owner(two, sid), lambda: accounts.rename_session(two, sid, 'stolen'), lambda: turn_store.load_turns(two, sid), lambda: turn_store.save_turn(two, sid, {})]:
        with pytest.raises(PermissionError):
            action()
    turn_store.save_turn(one, sid, {'question': 'Q', 'answer': 'A', 'sources': [{'id': 1}]})
    assert turn_store.load_turns(one, sid)[0]['sources'] == [{'id': 1}]
    with accounts.connection() as c:
        stored = c.execute('SELECT digest FROM users WHERE id=?', (one,)).fetchone()[0]
        assert stored != 'a-long-password-123'


def test_failed_login_throttling(users):
    for _ in range(5):
        assert accounts.authenticate('danial', 'wrong') is None
    with pytest.raises(ValueError, match='Too many'):
        accounts.authenticate('danial', 'wrong')


def test_upload_sources_stable_across_temp_paths(tmp_path):
    a, b = tmp_path / 'a.txt', tmp_path / 'b.txt'
    a.write_text('A source about neural networks. ' * 100)
    b.write_bytes(a.read_bytes())
    assert load_document(str(a))[0].metadata['source'] == load_document(str(b))[0].metadata['source']
    assert len(load_document(str(a))) > 1


def test_private_and_unsafe_urls_rejected(monkeypatch):
    with pytest.raises(ValueError):
        _public_https('http://example.com')
    monkeypatch.setattr('socket.getaddrinfo', lambda *a, **k: [(2, 1, 6, '', ('127.0.0.1', 443))])
    with pytest.raises(ValueError, match='Private'):
        _public_https('https://localhost')
    monkeypatch.delenv('ALLOW_WEB_IMPORT', raising=False)
    with pytest.raises(ValueError, match='disabled'):
        load_webpage('https://example.org')


def test_local_index_dedup_search_and_session_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, 'DB_PATH', tmp_path / 'papers.db')
    docs = [Document(page_content='Neural network methodology and experiment evidence', metadata={'source': 'upload:abc', 'title':'Paper','page':0})]
    vs.add_paper(docs, 'one')
    vs.add_paper(docs, 'one')
    assert vs.list_papers('one') == ['Paper']
    assert vs.list_papers('two') == []
    assert len(vs.search('neural', 'one')) == 1
    assert vs.search('neural','two') == []
    assert vs.search('unrelatedword', 'one') == []
    assert vs.search('" OR * --', 'one') == []
    with vs._connect() as c:
        assert c.execute('SELECT count(*) FROM chunks').fetchone()[0] == 1
    vs.delete_index('one')
    assert not vs.list_papers('one')


class FakeLLM:
    def __init__(self, route='retrieve', relevant=True):
        self.route, self.relevant, self.calls = route, relevant, []
    def with_structured_output(self, model):
        parent = self
        class Structured:
            def invoke(self, messages):
                parent.calls.append(messages)
                if model is RouterDecision:
                    return RouterDecision(route=parent.route)
                if model is RelevancyDecision:
                    return RelevancyDecision(is_relevant=parent.relevant, relevant_indices=[1] if parent.relevant else [])
                return ClaimVerificationResult(status='insufficient_evidence', verdict_summary='Limited snippets [1].')
        return Structured()
    def invoke(self, messages):
        self.calls.append(messages)
        if 'Rewrite' in str(messages[0].content):
            return AIMessage(content='standalone rewritten query')
        return AIMessage(content='Evidence indicates a limitation [1].')


def test_graph_retrieval_citations_history_and_persistence(tmp_path, monkeypatch):
    lm = FakeLLM()
    monkeypatch.setattr(rg, 'get_llm', lambda: lm)
    monkeypatch.setattr(rg, 'search', lambda *a, **k: [Document(page_content='Specific finding', metadata={'title': 'Test paper', 'page': 4})])
    graph = rg.build_graph(tmp_path / 'graph.db')
    cfg = {'configurable': {'thread_id': 'one'}}
    result = graph.invoke(rg.initial_state('What did they find?', 'one'), cfg)
    assert result['sources'][0]['page'] == 5
    assert '[1]' in result['answer']
    graph.invoke(rg.initial_state('Explain this limitation', 'one'), cfg)
    assert any('What did they find?' in str(m) for m in lm.calls[-1])
    assert graph.get_state(cfg).values['answer']


def test_no_evidence_rewrites_once_and_never_invents_answer(tmp_path, monkeypatch):
    lm = FakeLLM(relevant=False)
    monkeypatch.setattr(rg, 'get_llm', lambda: lm)
    monkeypatch.setattr(rg, 'search', lambda *a, **k: [])
    graph = rg.build_graph(tmp_path / 'graph.db')
    result = graph.invoke(rg.initial_state('Original question', 's'), {'configurable': {'thread_id': 's'}})
    assert result['rewrite_count'] == 1
    assert result['query'] == 'Original question'
    assert 'sufficient relevant evidence' in result['answer']
    assert result['sources'] == []


def test_verification_without_web_is_inconclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, 'get_llm', lambda: FakeLLM('verify_claim'))
    monkeypatch.setattr(rg, 'search', lambda *a, **k: [])
    monkeypatch.delenv('TAVILY_API_KEY', raising=False)
    graph = rg.build_graph(tmp_path / 'graph.db')
    result = graph.invoke(rg.initial_state('Verify this claim', 's'), {'configurable': {'thread_id': 's'}})
    assert result['verification']['status'] == 'insufficient_evidence'
    assert result['notices']


def test_invalid_citations_are_flagged(monkeypatch):
    class BadLLM:
        def invoke(self, messages):
            return AIMessage(content='Claim [99].')
    monkeypatch.setattr(rg, 'get_llm', BadLLM)
    result = rg.generate_answer_node({'route': 'retrieve', 'query': 'Q', 'messages': [], 'retrieved_docs': [Document(page_content='data')], 'notices': []})
    assert '[99]' not in result['answer']
    assert 'unverified reference' in result['answer']
    assert result['notices']


def test_pdf_export_escapes_markup_and_keeps_sources(tmp_path):
    import pymupdf
    turns = [{'question': 'What does x < y mean?', 'answer': 'A careful answer [1].\n' + 'Useful research notes. ' * 400, 'sources': [{'id': 1, 'title': 'Paper <Test>', 'page': 2, 'url': 'https://example.org/paper'}]}]
    data = pdf_report('Test research notes', turns)
    output = tmp_path / 'sample.pdf'
    output.write_bytes(data)
    with pymupdf.open(output) as doc:
        text = ''.join(page.get_text() for page in doc)
        assert len(doc) >= 2
    assert 'Muhammad Danial' in text and 'Muhammad Ahmad' in text
    assert 'Paper <Test>' in text and 'page 2' in text
    assert 'https://example.org/paper' in markdown_report('Notes', turns)


def test_streamlit_login_screen():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path('app.py').resolve()), default_timeout=20).run()
    assert not app.exception
    assert any('Sign in' in t.label for t in app.tabs)


def test_signed_in_local_workspace_requires_ai_for_chat(users, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(vs, 'DB_PATH', tmp_path / 'papers.db')
    monkeypatch.setenv('AI_PROVIDER','local')
    app=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=20)
    app.session_state['user_id']=users[0]
    app.session_state['username']='danial'
    app.run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any(b.label=='Connect free AI' for b in app.button)
    assert not turn_store.load_turns(users[0],app.session_state['active_sid'])
    next(b for b in app.button if b.label=='Sign out').click().run()
    assert not app.exception
    assert any(t.label=='Sign in' for t in app.tabs)


def test_offline_summary_is_labeled_as_excerpts(tmp_path, monkeypatch):
    from backend.offline import answer_locally
    monkeypatch.setattr(vs, 'DB_PATH', tmp_path / 'offline.db')
    vs.add_paper([Document(page_content='An abstract on experiments.', metadata={'title':'A'})], 'one')
    result = answer_locally('Summarize this paper', 'one')
    assert 'An abstract on experiments.' in result['answer']
    assert 'not an automatically generated summary' in str(result['notices'])
    assert not answer_locally('secretkeyword', 'two')['sources']


def test_azure_url_and_provider_configuration(monkeypatch):
    from backend.llm import azure_base_url
    from backend.config import ai_enabled, missing_settings
    assert azure_base_url('https://my-resource.openai.azure.com/') == 'https://my-resource.openai.azure.com/openai/v1/'
    assert azure_base_url('https://my-resource.services.ai.azure.com/openai/v1') == 'https://my-resource.services.ai.azure.com/openai/v1/'
    for url in ('http://example.com','https://evil.example.com','https://my-resource.openai.azure.com/projects/x'):
        with pytest.raises(ValueError):
            azure_base_url(url)
    monkeypatch.setenv('AI_PROVIDER','azure')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY','fake')
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT','https://test.openai.azure.com')
    monkeypatch.delenv('AZURE_OPENAI_DEPLOYMENT', raising=False)
    assert not ai_enabled()
    assert 'AZURE_OPENAI_DEPLOYMENT' in missing_settings()
    monkeypatch.setenv('AZURE_OPENAI_DEPLOYMENT','my-deployment')
    assert ai_enabled()


@pytest.mark.parametrize('provider,host,model', [('groq','api.groq.com','openai/gpt-oss-20b'),('azure','test.openai.azure.com','my-deployment')])
def test_cloud_adapter_uses_selected_endpoint_and_json_schema(monkeypatch, provider, host, model):
    import httpx
    from langchain_openai import ChatOpenAI
    from backend.llm import CloudChat
    calls = []
    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        assert body['model'] == model
        assert body['response_format']['type'] == 'json_object'
        return httpx.Response(200, json={'id':'test','object':'chat.completion','created':0,'model':model,'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'{"route":"retrieve"}'}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        chat = CloudChat(ChatOpenAI(api_key='fake', base_url=f'https://{host}/openai/v1/', model=model, use_responses_api=False, http_client=client, max_retries=0))
        answer = chat.with_structured_output(RouterDecision).invoke([HumanMessage(content='Find a method')])
        assert answer.route == 'retrieve'
    assert calls[0].url.host == host
    assert '/chat/completions' in calls[0].url.path


def test_api_failure_is_readable(monkeypatch):
    from backend.llm import friendly_error
    monkeypatch.setenv('AI_PROVIDER','azure')
    assert 'Azure' in friendly_error(SimpleNamespace(status_code=429))
    assert 'quota' in friendly_error(SimpleNamespace(status_code=429))


def test_old_openai_key_never_enables_paid_calls(monkeypatch):
    from backend.config import ai_enabled
    from backend.llm import get_llm
    monkeypatch.setenv('AI_PROVIDER','local')
    monkeypatch.setenv('OPENAI_API_KEY','old-key-not-used')
    assert not ai_enabled()
    with pytest.raises(ValueError, match='Local mode'):
        get_llm()


def test_cloud_quota_failure_is_not_saved_as_an_answer(users,tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from backend import chat_service as chat
    monkeypatch.setattr(vs,'DB_PATH',tmp_path/'papers.db')
    monkeypatch.setenv('AI_PROVIDER','groq')
    monkeypatch.setenv('GROQ_API_KEY','test-key')
    class QuotaError(Exception):
        status_code=429
    class FailedClient:
        def invoke(self,messages):raise QuotaError('mock quota')
    monkeypatch.setattr(chat,'get_llm',FailedClient)
    sid=accounts.create_session(users[0])
    vs.add_paper([Document(page_content='The model uses Logistic Regression.',metadata={'title':'Report'})],sid)
    app=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=20)
    app.session_state['user_id']=users[0]
    app.session_state['username']='danial'
    app.run()
    app.chat_input[0].set_value('What methodology is used?').run()
    assert not app.exception
    assert not turn_store.load_turns(users[0],sid)
    assert 'quota' in app.session_state['failed_question']['message']
    assert not any('Local source excerpts' in m.value for m in app.markdown)


def test_in_app_groq_connection_then_sample_question(users,tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from backend import llm, chat_service as chat
    from backend.config import setting
    monkeypatch.setattr(vs,'DB_PATH',tmp_path/'papers.db')
    monkeypatch.setenv('AI_PROVIDER','local')
    calls=[]
    class TestClient:
        def invoke(self,messages):
            calls.append((setting('GROQ_API_KEY'),messages))
            return AIMessage(content='READY' if len(messages)==1 else 'This tutorial describes SQLite full-text search [1].')
    monkeypatch.setattr(llm,'get_llm',TestClient)
    monkeypatch.setattr(chat,'get_llm',TestClient)
    app=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=20)
    app.session_state['user_id']=users[0]
    app.session_state['username']='danial'
    app.run()
    next(b for b in app.button if b.label=='Try tutorial document').click().run()
    assert app.chat_input[0].disabled
    next(b for b in app.button if b.label=='Connect free AI').click().run()
    assert not app.exception
    next(t for t in app.text_input if t.label=='Groq API key').set_value('private-test-key')
    next(b for b in app.button if b.label=='Check & connect Groq').click().run()
    assert not app.exception
    assert not app.chat_input[0].disabled
    assert app.session_state['page']=='Chat'
    app.chat_input[0].set_value('What methodology does this document use?').run()
    assert not app.exception
    turns=turn_store.load_turns(users[0],app.session_state['active_sid'])
    assert turns[0]['answer_kind']=='ai'
    assert '[1]' in turns[0]['answer']
    assert 'Local source excerpts' not in turns[0]['answer']
    assert all(key=='private-test-key' for key,_ in calls)
    assert 'private-test-key' not in json.dumps(turns)
    next(b for b in app.button if b.label=='Sign out').click().run()
    with pytest.raises(KeyError):
        app.session_state['connection']


def test_rejected_in_app_key_does_not_enable_chat(users,tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from backend import llm
    monkeypatch.setattr(vs,'DB_PATH',tmp_path/'papers.db')
    monkeypatch.setenv('AI_PROVIDER','local')
    class AuthenticationError(Exception):
        status_code=401
    class InvalidClient:
        def invoke(self,messages):raise AuthenticationError('mock rejected key')
    monkeypatch.setattr(llm,'get_llm',InvalidClient)
    app=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=20)
    app.session_state['user_id']=users[0]
    app.session_state['username']='danial'
    app.run()
    next(b for b in app.button if b.label=='Connect free AI').click().run()
    next(t for t in app.text_input if t.label=='Groq API key').set_value('rejected-test-key')
    next(b for b in app.button if b.label=='Check & connect Groq').click().run()
    assert not app.exception
    assert any('rejected' in e.value for e in app.error)
    with pytest.raises(KeyError):app.session_state['connection']
