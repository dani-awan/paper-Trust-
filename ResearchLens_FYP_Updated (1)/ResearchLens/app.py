"""English, chat-first research assistant with explicit AI connection checks."""
import hashlib
import logging
import os
import tempfile
from html import escape
from pathlib import Path
import streamlit as st
from backend import accounts
from backend.config import CREDITS, MAX_UPLOAD_BYTES, ai_enabled, provider, provider_label, model_name, setting, connection_scope
from backend.llm import check_connection, friendly_error
from backend.paper_loader import load_document, load_arxiv
from backend.vector_store import add_paper, list_papers, document_catalog
from backend.turn_store import load_turns, save_turn
from backend.chat_service import answer_question
from backend.offline import answer_locally
from backend.reports import pdf_report, markdown_report

ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
st.set_page_config(page_title='ResearchLens',page_icon='📘',layout='wide',initial_sidebar_state='auto')
st.markdown((ROOT/'ui/theme.css').read_text(),unsafe_allow_html=True)


def fingerprint():
    fields = [provider(),model_name(),setting('GROQ_API_KEY'),setting('AZURE_OPENAI_API_KEY'),setting('AZURE_OPENAI_ENDPOINT')]
    return hashlib.sha256('|'.join(fields).encode()).hexdigest()


def navigate(page):
    st.session_state.page = page


def connect(candidate):
    try:
        with connection_scope(candidate):
            with st.spinner('Checking your AI connection...'):
                check_connection()
            checked = fingerprint()
        st.session_state.connection = candidate
        st.session_state.connection_verified = checked
        st.session_state.flash = 'AI connected. You can now ask questions and get written answers.'
        st.session_state.next_page = 'Chat'
        st.rerun()
    except Exception as error:
        with connection_scope(candidate):
            st.error(friendly_error(error))
        st.caption('The connection was not saved. Check the key, account limits and model, then try again.')


def login():
    left,right = st.columns([1.05,1],gap='large')
    with left:
        st.markdown('''<div class="login-wrap"><div class="eyebrow">READ. ASK. UNDERSTAND.</div><div class="login-title">Your papers.<br>A clearer <em>conversation.</em></div><p class="login-desc">Ask natural questions, get written answers, and check the passages behind every document reference.</p><div class="login-feature"><span>1</span>Add a PDF or try a tutorial document</div><div class="login-feature"><span>2</span>Connect free-tier AI inside the app</div><div class="login-feature"><span>3</span>Ask questions and export your notes</div></div>''',unsafe_allow_html=True)
    with right,st.container(border=True):
        st.subheader('Welcome to ResearchLens')
        signin,signup = st.tabs(['Sign in','Create account'])
        with signin:
            with st.form('login'):
                name=st.text_input('Username',key='login_name')
                password=st.text_input('Password',type='password',key='login_pass')
                submitted=st.form_submit_button('Open my workspace',type='primary',use_container_width=True)
            if submitted:
                try:
                    uid=accounts.authenticate(name,password)
                    if not uid:
                        st.error('Incorrect username or password.')
                    else:
                        st.session_state.user_id=uid
                        st.session_state.username=name.strip().lower()
                        st.rerun()
                except ValueError as error:st.error(str(error))
        with signup:
            with st.form('register'):
                name=st.text_input('Choose username',help='3-40 letters, digits or underscores')
                password=st.text_input('Choose password',type='password',help='At least 12 characters')
                confirm=st.text_input('Confirm password',type='password')
                submitted=st.form_submit_button('Create my workspace',type='primary',use_container_width=True)
            if submitted:
                try:
                    if password!=confirm:raise ValueError('Passwords do not match.')
                    uid=accounts.register(name,password)
                    st.session_state.user_id=uid
                    st.session_state.username=name.strip().lower()
                    st.rerun()
                except ValueError as error:st.error(str(error))
    st.caption(f'Customization & improvements · {CREDITS}')


def show_sources(sources):
    if sources:
        with st.expander(f'Sources · {len(sources)} references'):
            for source in sources:
                st.markdown(f"**[{source['id']}] {source['title']}**"+(f" · Page {source['page']}" if source.get('page') else ''))
                if source.get('url'):st.link_button('Open original source',source['url'])
                if source.get('excerpt'):st.caption(source['excerpt'])


def add_files(files,owner,sid):
    for file in files or []:
        try:
            accounts.require_owner(owner,sid)
            if file.size>MAX_UPLOAD_BYTES:raise ValueError('Choose a file smaller than 20 MB.')
            with st.spinner(f'Reading {file.name}...'):
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/('paper'+Path(file.name).suffix)
                    path.write_bytes(file.getvalue())
                    docs=load_document(str(path))
                for doc in docs:doc.metadata['title']=Path(file.name).stem
                add_paper(docs,sid)
        except ValueError as error:st.error(str(error));return
        except Exception:
            log.exception('Document import failed')
            st.error('The file could not be read. Try a text-based PDF or check available disk space.');return
    st.session_state.flash='Documents ready. Ask your first question.'
    st.rerun()


def connection_page():
    st.header('Connect your AI')
    st.write('This enables real written answers. Uploading documents alone only prepares them for search.')
    st.info('For your free setup, choose Groq. Its Free plan has request and token limits; no OpenAI Platform credits are used.')
    chosen=st.selectbox('Provider',['Groq · free tier','Azure · existing credits'],key='connect_provider')
    active=st.session_state.get('connection',{})
    if chosen.startswith('Groq'):
        with st.form('connect_groq',clear_on_submit=True):
            key=st.text_input('Groq API key',type='password',placeholder='Paste your key here',help='Stored only in this signed-in browser session. It is not written to your notes or database.')
            model=st.text_input('Model',value=setting('GROQ_MODEL','openai/gpt-oss-20b'))
            st.caption('Get a key from console.groq.com/keys. Stay on the Free plan for free-tier usage.')
            submit=st.form_submit_button('Check & connect Groq',type='primary',use_container_width=True)
        if submit:
            key=key.strip() or active.get('GROQ_API_KEY','') or os.getenv('GROQ_API_KEY','')
            if not key:st.error('Paste your Groq key before connecting.')
            else:connect({'AI_PROVIDER':'groq','GROQ_API_KEY':key,'GROQ_MODEL':model.strip() or 'openai/gpt-oss-20b'})
        if os.getenv('GROQ_API_KEY','').strip():
            if st.button('Activate the Groq key already in .env',use_container_width=True):
                connect({'AI_PROVIDER':'groq','GROQ_API_KEY':os.environ['GROQ_API_KEY'],'GROQ_MODEL':os.getenv('GROQ_MODEL','openai/gpt-oss-20b')})
    else:
        with st.form('connect_azure',clear_on_submit=True):
            key=st.text_input('Azure resource key',type='password')
            endpoint=st.text_input('Azure resource endpoint',placeholder='https://YOUR-RESOURCE.openai.azure.com/')
            deployment=st.text_input('Deployment name')
            st.caption('Azure calls can consume credits or incur charges. A supported deployment and quota are required.')
            submit=st.form_submit_button('Check & connect Azure',type='primary')
        if submit:connect({'AI_PROVIDER':'azure','AZURE_OPENAI_API_KEY':key.strip(),'AZURE_OPENAI_ENDPOINT':endpoint.strip(),'AZURE_OPENAI_DEPLOYMENT':deployment.strip()})
    if ai_enabled():
        st.caption(f'Current provider: {provider_label()} · Model/deployment: {model_name()}')
        if st.button('Test current connection'):
            try:
                with st.spinner('Testing...'):check_connection()
                st.session_state.connection_verified=fingerprint()
                st.success('The AI service replied successfully.')
            except Exception as error:st.error(friendly_error(error))
    st.caption('Keys entered here are session-only. Sign out to clear them. Never share your key in screenshots or chat.')
    with st.expander('Why did I see copied passages before?'):
        st.write('The previous local mode was a source search, not a generative model. Chat now requires a configured AI connection. Source passages are available separately under Documents, and failed AI calls are not saved as completed answers.')


def main():
    if not st.session_state.get('user_id'):
        login();return
    owner=st.session_state.user_id
    sessions=accounts.list_sessions(owner)
    if not sessions:
        accounts.create_session(owner);sessions=accounts.list_sessions(owner)
    labels={s['id']:s['name'] for s in sessions}
    if st.session_state.get('active_sid') not in labels:st.session_state.active_sid=sessions[0]['id']
    if st.session_state.get('next_page'):
        st.session_state.page=st.session_state.pop('next_page')
    if 'page' not in st.session_state:st.session_state.page='Chat'
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-icon">▤</span>ResearchLens<span style="color:#4d80eb">.</span></div>',unsafe_allow_html=True)
        if st.button('+ New chat',type='primary',use_container_width=True):
            st.session_state.active_sid=accounts.create_session(owner)
            st.session_state.pop('failed_question',None)
            st.session_state.page='Chat';st.rerun()
        st.radio('Workspace',['Chat','Documents','AI connection'],key='page',label_visibility='collapsed')
        sid=st.selectbox('Your chats',list(labels),index=list(labels).index(st.session_state.active_sid),format_func=labels.get)
        st.session_state.active_sid=sid
        accounts.require_owner(owner,sid)
        titles=list_papers(sid)
        st.divider()
        with st.expander(f'Documents · {len(titles)}',expanded=not titles):
            files=st.file_uploader('Attach documents',type=['pdf','txt','md','markdown'],accept_multiple_files=True,key='files_'+sid)
            if st.button('Add files',disabled=not files,use_container_width=True):add_files(files,owner,sid)
            for title in titles[:5]:st.caption('▤ '+title)
            if not titles and st.button('Try tutorial document',use_container_width=True):
                docs=load_document(str(ROOT/'examples/Getting_Started.txt'))
                for doc in docs:doc.metadata['title']='ResearchLens tutorial - not a research paper'
                add_paper(docs,sid)
                st.session_state.flash='Tutorial attached. Connect AI, then ask a question.';st.rerun()
        st.divider()
        status=('Connected · '+provider_label()) if ai_enabled() and st.session_state.get('connection_verified')==fingerprint() else ((provider_label()+' configured · not tested') if ai_enabled() else 'AI not connected')
        st.caption(status)
        st.button('Manage AI connection',on_click=navigate,args=('AI connection',),use_container_width=True)
        st.caption('Signed in as '+st.session_state.username)
        if st.button('Sign out',use_container_width=True):st.session_state.clear();st.rerun()
        st.caption(f'Customization & improvements\n\n{CREDITS}')
    turns=load_turns(owner,sid)
    if st.session_state.get('flash'):st.success(st.session_state.pop('flash'))
    if st.session_state.page=='AI connection':connection_page();return
    if st.session_state.page=='Documents':
        st.header('Your documents')
        st.caption('Attach files from the sidebar. Documents are available only in this chat.')
        for title in titles:st.markdown(f'<div class="paper-row">▤ {escape(title)}</div>',unsafe_allow_html=True)
        if not titles:st.info('No documents yet. Upload a readable PDF or try the tutorial in the sidebar.')
        with st.expander('Search source passages without AI'):
            query=st.text_input('Search for words in your documents')
            if st.button('Search passages',disabled=not titles):
                st.markdown(answer_locally(query,sid,turns)['answer'])
        with st.expander('Add a paper from arXiv'):
            arxiv=st.text_input('arXiv ID or exact paper title')
            if st.button('Import paper') and arxiv.strip():
                try:
                    accounts.require_owner(owner,sid)
                    with st.spinner('Importing...'):add_paper(load_arxiv(arxiv.strip()),sid)
                    st.rerun()
                except Exception:st.error('Import failed. Check the ID or upload the PDF directly.')
        return
    top,export=st.columns([4,1])
    with top:
        st.subheader('Research chat')
        st.caption((str(len(titles))+' document(s) attached · ' if titles else '')+('AI answers with source references' if ai_enabled() else 'Connect AI to start a conversation'))
    with export:
        if turns:
            st.download_button('Export PDF',pdf_report(labels[sid],turns),file_name='ResearchLens_Notes.pdf',mime='application/pdf',use_container_width=True)
            st.download_button('Text notes',markdown_report(labels[sid],turns),file_name='ResearchLens_Notes.md',mime='text/markdown',use_container_width=True)
    if not ai_enabled():
        with st.container(border=True):
            st.markdown('**Connect free-tier AI for written answers**')
            st.write('Your documents can be uploaded now. Paste your Groq key in AI connection to enable summaries and natural questions. No OpenAI credits are required.')
            st.button('Connect free AI',type='primary',on_click=navigate,args=('AI connection',))
    catalog=document_catalog(sid)
    choices=[None]+[item['source'] for item in catalog]
    names={item['source']:item['title'] for item in catalog}
    selected_source=st.selectbox('Answer using',choices,format_func=lambda key: 'All documents' if key is None else names[key],key='document_scope_'+sid)
    if catalog:st.caption('Ask about the method, results, limitations, or a specific PDF page. Select one document for page questions.')
    st.markdown('<p class="question-guide">↓ Ask about your PDF in the blue box at the bottom. Press Enter or click the arrow to send.</p>',unsafe_allow_html=True)
    prompt=None
    if not turns:
        st.markdown('''<div class="chat-welcome"><div class="empty-symbol">⌕</div><h1>What would you like to understand?</h1><p>Ask naturally. Your assistant will explain the document and show its sources.</p></div>''',unsafe_allow_html=True)
        columns=st.columns(3)
        for col,label,question in zip(columns,['Summarize','Explain the method','Find limitations'],['Summarize this document in clear English.','What methodology does this document use?','What limitations are described in this document?']):
            if col.button(label,disabled=not ai_enabled() or not titles,use_container_width=True):prompt=question
        if not titles:st.caption('Attach a PDF from the sidebar to ask questions about it. You can also ask a general question once AI is connected.')
    for index,turn in enumerate(turns):
        with st.chat_message('user'):st.markdown(turn['question'])
        with st.chat_message('assistant'):
            old_source=turn.get('answer_kind')!='ai' and ('Local source excerpts' in turn['answer'] or 'No matching passages' in turn['answer'])
            if old_source:
                st.caption('Earlier source-search result — this was not an AI-written answer.')
                with st.expander('View older search result'):st.markdown(turn['answer'])
                if st.button('Answer this with AI',key=f'reanswer_{sid}_{index}',disabled=not ai_enabled()):prompt=turn['question']
            else:
                st.markdown(turn['answer'])
                for notice in turn.get('notices',[]):st.caption(notice)
                show_sources(turn.get('sources',[]))
    failed=st.session_state.get('failed_question')
    if failed and failed['sid']==sid:
        st.error(failed['message'])
        if st.button('Retry last question',disabled=not ai_enabled()):prompt=failed['question']
    typed=st.chat_input('Ask about your PDF here…' if ai_enabled() else 'Connect free AI above to ask about your PDF…',disabled=not ai_enabled(),key='research_question')
    if typed:prompt=typed
    if prompt:
        if len(prompt)>8000:st.error('Please keep your question below 8,000 characters.');return
        accounts.require_owner(owner,sid)
        with st.chat_message('user'):st.markdown(prompt)
        with st.chat_message('assistant'):
            placeholder=st.empty()
            placeholder.markdown('Reading your document…')
            parts=[]
            def on_token(token):
                parts.append(token);placeholder.markdown(''.join(parts)+' ▌')
            try:
                # Ignore earlier raw excerpt dumps as conversational assistant history.
                history=[t for t in turns if t.get('answer_kind')=='ai' and t.get('document_source')==selected_source]
                turn=answer_question(prompt,sid,history,on_token=on_token,source=selected_source)
                placeholder.markdown(turn['answer'])
                save_turn(owner,sid,turn)
                st.session_state.connection_verified=fingerprint()
                st.session_state.pop('failed_question',None)
                if not turns:accounts.rename_session(owner,sid,prompt[:65])
                st.rerun()
            except Exception as error:
                message=str(error) if isinstance(error,ValueError) else friendly_error(error)
                if isinstance(error,ValueError) and any(word in str(error).lower() for word in ('empty','no answer')):message='The AI service returned an empty answer. Please retry.'
                placeholder.error(message)
                st.session_state.failed_question={'sid':sid,'question':prompt,'message':message}
                st.caption('This request was not saved as an answer. Retry after resolving the connection or quota issue.')
                st.rerun()


with connection_scope(st.session_state.get('connection')):
    main()
