# ResearchLens - Conversational Research Assistant

Customization and improvements: **Muhammad Danial & Muhammad Ahmad**.
Adapted from the supplied Papeer project; see `PROVENANCE.md`.

## Start here

1. Extract this complete ZIP into a fresh folder.
2. Install Python 3.12. On Windows with the Python install manager, run
   `py install 3.12`, then check `py -3.12 --version`.
3. Double-click `START_WINDOWS.bat`. The first installation needs internet.
4. Create a local account or sign in.
5. Open **AI connection**. Choose **Groq · free tier**, paste your Groq API key,
   and click **Check & connect Groq**. Get your key from https://console.groq.com/keys.
6. Attach a PDF using **Documents** in the sidebar and click **Add files**.
7. Ask natural questions in **Chat**. Written answers stream into the conversation;
   references and original passages stay under **Sources**.
8. Use **Export PDF** or **Text notes** to save the conversation.

No OpenAI Platform credits or local LLM download are required. Groq's Free plan
has usage limits and requires an account. Azure is optional and can consume
credits or incur charges; it is not needed for the Groq setup. Do not upgrade
to a paid plan just to run this demonstration.

You no longer need to open `.env` or select an answer mode to begin. Keys entered
inside the app are stored only in the current signed-in Streamlit session, are
not written to the research database/exports, and are cleared by signing out.
Restarting the app or ending the session may require reconnecting. A persistent
personal environment setup remains optional below. Use HTTPS for remote hosting.

## Document and page questions

Use **Answer using** to choose an attached document or all documents. For a page
question select one PDF, then ask "Explain page 2", "Summarize pages 2–4", or
"Compare pages 2 and 4". Page numbers mean physical PDF pages, starting at 1.
"Next page" and "previous page" follow the most recent page question within
the same document scope. Requests are limited to seven pages at a time.

If a page is blank, scanned, or outside the indexed document, the assistant is
told that no readable text was found; unrelated pages are not substituted.
Password-protected or damaged PDFs show an actionable upload error. Scanned
PDFs need OCR before upload. Retrieval uses bounded excerpts, so answers can
miss details; check the cited text. This is not a guarantee of correctness.

Try these questions for your FYP demonstration:
- What is this document about?
- Explain its methodology in simple English.
- What limitations does the document describe?
- Explain page 2.
- Does the document provide a dataset size? If not, say what is missing.

## Why this release changes the previous behavior

The previous local mode returned copied passages. It did not generate a summary,
and exact keyword search missed ordinary questions. This release separates
those capabilities:

- **Chat** requires a configured AI provider and writes English answers.
- **Documents** contains a separate, explicitly labeled keyless passage search.
- A missing key is shown before chat starts. An in-app connection check makes
  one small provider request and confirms an actual response.
- Failed/limited AI requests show an error and a retry action. They are not
  saved as successful answers and are not replaced with excerpt dumps.
- Older excerpt-only chat results are collapsed and labeled; you can re-answer
  their questions with AI after connecting.
- Broad summary/about questions use representative opening, middle and closing
  sections. Method/limitation questions expand search terms, so they do not depend
  on exact section-heading matches.
- Informal English and Roman Urdu questions are understood by the prompt and
  common intent rules; the interface and normal generated answers are English.
- Ordinary document questions use one LLM request, rather than separate LLM
  routing, relevance-check and rewriting requests. This reduces free-tier usage.

## Supported questions

- What is this document about?
- Summarize this paper.
- What methodology does this paper use?
- What limitations are described?
- Explain the model scores in simple language.
- What information is missing from this report?

If the uploaded file is an analysis report rather than a research article, the
assistant is instructed to say so. Missing methods/training details must remain
unknown; generated answers should not invent them. A report containing uncalibrated
behavioral model scores must not be presented as a diagnosis or clinical probability.

## Architecture

`app.py` supplies the English chat interface, owner checks, connection forms,
streaming display, retry actions and export controls.

The main answer path is a LangGraph workflow in `backend/chat_service.py`:

1. `gather`: intent-aware SQLite FTS5/BM25 retrieval with representative passages.
2. `respond`: one source-grounded AI request using conversation history, followed
   by citation-number validation.

`backend/retrieval.py` handles intent expansion. It is still lexical retrieval,
**not neural embedding/semantic search**. It can miss paraphrases and relevant
sections. Source context is bounded, so a response is not an exhaustive review.
`backend/rag_graph.py` retains the earlier experimental multi-step workflow and
shared source/web helpers. Its old tests are not a quality score for the new chat.
Finalized, owner-scoped turns are stored through `backend/turn_store.py`.
The connection scope uses context-local settings; API clients are not globally
cached with another user's credentials.

## Optional persistent configuration

For a private local installation you may put these values in `.env`:

```env
AI_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b
```

If you already added a Groq key but left `AI_PROVIDER=local`, the connection page
provides **Activate the Groq key already in .env**. No key is displayed on screen.
Never commit `.env` or share API keys in chat/screenshots.

Azure remains optional:

```env
AI_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_resource_key
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_exact_deployment_name
```

Azure requires an eligible subscription, deployment and quota. The app uses its
v1 Chat Completions endpoint. The deployment must support text chat and, for
legacy structured workflows, JSON-object output. Public Azure resource endpoints
are supported; sovereign cloud and Entra authentication are not implemented.

`TAVILY_API_KEY` optionally enables current-web search. Without it, current-claim
requests are explicitly limited to uploaded evidence; they are not called verified.
Web snippets are not a systematic literature review. The app does not register
accounts, create cloud deployments or purchase/upgrade anything for you.

## Files and data

`DATA_DIR=data` contains private accounts, indexed passages and saved conversations.
Back it up as one unit. Passwords are salted PBKDF2 hashes; documents/chat databases
are not encrypted at rest. The app is a prototype with local accounts, not a
managed identity/enterprise security platform. Add HTTPS, registration/traffic
controls and backups before public hosting.

Existing SQLite accounts and finalized turns remain compatible. Old Qdrant data
is not deleted, but its papers must be uploaded into the newer local index.
Scanned PDFs need OCR first. Upload limit: 20 MB; extracted-text limit: 2 million
characters. Complex formulas/tables may lose formatting during PDF extraction.

For a fresh test, use **Try tutorial document** in the sidebar. This is a tutorial,
not a published research paper. It still needs an AI connection for written chat.
Passage search under Documents remains keyless.

## Manual launch

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On Linux/macOS, use a Python 3.12 virtual environment and
`python -m streamlit run app.py`. The refreshed `uv.lock` supports `uv sync` too.
Docker is optional; persistent data-volume instructions are in `DOCKER_GUIDE.md`.

## Verification and scientific limits

Run `pip install pytest==9.1.1`, then `python -m pytest -q`.

The tests cover actual SQLite retrieval for the user's reported question wording,
the new chat graph, one-call request behavior, English prompting, per-session
credential isolation, connection -> upload -> chat -> saved reference flow,
streaming, quota errors and the rule that failed AI calls are not saved as answers.
They also retain account/PDF/source validation checks.

AI services are mocked in automated tests. No real Groq/Azure key was available
for live answer testing in this environment. These checks establish implementation
behavior, not research accuracy or whether your account can access the service.
Click **Check & connect Groq** on your own computer for the live connection test.
Do not send your key to the developer or paste it into this chat.

For an FYP, run a separate, human-reviewed evaluation on your actual documents:
answer correctness, evidence/citation support, missing-information handling,
response time and limits across different document types. Do not claim the old
historical scores in `provenance/` as this release's accuracy.
