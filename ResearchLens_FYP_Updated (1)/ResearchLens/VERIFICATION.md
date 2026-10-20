# Verification - Chat Edition 0.5.0

Date: 19 October 2026. Python 3.12, Linux.

46 offline automated checks pass. They cover account isolation, password
throttling, document IDs, SQLite search, source/reference handling, PDF export,
legacy graph behavior, and the new two-node conversational graph.

Specific regression checks use a fixture built from the report excerpts pasted
by the user. The questions include:
- Summarize this paper.
- What methodology does this paper use?
- What limitations are described?
- ye pdf kiss chees k ha ?
- ye paper kia ha ?
- What is this PDF about?

These checks establish that relevant context is retrieved despite absent exact
headings and informal wording. Fake model responses test English prompting,
normal one-request behavior, streaming, citation checks, missing-connection
behavior, and credential scope through graph worker threads. They do not prove
actual model answer quality on the complete uploaded PDF.

Streamlit AppTest exercises a signed-in tutorial -> connect Groq (mocked service)
-> written answer -> saved references -> sign-out flow. A rejected key is not
saved as a working connection. A quota-failed question is not stored as an answer
and is not replaced by copied source passages. Keys are isolated per session,
not cached globally or copied into research notes.

A real headless browser rendered and visually inspected the new English Chat
and AI connection screens, including navigation from Connect free AI to the
connection form. Browser/model-service network requests were not used to claim
any live AI response. Desktop rendering was checked; mobile and Windows visual
verification remain pending.

No real Groq/Azure credentials were provided for live service testing. No cloud
resources were created and no credits were spent. The user must run Check &
connect in their installation. Free-tier eligibility, provider availability,
quota and actual answer quality remain account/service-dependent.

Not performed: Windows batch execution, Docker build/run, live web search,
independent human-reviewed research-accuracy evaluation, or an 8 GB RAM benchmark.
Historical evaluation scores in provenance/ are not scores for this version.

Version 0.5.0 adds tested document-scoped and physical-page retrieval, bounded
page ranges, relative page follow-ups, missing-page behavior, source isolation,
and friendly errors for damaged/password-protected PDFs. Tests create real
PDF fixtures with blank and encrypted pages. Model requests remain mocked;
the tests verify supplied evidence and instructions, not live response quality.
