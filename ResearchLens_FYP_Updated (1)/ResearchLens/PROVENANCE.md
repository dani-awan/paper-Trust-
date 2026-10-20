# Source and contribution record

ResearchLens adapts the user-supplied archive `Paper_papeer-main.zip`, whose
original README identifies the application as **Papeer — Research Paper Assistant**.
The ZIP archive comment identifies snapshot `c774bbd60043e26cbe82fab3e03ac41bfcd55cf7`.
Original author identity and repository URL were not established from the supplied files.

Muhammad Danial and Muhammad Ahmad are credited for customization and improvements
in this edition. This credit does not assert authorship of the original supplied code.
Original project documentation, evaluation evidence and selected historical files
are retained in `provenance/` for traceability. No license file was present in the
supplied archive; this edition does not invent a license grant for the original code.

Third-party Python libraries retain their respective licenses. The bundled
DejaVu Sans font retains its license in `assets/FONT_LICENSE.txt`.
The supplied Openclaw research report and synthetic evaluation questions are
sample materials inherited from the archive, not publications authored by the
customization contributors.

Version 0.3 replaces embedding-based indexing with local SQLite FTS search and
adds optional Groq/Azure inference. It does not reuse the historical evaluation
scores as claims about local retrieval or cloud answer accuracy.

Version 0.4 adds the English chat-first interface, session-scoped Groq/Azure
connection checks, streaming answers, an intent-aware retrieval layer and a
two-node conversational LangGraph. The legacy source-search tool remains
separate from written AI answers. Tests do not claim live model accuracy.
