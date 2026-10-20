"""Environment defaults plus isolated, per-session connection settings."""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
APP_NAME = 'ResearchLens'
CREDITS = 'Muhammad Danial & Muhammad Ahmad'
DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_DOCUMENT_CHARS = 2_000_000
_connection = ContextVar('researchlens_connection', default=None)


@contextmanager
def connection_scope(settings=None):
    token = _connection.set(dict(settings) if settings is not None else None)
    try:
        yield
    finally:
        _connection.reset(token)


def setting(name, default=''):
    settings = _connection.get()
    if settings is not None and name in settings:
        return settings[name]
    return os.getenv(name, default)


def provider():
    return setting('AI_PROVIDER', 'local').strip().lower()


def provider_label():
    return {'local':'Not connected','groq':'Groq','azure':'Azure'}.get(provider(), 'Unknown provider')


def missing_settings():
    required = {'local':(), 'groq':('GROQ_API_KEY',), 'azure':('AZURE_OPENAI_API_KEY','AZURE_OPENAI_ENDPOINT','AZURE_OPENAI_DEPLOYMENT')}
    if provider() not in required:
        return ['AI_PROVIDER must be local, groq or azure']
    return [key for key in required[provider()] if not str(setting(key)).strip()]


def ai_enabled():
    return provider() in ('groq','azure') and not missing_settings()


def model_name():
    return setting('AZURE_OPENAI_DEPLOYMENT') if provider() == 'azure' else setting('GROQ_MODEL','openai/gpt-oss-20b')

MODEL = model_name()
