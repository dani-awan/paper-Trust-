"""Explicit Groq or Azure endpoint adapter. No OpenAI Platform fallback."""
import json
import os
from urllib.parse import urlsplit
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from backend.config import model_name, provider, provider_label, missing_settings, setting


class StructuredCall:
    def __init__(self, client, schema):
        self.client, self.schema = client, schema

    def invoke(self, messages):
        instructions = 'Return only a JSON object matching this JSON schema. No markdown fences.\n' + json.dumps(self.schema.model_json_schema())
        response = self.client.bind(response_format={'type': 'json_object'}).invoke([SystemMessage(content=instructions), *messages])
        return self.schema.model_validate_json(response.content)


class CloudChat:
    def __init__(self, client):
        self.client = client

    def invoke(self, messages):
        return self.client.invoke(messages)

    def stream(self, messages):
        return self.client.stream(messages)

    def with_structured_output(self, schema):
        return StructuredCall(self.client, schema)


def azure_base_url(endpoint):
    parsed = urlsplit(endpoint.strip())
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.port not in (None, 443):
        raise ValueError('Use the HTTPS endpoint from your Azure resource, without credentials or query parameters.')
    host = parsed.hostname or ''
    if not any(host.endswith('.' + suffix) for suffix in ('openai.azure.com', 'services.ai.azure.com', 'cognitiveservices.azure.com')):
        raise ValueError('Use your Azure OpenAI/Foundry resource endpoint, not the portal or project URL.')
    path = parsed.path.rstrip('/')
    if path not in ('', '/openai/v1'):
        raise ValueError('Use the resource root endpoint or its /openai/v1/ endpoint, not a project URL.')
    return f'https://{host}/openai/v1/'


def get_llm():
    if missing_settings():
        raise ValueError('Missing configuration: ' + ', '.join(missing_settings()))
    options = dict(model=model_name(), use_responses_api=False, timeout=45, max_retries=0, max_tokens=2048)
    if provider() == 'groq':
        options.update(api_key=setting('GROQ_API_KEY'), base_url='https://api.groq.com/openai/v1', reasoning_effort='low')
    elif provider() == 'azure':
        options.update(api_key=setting('AZURE_OPENAI_API_KEY'), base_url=azure_base_url(setting('AZURE_OPENAI_ENDPOINT')))
    else:
        raise ValueError('Local mode does not call an AI service.')
    return CloudChat(ChatOpenAI(**options))


def friendly_error(error):
    label = provider_label()
    status = getattr(error, 'status_code', None)
    if status == 429:
        return f'{label} usage/quota limit reached. Check account quota or credits, or wait for the rate limit to reset.'
    if status in (401, 403):
        return f'{label} rejected the key or deployment access. Check the configured key, endpoint and permissions, then restart.'
    if status == 404:
        return f'{label} model/deployment was not found. Check the model or deployment name in .env.'
    if status == 413:
        return 'The request is too large. Start a new chat or ask a shorter question.'
    return f'{label} could not finish the AI request. Check connectivity and configuration, or retry later.'


def check_connection():
    """An explicit UI action makes one small call to the chosen provider."""
    response = get_llm().invoke([HumanMessage(content='Reply with only READY. Do not use tools.')])
    if not str(response.content).strip():
        raise ValueError('The provider returned an empty response. Check the model and try again.')
    return True
