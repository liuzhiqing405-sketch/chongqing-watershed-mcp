"""Provider-specific relay fields must not change intent or execution authority."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from slope_evaluation_agent.config import LLMSettings
from slope_evaluation_agent.llm import OpenAICompatibleClient, LLMRequestFailed, api_endpoint


def settings(provider, **changes):
    return LLMSettings.model_validate({'provider':provider,'base_url':'https://4router.net/v1',
        'model':{'grok':'grok-4.5','gemini':'gemini-3.1-pro-preview','qwen':'qwen3.8-max'}[provider],
        'api_key':'fake-relay-key','temperature':None,'maximum_tokens':4096,'max_retries':0,**changes})


def reply(text='{"ok":true}', **changes):
    return {'model':'served-alias','choices':[{'finish_reason':'stop','message':{'content':text,'reasoning_content':'private-reasoning'}}],**changes}


def response(payload):
    return SimpleNamespace(status_code=200,headers={},json=lambda:payload)


class HTTP:
    def __init__(self,*replies):self.replies=list(replies);self.requests=[]
    def AsyncClient(self,**kwargs):
        outer=self
        class Client:
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,url,headers,json):
                outer.requests.append((url,dict(headers),__import__('json').loads(__import__('json').dumps(json))))
                return outer.replies.pop(0)
        return Client()


@pytest.mark.asyncio
@pytest.mark.parametrize('provider',['grok','gemini','qwen'])
async def test_relay_chat_defaults_use_bearer_and_do_not_leak_other_provider_settings(provider,monkeypatch):
    client=OpenAICompatibleClient(settings(provider))
    http=HTTP(response(reply()));monkeypatch.setattr(client,'_client',lambda:http)
    assert await client.complete_json(system='Return JSON',user='hello')=={'ok':True}
    url,headers,body=http.requests[0]
    assert url=='https://4router.net/v1/chat/completions'
    assert headers['Authorization']=='Bearer fake-relay-key'
    assert 'fake-relay-key' not in json.dumps(body)
    assert 'thinking' not in body and 'output_config' not in body and 'reasoning_effort' not in body
    assert body.get('stream') is (True if provider=='qwen' else None)
    assert client.last_response_model=='served-alias'


@pytest.mark.asyncio
@pytest.mark.parametrize('provider,effort',[('grok','high'),('gemini','low'),('qwen','medium')])
async def test_compatible_effort_is_at_top_level(provider,effort,monkeypatch):
    client=OpenAICompatibleClient(settings(provider,reasoning_effort=effort))
    http=HTTP(response(reply()));monkeypatch.setattr(client,'_client',lambda:http)
    await client.complete_json(system='JSON',user='u')
    assert http.requests[0][2]['reasoning_effort']==effort


@pytest.mark.asyncio
async def test_qwen_explicit_thinking_and_budget_are_rest_fields_not_sdk_wrappers(monkeypatch):
    client=OpenAICompatibleClient(settings('qwen',qwen_thinking='enabled',qwen_thinking_budget_tokens=2048))
    http=HTTP(response(reply()));monkeypatch.setattr(client,'_client',lambda:http)
    await client.complete_json(system='JSON',user='u')
    body=http.requests[0][2]
    assert body['enable_thinking'] is True and body['thinking_budget']==2048 and body['stream'] is True
    assert 'extra_body' not in body and 'thinking' not in body


@pytest.mark.asyncio
async def test_gemini_compatible_budget_uses_google_extra_body(monkeypatch):
    client=OpenAICompatibleClient(settings('gemini',gemini_thinking_budget_tokens=1024))
    http=HTTP(response(reply()));monkeypatch.setattr(client,'_client',lambda:http)
    await client.complete_json(system='JSON',user='u')
    assert http.requests[0][2]['extra_body']=={'google':{'thinking_config':{'thinking_budget':1024}}}
    assert 'reasoning_effort' not in http.requests[0][2]


@pytest.mark.parametrize('provider,options',[
    ('grok',{'reasoning_effort':'none'}),('grok',{'reasoning_effort':'max'}),
    ('qwen',{'reasoning_effort':'low','qwen_thinking_budget_tokens':1024}),
    ('qwen',{'qwen_thinking':'disabled','qwen_thinking_budget_tokens':1024}),
    ('qwen',{'qwen_thinking':'enabled','reasoning_effort':'none'}),
    ('qwen',{'qwen_thinking':'disabled','reasoning_effort':'high'}),
    ('gemini',{'reasoning_effort':'low','gemini_thinking_budget_tokens':1024}),
    ('gemini',{'api_format':'responses'}),('qwen',{'api_format':'responses'}),
])
def test_conflicting_or_unsupported_parameters_are_refused_before_network(provider,options):
    with pytest.raises(ValidationError):settings(provider,**options)


def stream(*events,done=True):
    text=''.join('data: '+json.dumps(e)+'\n\n' for e in events)
    if done:text+='data: [DONE]\n\n'
    return SimpleNamespace(status_code=200,headers={'content-type':'text/event-stream'},text=text)


def chat_events(finish='stop'):
    return [
        {'model':'qwen3.8-max','choices':[{'delta':{'reasoning_content':'private-reasoning'},'finish_reason':None}]},
        {'model':'qwen3.8-max','choices':[{'delta':{'content':'{"ok":'},'finish_reason':None}]},
        {'model':'qwen3.8-max','choices':[{'delta':{'content':'true}'},'finish_reason':None}]},
        {'model':'qwen3.8-max','choices':[{'delta':{},'finish_reason':finish}]},
        {'choices':[],'usage':{'completion_tokens':20}},
    ]


@pytest.mark.asyncio
async def test_qwen_stream_uses_text_only_and_requires_completion(monkeypatch):
    client=OpenAICompatibleClient(settings('qwen'))
    raw=stream(*chat_events())
    assert 'private-reasoning' not in json.dumps(client._chat_payload(raw))
    http=HTTP(raw);monkeypatch.setattr(client,'_client',lambda:http)
    assert await client.complete_json(system='JSON',user='u')=={'ok':True}
    assert client.last_response_model=='qwen3.8-max'


@pytest.mark.parametrize('raw',[stream(*chat_events(),done=False),stream(*chat_events()[:-2])])
def test_json_deltas_alone_never_mean_stream_success(raw):
    with pytest.raises(LLMRequestFailed) as e:OpenAICompatibleClient._chat_payload(raw)
    assert e.value.reason=='response_truncated'


@pytest.mark.parametrize('finish,extra,reason',[
    ('length',{},'response_truncated'),('content_filter',{},'request_failed'),
    ('tool_calls',{'tool_calls':[{'function':{'name':'run'}}]},'unexpected_tool_call'),
])
def test_partial_filtered_and_tool_answers_are_not_a_success(finish,extra,reason):
    payload=reply(choices=[{'finish_reason':finish,'message':{'content':'{"ok":true}',**extra}}])
    with pytest.raises(LLMRequestFailed) as e:OpenAICompatibleClient._content(payload)
    assert e.value.reason==reason


def gemini_reply(finish='STOP',**changes):
    return {'modelVersion':'gemini-3.1-pro-preview','candidates':[{'finishReason':finish,'content':{'parts':[
        {'text':'private-reasoning','thought':True},{'text':'{"ok":true}'}]}}],**changes}


@pytest.mark.asyncio
@pytest.mark.parametrize('effort,budget,expected',[
    ('low',None,{'thinkingLevel':'LOW'}),(None,1024,{'thinkingBudget':1024}),
    ('none',None,{'thinkingBudget':0}),(None,None,None),
])
async def test_gemini_native_body_auth_thought_filter_and_model(effort,budget,expected,monkeypatch):
    client=OpenAICompatibleClient(settings('gemini',api_format='gemini_generate_content',reasoning_effort=effort,
                                         gemini_thinking_budget_tokens=budget))
    http=HTTP(response(gemini_reply()));monkeypatch.setattr(client,'_client',lambda:http)
    assert await client.complete_json(system='SYSTEM',user='USER')=={'ok':True}
    url,headers,body=http.requests[0]
    assert url=='https://4router.net/v1beta/models/gemini-3.1-pro-preview:generateContent'
    assert headers['x-goog-api-key']=='fake-relay-key' and headers['Authorization']=='Bearer fake-relay-key'
    assert body['systemInstruction']=={'parts':[{'text':'SYSTEM'}]}
    assert body['contents']==[{'role':'user','parts':[{'text':'USER'}]}]
    assert body['generationConfig']['responseMimeType']=='application/json'
    assert body['generationConfig'].get('thinkingConfig')==expected
    assert 'response_format' not in body and 'model' not in body and 'max_tokens' not in body
    assert client.last_response_model=='gemini-3.1-pro-preview'


@pytest.mark.asyncio
async def test_gemini_native_truncation_retry_increases_nested_output_budget(monkeypatch):
    client=OpenAICompatibleClient(settings('gemini',api_format='gemini_generate_content',max_retries=1))
    http=HTTP(response(gemini_reply('MAX_TOKENS')),response(gemini_reply()))
    monkeypatch.setattr(client,'_client',lambda:http)
    await client.complete_json(system='JSON',user='u')
    assert [r[2]['generationConfig']['maxOutputTokens'] for r in http.requests]==[4096,8192]


@pytest.mark.parametrize('payload,reason',[
    (gemini_reply('SAFETY'),'request_failed'),
    (gemini_reply(promptFeedback={'blockReason':'SAFETY'}),'request_failed'),
    ({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'secret','thought':True}]}}]},'empty_response'),
    ({'candidates':[{'finishReason':'STOP','content':{'parts':[{'functionCall':{'name':'run'}}]}}]},'unexpected_tool_call'),
])
def test_gemini_blocks_and_tools_cannot_escape_validation(payload,reason):
    with pytest.raises(LLMRequestFailed) as e:OpenAICompatibleClient._gemini_content(payload)
    assert e.value.reason==reason


def test_native_gemini_model_names_cannot_inject_paths_or_queries():
    s=settings('gemini',api_format='gemini_generate_content',model='models/gemini-test/../x?secret=no')
    url=api_endpoint(s,'generateContent')
    assert '?' not in url and '/models/gemini-test%2F..%2Fx%3Fsecret%3Dno:' in url
    assert api_endpoint(s,'models')=='https://4router.net/v1/models'
