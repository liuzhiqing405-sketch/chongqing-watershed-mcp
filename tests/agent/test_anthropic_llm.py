"""Native Claude transport must preserve the agent's JSON and execution contract."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from slope_evaluation_agent.config import LLMSettings
from slope_evaluation_agent.llm import LLMRequestFailed, OpenAICompatibleClient, api_endpoint


def settings(**changes):
    return LLMSettings(provider='anthropic', base_url='https://4router.net/v1',
                       model='claude-sonnet-5', api_key='fake-claude-key', temperature=None,
                       maximum_tokens=4096, max_retries=0, **changes)


def message(text='{"answer":"孔隙水压力会改变有效应力。"}', **changes):
    return {'type':'message','model':'claude-sonnet-5','stop_reason':'end_turn',
            'content':[{'type':'thinking','thinking':'private-thoughts'},
                       {'type':'text','text':text}],**changes}


class Transport:
    def __init__(self, *replies):
        self.replies=list(replies);self.requests=[]
    def AsyncClient(self, **kwargs):
        outer=self
        class Client:
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,url,headers,json):
                outer.requests.append((url,dict(headers),dict(json)))
                return outer.replies.pop(0)
        return Client()


def response(payload):
    return SimpleNamespace(status_code=200,headers={},json=lambda:payload)


@pytest.mark.parametrize('base', ['https://4router.net','https://4router.net/','https://4router.net/v1/'])
def test_native_root_or_versioned_url_has_exactly_one_version_prefix(base):
    s=LLMSettings(provider='anthropic',base_url=base,model='claude-sonnet-5',temperature=None)
    assert s.api_format=='anthropic_messages'
    assert api_endpoint(s,'messages')=='https://4router.net/v1/messages'
    assert api_endpoint(s,'models')=='https://4router.net/v1/models'


@pytest.mark.asyncio
async def test_native_message_auth_roles_and_final_text(monkeypatch):
    client=OpenAICompatibleClient(settings())
    http=Transport(response(message()));monkeypatch.setattr(client,'_client',lambda:http)
    assert await client.complete_json(system='SYSTEM',user='USER')=={'answer':'孔隙水压力会改变有效应力。'}
    url,headers,body=http.requests[0]
    assert url=='https://4router.net/v1/messages'
    assert headers['x-api-key']=='fake-claude-key'
    assert headers['Authorization']=='Bearer fake-claude-key'
    assert headers['anthropic-version']=='2023-06-01'
    assert body['system'].startswith('SYSTEM')
    assert body['messages']==[{'role':'user','content':'USER'}]
    assert body['stream'] is True and body['max_tokens']==4096
    for field in ['response_format','reasoning_effort','reasoning','input','text','thinking','tools','temperature']:
        assert field not in body
    assert 'fake-claude-key' not in json.dumps(body)
    assert client.last_response_model=='claude-sonnet-5'


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,effort', [('adaptive','medium'),('enabled','low'),('disabled',None)])
async def test_native_thinking_and_effort_are_separate_parameters(monkeypatch,mode,effort):
    client=OpenAICompatibleClient(settings(anthropic_thinking=mode,reasoning_effort=effort))
    http=Transport(response(message()));monkeypatch.setattr(client,'_client',lambda:http)
    await client.complete_json(system='JSON',user='u')
    body=http.requests[0][2]
    assert body['thinking']==({'type':'enabled','budget_tokens':1024} if mode=='enabled' else {'type':mode})
    assert body.get('output_config')==({'effort':effort} if effort else None)


@pytest.mark.parametrize('options', [
    {'reasoning_effort':'none'}, {'reasoning_effort':'ultra'},
    {'anthropic_thinking':'enabled','anthropic_thinking_budget_tokens':4096},
    {'anthropic_thinking':'enabled','temperature':0}, {'temperature':1.5},
    {'api_format':'responses'},
])
def test_incompatible_options_are_rejected_before_a_paid_request(options):
    with pytest.raises(ValidationError):
        LLMSettings.model_validate({'provider':'anthropic','base_url':'https://4router.net/v1',
            'model':'claude-sonnet-5','temperature':None,'maximum_tokens':4096,**options})


def sse(*events):
    return SimpleNamespace(status_code=200,headers={'content-type':'text/event-stream'},
                           text=''.join('data: '+json.dumps(event)+'\n\n' for event in events))


def stream_events():
    return [
        {'type':'message_start','message':{'type':'message','model':'claude-opus-5','content':[],'stop_reason':None}},
        {'type':'content_block_start','index':0,'content_block':{'type':'thinking','thinking':''}},
        {'type':'content_block_delta','index':0,'delta':{'type':'thinking_delta','thinking':'private-thoughts'}},
        {'type':'content_block_delta','index':0,'delta':{'type':'signature_delta','signature':'private-signature'}},
        {'type':'content_block_stop','index':0},
        {'type':'ping'},
        {'type':'content_block_start','index':1,'content_block':{'type':'text','text':''}},
        {'type':'content_block_delta','index':1,'delta':{'type':'text_delta','text':'{"ok":'}},
        {'type':'content_block_delta','index':1,'delta':{'type':'text_delta','text':'true}'}},
        {'type':'content_block_stop','index':1},
        {'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'output_tokens':20}},
        {'type':'message_stop'},
    ]


@pytest.mark.asyncio
async def test_completed_stream_joins_only_final_text_and_observes_serving_model(monkeypatch):
    client=OpenAICompatibleClient(settings())
    raw=sse(*stream_events());payload=client._anthropic_payload(raw)
    assert 'private-' not in json.dumps(payload)
    http=Transport(raw);monkeypatch.setattr(client,'_client',lambda:http)
    assert await client.complete_json(system='JSON',user='u')=={'ok':True}
    assert client.last_response_model=='claude-opus-5'


@pytest.mark.parametrize('events', [
    stream_events()[:-1],
    [e for e in stream_events() if not (e['type']=='content_block_stop' and e.get('index')==1)],
    [e for e in stream_events() if e['type']!='message_delta'],
])
def test_a_valid_json_delta_is_not_a_complete_answer_without_terminal_events(events):
    with pytest.raises(LLMRequestFailed) as error:
        OpenAICompatibleClient._anthropic_payload(sse(*events))
    assert error.value.reason=='response_truncated'


def test_stream_error_never_exposes_provider_body_or_credential():
    with pytest.raises(LLMRequestFailed) as error:
        OpenAICompatibleClient._anthropic_payload(sse({'type':'error','error':{'message':'fake-claude-key private-thoughts'}}))
    assert 'fake-claude-key' not in str(error.value) and 'private-thoughts' not in str(error.value)


@pytest.mark.parametrize('stop,blocks,reason', [
    ('max_tokens',[{'type':'text','text':'{"ok":true}'}],'response_truncated'),
    ('end_turn',[{'type':'thinking','thinking':'{"ok":true}'}],'empty_response'),
    ('tool_use',[{'type':'tool_use','name':'run','input':{'unsafe':True}}],'unexpected_tool_call'),
    ('refusal',[{'type':'text','text':'{"ok":true}'}],'request_failed'),
])
def test_non_answers_and_tools_cannot_be_used_as_json_success(stop,blocks,reason):
    with pytest.raises(LLMRequestFailed) as error:
        OpenAICompatibleClient._anthropic_content(message(stop_reason=stop,content=blocks))
    assert error.value.reason==reason


@pytest.mark.asyncio
async def test_truncation_retry_raises_native_budget_without_openai_fields(monkeypatch):
    s=settings();s.max_retries=1
    client=OpenAICompatibleClient(s)
    http=Transport(response(message(stop_reason='max_tokens')),response(message()))
    monkeypatch.setattr(client,'_client',lambda:http)
    await client.complete_json(system='JSON',user='u')
    assert [r[2]['max_tokens'] for r in http.requests]==[4096,8192]
    assert all('max_output_tokens' not in r[2] for r in http.requests)
