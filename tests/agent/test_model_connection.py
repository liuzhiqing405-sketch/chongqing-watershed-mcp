"""Model switching must change real requests, preserve gates, and retain no keys on disk."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from slope_evaluation_agent.config import AgentSettings, LLMSettings
from slope_evaluation_agent.llm import LLMRequestFailed, OpenAICompatibleClient
from slope_evaluation_agent.model_connection import ConnectionRequest, candidate_settings
from slope_evaluation_agent.session_store import SessionStore
from slope_evaluation_agent.web import create_app


class IdleAgent:
    def __init__(self, settings):
        self.settings = settings
        self.store = SessionStore(settings.sessions)
        self.llm = None
        for name in ('router','parser','responder','rewriter','locator'):
            setattr(self, name, SimpleNamespace(llm=None))
    async def open(self): pass
    async def close(self): pass


@pytest.fixture
def web(tmp_path):
    settings = AgentSettings.model_validate({'sessions':{'root':str(tmp_path/'isolated')},
        'llm':{'base_url':'https://old.example.invalid/v1','model':'old-model','api_key':'fake-original-credential'}})
    agent = IdleAgent(settings)
    with TestClient(create_app(settings=settings,agent=agent)) as client:
        token = client.get('/api/v1/model-connection').json()['form_token']
        yield client,agent,{'X-Model-Connection-Token':token}


def body(**changes):
    return {'base_url':'https://relay.example.invalid/v1','model':'another-model','provider':'openai-compatible',
            'api_format':'responses','reasoning_effort':'high','temperature':None,**changes}


@pytest.mark.parametrize('provider,api_format', [('openai-compatible','responses'),('anthropic','anthropic_messages'),
    ('grok','responses'),('gemini','chat_completions'),('gemini','gemini_generate_content'),('qwen','chat_completions')])
def test_switch_rewires_all_clients_without_a_conversation_or_credential_echo(web,provider,api_format):
    client,agent,headers=web
    response=client.post('/api/v1/model-connection',json=body(api_key='fake-new-credential',provider=provider,api_format=api_format),headers=headers)
    assert response.status_code==200
    data=response.json()
    assert data['model']=='another-model' and data['reasoning_effort']=='high' and data['saved_in_memory_only']
    assert 'fake-new-credential' not in response.text
    assert agent.settings.llm.api_key.get_secret_value()=='fake-new-credential'
    assert all(getattr(agent,n).llm is agent.llm for n in ('router','parser','rewriter','responder','locator'))
    assert not list(agent.store.root.rglob('session.json')) if hasattr(agent.store,'root') else not agent.store.list_sessions()
    health=client.get('/api/v1/health')
    assert health.json()['llm']['model']=='another-model'
    assert health.json()['llm']['api_format']==api_format
    assert 'credential' not in health.text and 'api_key' not in health.text


@pytest.mark.parametrize('endpoint,retain,expected',[
    ('https://old.example.invalid/v1',True,'fake-original-credential'),
    ('https://other.example.invalid/v1',True,None),
    ('https://old.example.invalid/v1',False,None)])
def test_key_is_reused_only_at_same_endpoint(endpoint,retain,expected):
    current=LLMSettings(base_url='https://old.example.invalid/v1',model='old',api_key='fake-original-credential')
    settings=candidate_settings(ConnectionRequest(**body(base_url=endpoint,retain_key=retain)),current)
    assert (settings.api_key.get_secret_value() if settings.api_key else None)==expected


def test_origin_token_and_validation_failures_do_not_echo_secrets_or_change_model(web):
    client,agent,headers=web
    original=agent.settings.llm
    for request_headers, payload in [({},body()),({**headers,'Origin':'https://unrelated.invalid'},body()),
        (headers,body(base_url='https://example.invalid/?secret=fake-private-value',api_key='fake-private-value'))]:
        response=client.post('/api/v1/model-connection',json=payload,headers=request_headers)
        assert response.status_code in {400,403}
        assert 'fake-private-value' not in response.text
        assert agent.settings.llm is original


def test_untrusted_host_cannot_read_connection_token(web):
    client,_,_=web
    response=client.get('/api/v1/model-connection',headers={'Host':'unrelated.invalid'})
    assert response.status_code==403 and 'form_token' not in response.text


def test_inflight_task_refuses_switch(web):
    client,agent,headers=web
    lock=asyncio.Lock();client.app.state.runtime.locks['active']=lock
    client.portal.call(lock.acquire)
    try:
        response=client.post('/api/v1/model-connection',json=body(),headers=headers)
        assert response.status_code==409 and agent.settings.llm.model=='old-model'
    finally:client.portal.call(lock.release)


def test_json_probe_uses_candidate_only_without_applying_or_saving_history(web,monkeypatch):
    client,agent,headers=web
    seen=[]
    class Probe:
        last_response_model='actual-relay-model'
        def __init__(self,settings):seen.append(settings)
        async def complete_json(self,**kwargs):return {'ok':True}
    monkeypatch.setattr('slope_evaluation_agent.model_connection.OpenAICompatibleClient',Probe)
    response=client.post('/api/v1/model-connection/check',json=body(api_key='fake-probe-key'),headers=headers)
    assert response.status_code==200 and response.json()['response_model']=='actual-relay-model'
    assert seen[0].reasoning_effort=='high' and seen[0].api_format=='responses'
    assert agent.settings.llm.model=='old-model'


@pytest.mark.parametrize('failure', [None, 'greeting', 'engineering_question', 'invalid_schema'])
def test_connection_and_model_routing_are_checked_separately_without_history(web,monkeypatch,failure):
    client,agent,headers=web
    calls=[]
    original=agent.settings.llm
    before=list(agent.store.list_sessions())
    class Probe:
        last_response_model='candidate-response-model'
        def __init__(self,settings):pass
        async def complete_json(self,*,system,user):
            calls.append((system,user))
            if len(calls)==1:return {'ok':True}
            message=json.loads(user)['user_text']
            case='greeting' if message=='你好' else 'engineering_question'
            if failure=='invalid_schema':return {'ok':True}
            return {'route':'out_of_scope' if case==failure else 'knowledge_chat',
                    'intent':'unknown' if case==failure else 'ask_explanation',
                    'confidence':0.99,'candidates':[],'missing_fields':[],
                    'topic':None if case==failure else ('greeting' if case=='greeting' else 'engineering_general'),
                    'ambiguity':None}
    monkeypatch.setattr('slope_evaluation_agent.model_connection.OpenAICompatibleClient',Probe)
    response=client.post('/api/v1/model-connection/check',json=body(api_key='fake-probe-key'),headers=headers)
    assert response.status_code==200 and response.json()['ok'] is True
    data=response.json()
    assert data['workflow_compatible'] is (failure is None)
    assert data['compatibility_scope']=='basic_intent_routing_only'
    if failure is None:
        assert len(calls)==3 and all(check['source']=='model' for check in data['routing_checks'])
    else:
        failed=data['routing_checks'][-1]
        assert failed['passed'] is False
        assert failed['name']==('greeting' if failure=='invalid_schema' else failure)
        if failure=='invalid_schema':
            # The built-in greeting works after model schema failure, but it
            # must never be reported as proof that this model understood it.
            assert failed['source']=='model_fallback' and failed['route']=='knowledge_chat'
            assert failed['reason']=='invalid_route_schema'
    assert agent.settings.llm is original
    assert agent.store.list_sessions()==before
    assert 'fake-probe-key' not in response.text


def test_unavailable_model_reports_observed_503_without_claiming_a_bad_key(web,monkeypatch):
    client,agent,headers=web
    class Probe:
        def __init__(self,settings):pass
        async def complete_json(self,**kwargs):
            raise LLMRequestFailed('remote echoed fake-private-key',reason='http_503')
    monkeypatch.setattr('slope_evaluation_agent.model_connection.OpenAICompatibleClient',Probe)
    response=client.post('/api/v1/model-connection/check',json=body(),headers=headers)
    assert response.status_code==400
    assert '503' in response.json()['detail'] and '仅凭状态码不能确定原因' in response.json()['detail']
    assert 'fake-private-key' not in response.text
    assert '密钥无效' not in response.text and agent.settings.llm.model=='old-model'


def test_model_list_uses_entered_endpoint_and_never_sends_old_key_to_it(web,monkeypatch):
    client,agent,headers=web
    requests=[]
    class HTTP:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url,**kw):
            requests.append((url,kw))
            return SimpleNamespace(status_code=200,json=lambda:{'data':[{'id':'model-b'},{'id':'model-a'}]})
    monkeypatch.setattr('httpx.AsyncClient',HTTP)
    response=client.post('/api/v1/model-connection/models',json=body(model=''),headers=headers)
    assert response.status_code==200 and response.json()['models']==['model-a','model-b']
    assert requests==[('https://relay.example.invalid/v1/models',{'headers':{}})]


def test_claude_list_and_compatibility_probe_use_native_transport_without_session_writes(web,monkeypatch):
    client,agent,headers=web
    calls=[]
    class HTTP:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url,headers):
            calls.append(('GET',url,headers,None))
            return SimpleNamespace(status_code=200,json=lambda:{'data':[{'id':'claude-sonnet-5'}]})
        async def post(self,url,headers,json):
            calls.append(('POST',url,headers,json))
            try:
                prompt=__import__('json').loads(json['messages'][0]['content'])
            except ValueError:prompt={}
            text=prompt.get('user_text')
            result={'ok':True} if text is None else {
                'route':'knowledge_chat','intent':'ask_explanation','confidence':0.99,
                'candidates':[],'missing_fields':[],'ambiguity':None,
                'topic':'greeting' if text=='你好' else 'engineering_general'}
            payload={'type':'message','model':'claude-sonnet-5','stop_reason':'end_turn',
                     'content':[{'type':'text','text':__import__('json').dumps(result)}]}
            return SimpleNamespace(status_code=200,headers={},json=lambda:payload)
    monkeypatch.setattr('httpx.AsyncClient',HTTP)
    request=body(provider='anthropic',api_format='anthropic_messages',base_url='https://claude.example.invalid',
                 model='claude-sonnet-5',api_key='fake-claude-probe',reasoning_effort=None)
    before=agent.store.list_sessions()
    listed=client.post('/api/v1/model-connection/models',json=request,headers=headers)
    assert listed.status_code==200 and listed.json()['models']==['claude-sonnet-5']
    checked=client.post('/api/v1/model-connection/check',json=request,headers=headers)
    assert checked.status_code==200 and checked.json()['workflow_compatible'] is True
    assert len(calls)==4
    assert calls[0][1]=='https://claude.example.invalid/v1/models'
    assert all(item[1]=='https://claude.example.invalid/v1/messages' for item in calls[1:])
    assert all(item[2]['x-api-key']=='fake-claude-probe' for item in calls)
    assert all('Authorization' not in item[2] for item in calls)
    assert agent.settings.llm.model=='old-model' and agent.store.list_sessions()==before
    assert 'fake-claude-probe' not in checked.text


class Transport:
    def __init__(self, replies):self.replies=list(replies);self.requests=[]
    def AsyncClient(self,**kwargs):
        outer=self
        class Client:
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,url,headers,json):
                outer.requests.append((url,dict(json)))
                return outer.replies.pop(0)
        return Client()


def response(content='{"ok":true}',**updates):
    data={'status':'completed','model':'actual-snapshot','output':[
        {'type':'reasoning','summary':[]}, {'type':'message','content':[{'type':'output_text','text':content}]}],**updates}
    return SimpleNamespace(status_code=200,headers={},json=lambda:data)


@pytest.mark.asyncio
async def test_responses_parameters_and_model_identification(monkeypatch):
    settings=LLMSettings(base_url='https://relay.invalid/v1',model='chosen-alias',api_format='responses',
                         reasoning_effort='high',temperature=None)
    client=OpenAICompatibleClient(settings);http=Transport([response()]);monkeypatch.setattr(client,'_client',lambda:http)
    assert await client.complete_json(system='JSON',user='hello')=={'ok':True}
    url,data=http.requests[0]
    assert url.endswith('/responses') and data['reasoning']=={'effort':'high'}
    assert data['max_output_tokens']==1200 and data['store'] is False and data['stream'] is True
    assert 'temperature' not in data and 'messages' not in data and 'thinking' not in data
    assert data['text']=={'format':{'type':'json_object'}}
    assert client.last_response_model=='actual-snapshot'


@pytest.mark.asyncio
async def test_incomplete_responses_retry_uses_output_token_field(monkeypatch):
    client=OpenAICompatibleClient(LLMSettings(base_url='https://relay.invalid/v1',model='m',api_format='responses',max_retries=1))
    http=Transport([response(status='incomplete'),response()]);monkeypatch.setattr(client,'_client',lambda:http)
    assert await client.complete_json(system='JSON',user='u')=={'ok':True}
    assert [x[1]['max_output_tokens'] for x in http.requests]==[1200,2400]


@pytest.mark.asyncio
async def test_completed_sse_is_parsed_without_using_reasoning_deltas(monkeypatch):
    data=response().json()
    stream='event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"not complete"}\n\n'
    stream+='event: response.completed\ndata: '+json.dumps({'type':'response.completed','response':data})+'\n\n'
    r=SimpleNamespace(status_code=200,headers={'content-type':'text/event-stream'},text=stream)
    client=OpenAICompatibleClient(LLMSettings(base_url='https://relay.invalid/v1',model='m',api_format='responses'))
    http=Transport([r]);monkeypatch.setattr(client,'_client',lambda:http)
    assert await client.complete_json(system='JSON',user='u')=={'ok':True}


def test_unterminated_stream_is_never_accepted_as_completed():
    r=SimpleNamespace(headers={'content-type':'text/event-stream'},text='data: {"type":"response.output_text.delta","delta":"{}"}\n\n')
    with pytest.raises(LLMRequestFailed,match='未正常结束') as exc:OpenAICompatibleClient._payload(r)
    assert exc.value.reason=='response_truncated'


@pytest.mark.asyncio
@pytest.mark.parametrize('api_format,effort,expected', [('chat_completions','high','enabled'),('chat_completions','none','disabled'),('responses',None,'none')])
async def test_deepseek_switch_and_effort_do_not_conflict(monkeypatch,api_format,effort,expected):
    r=response() if api_format=='responses' else SimpleNamespace(status_code=200,headers={},json=lambda:{'choices':[{'message':{'content':'{"ok":true}'}}]})
    client=OpenAICompatibleClient(LLMSettings(provider='deepseek',base_url='https://api.deepseek.com',model='deepseek-flash',api_key='fake-key',api_format=api_format,reasoning_effort=effort))
    http=Transport([r]);monkeypatch.setattr(client,'_client',lambda:http)
    await client.complete_json(system='JSON',user='u');data=http.requests[0][1]
    assert data.get('thinking',{}).get('type')==expected if api_format=='chat_completions' else data['reasoning']['effort']==expected
