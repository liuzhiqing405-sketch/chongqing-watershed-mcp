"""Exercise browser state transitions without deleting user conversations."""
import json
import shutil
import subprocess

import pytest

from slope_evaluation_agent.web import STATIC_ROOT


@pytest.mark.parametrize('scenario',['success','partial','cancel','network','old_backend'])
def test_bulk_history_response_keeps_the_correct_selection_and_current_session(scenario):
    node=shutil.which('node')
    if not node:
        pytest.skip('Node.js is unavailable for the isolated frontend state test')
    script=(STATIC_ROOT/'app.js').read_text(encoding='utf-8')
    start=script.index('async function deleteSelectedHistory()')
    end=script.index('\nasync function deleteHistory(',start)
    function=script[start:end]
    harness=r'''
const vm=require('node:vm');
const scenario=SCENARIO;
const context={scenario};vm.createContext(context);
vm.runInContext(`
let sessionId='a',busy=false,historyDeleting=false,historyManaging=true;
let historyItems=[{session_id:'a',title:'Current'},{session_id:'b',title:'Other'},{session_id:'c',title:'Keep'}];
const selectedHistory=new Set(['a','b']);
const sendButton={disabled:false},storageKey='session';
let calls=0,created=0,cleared=0,status='',errorStatus=false;
let serverIds=['a','b','c'];
const localStorage={removeItem(){cleared++;}},conversation={replaceChildren(){}};
function stopPoll(){} function renderHistory(){} function renderTaskWorkspace(){} function message(){}
function historyStatus(text,error=false){status=text;errorStatus=error;}
function confirm(){return scenario!=='cancel';}
async function createSession(){created++;sessionId='new';serverIds.push('new');await loadHistory();}
async function loadHistory(){
 historyItems=serverIds.map(id=>({session_id:id,title:id}));
 for(const id of selectedHistory)if(!serverIds.includes(id))selectedHistory.delete(id);
 return true;
}
async function api(url,options){
 if(url.endsWith('/health'))return {batch_session_product_cleanup:scenario!=='old_backend'};
 calls++;
 const ids=JSON.parse(options.body).session_ids;
 if(ids.join(',')!=='a,b')throw Error('Unexpected selection');
 if(scenario==='network'){serverIds=['c'];throw Error('Connection interrupted');}
 if(scenario==='partial'){
   serverIds=['a','c'];
   return {deleted:['b'],failed:[{session_id:'a',detail:'Task is running'}],cleanup:{retained_ids:['shared']}};
 }
 serverIds=['c'];return {deleted:['a','b'],failed:[],cleanup:{retained_ids:[]}};
}
`,context);
vm.runInContext(FUNCTION,context);
(async()=>{
 await vm.runInContext('deleteSelectedHistory()',context);
 process.stdout.write(vm.runInContext(`JSON.stringify({sessionId,busy,historyDeleting,historyManaging,
  selected:[...selectedHistory],visible:historyItems.map(r=>r.session_id),calls,created,cleared,status,errorStatus,
  sendDisabled:sendButton.disabled})`,context));
})().catch(e=>{process.stderr.write(String(e));process.exitCode=1;});
'''
    harness=harness.replace('SCENARIO',json.dumps(scenario)).replace('FUNCTION',json.dumps(function))
    result=subprocess.run([node,'-'],input=harness,encoding='utf-8',capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr
    state=json.loads(result.stdout)
    assert not state['busy'] and not state['historyDeleting'] and not state['sendDisabled']
    assert 'c' in state['visible']
    if scenario in {'success','network'}:
        assert state['sessionId']=='new' and state['created']==state['cleared']==1
        assert 'a' not in state['visible'] and 'b' not in state['visible']
        assert not state['selected'] and state['calls']==1
        assert state['errorStatus']==(scenario=='network')
    elif scenario=='partial':
        assert state['sessionId']=='a' and state['created']==0
        assert state['selected']==['a'] and 'b' not in state['visible']
        assert state['historyManaging'] and state['errorStatus'] and 'Task is running' in state['status']
    else:
        assert state['calls']==state['created']==state['cleared']==0
        assert state['selected']==['a','b'] and state['visible']==['a','b','c']
        assert state['errorStatus']==(scenario=='old_backend')
