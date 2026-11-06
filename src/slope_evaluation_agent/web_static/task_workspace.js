/* Helpers are loaded before app.js; task state is read only when invoked. */
let draftTimer = null;
let revisionChoice = null;
let revisionChoiceSession = null;
function areaGeometry() { return {points:area.points.map(p=>[...p]),closed:area.closed}; }
function areaRemember() { area.undo.push(areaGeometry()); if(area.undo.length>100) area.undo.shift(); area.redo=[]; }
function areaZoom(value) { area.zoom=Math.max(1,Math.min(4,value)); $('#area-stage').style.width=`${area.zoom*100}%`; $('#area-zoom-label').textContent=`${Math.round(area.zoom*100)}%`; }
function eventToMapPoint(event) {
  if (!area.geo) return null;
  const rect=$('#area-map').getBoundingClientRect(); if(!rect.width||!rect.height) return null;
  const [left,bottom,width,height]=area.geo.axes_rect;
  const x=((event.clientX-rect.left)/rect.width-left)/width;
  const y=(1-(event.clientY-rect.top)/rect.height-bottom)/height;
  if(x<0||x>1||y<0||y>1) return null;
  const [w,s,e,n]=area.geo.bbox_wgs84; return [w+x*(e-w),s+y*(n-s)];
}
function saveAreaDraft() {
  clearTimeout(draftTimer);
  if(!area.context?.map_id || !sessionId || busy) return;
  const identifier=sessionId, mapId=area.context.map_id;
  const body={map_id:mapId,coordinates:area.points.map(p=>[...p]),closed:area.closed,assessment_mode:$('#area-assessment-mode').value};
  $('#area-draft-status').textContent='正在保存边界草稿…';
  draftTimer=setTimeout(async()=>{
    try {
      const data=await api(`/api/v1/sessions/${encodeURIComponent(identifier)}/task/draft`,{
        method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      if(identifier!==sessionId||mapId!==area.context?.map_id) return;
      updateSession(data.session); $('#area-draft-status').textContent='草稿已随当前任务保存，刷新后可继续编辑。';
    } catch(error) { if(identifier===sessionId) $('#area-draft-status').textContent=`草稿未保存：${error.message}`; }
  },500);
}
function drawDiagnosticMarkers(svg) {
  if(!area.geo) return;
  const hydro=area.qa?.hydrologic_check||{};
  const [w,s,e,n]=area.geo.bbox_wgs84, [left,bottom,width,height]=area.geo.axes_rect;
  for(const [kind,points] of [['outlet',hydro.outlets||[]],['inlet',hydro.inlets||[]]]) {
    for(const point of points) {
      const lon=point.longitude??point.lon, lat=point.latitude??point.lat;
      if(!Number.isFinite(lon)||!Number.isFinite(lat)||lon<w||lon>e||lat<s||lat>n) continue;
      const x=(left+(lon-w)/(e-w)*width)*1000,y=(1-bottom-(lat-s)/(n-s)*height)*1000;
      const marker=document.createElementNS('http://www.w3.org/2000/svg','polygon');
      const vertices=kind==='inlet'?[[0,-9],[-9,8],[9,8]]:
        Array.from({length:10},(_,i)=>{const a=-Math.PI/2+i*Math.PI/5,r=i%2?4:10;return [Math.cos(a)*r,Math.sin(a)*r];});
      marker.setAttribute('points',vertices.map(([dx,dy])=>`${x+dx},${y+dy}`).join(' '));
      marker.setAttribute('fill',kind==='inlet'?'#d11b20':'#111');marker.setAttribute('stroke','#fff');
      marker.setAttribute('stroke-width','2');marker.classList.add('diagnostic-marker');
      const explanation=`${kind==='inlet'?'框外上游入流点：存在从评价区域外进入的汇水。':'推测出流点：DEM 流向在此离开评价区域。'} ${lon.toFixed(5)}°E, ${lat.toFixed(5)}°N。需结合河网与地形核对。`;
      const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=explanation;marker.append(title);
      marker.addEventListener('click',event=>{event.stopPropagation();$('#area-diagnostic').textContent=explanation;});svg.append(marker);
    }
  }
}
const taskStageNames={sync:'同步输入',validate:'校验输入',delineate:'划分子汇水区',rainfall:'准备降雨',
  build:'构建模型',rainfall_plot:'降雨图件',swmm:'水文计算',safety:'坡面稳定性计算',
  analysis:'参数与时序分析',export:'导出成果',stability:'坡面稳定性计算',qa:'质量检查',assessment:'形成评价'};
function renderTaskWorkspace() {
  const workspace=taskWorkspace||{}, actions=workspace.actions||{};
  $('#cancel-run').disabled=busy;
  for(const [id,key] of [['task-edit-location','edit_location'],['task-edit-boundary','edit_boundary'],
    ['task-edit-rainfall','edit_rainfall'],['task-edit-parameters','edit_parameters'],['task-confirm','confirm'],
    ['task-prepare','prepare'],['task-plan','plan'],['task-start','start'],['task-refresh','refresh']])
    $('#'+id).disabled=busy||!actions[key];
  $('#task-restore').disabled=busy||!actions.restore||!$('#task-revision').value;
  $('#task-explanation').textContent=workspace.explanation||'任务输入、边界草稿和版本会自动保存。';
  // Boundary checks remain in task evidence and map editing, available on
  // request. They are not repeated as a permanent sidebar report.
  const run=workspace.run_detail||{}, submitted=workspace.background?.run?.submitted;
  const statuses={queued:'排队中',running:'运行中',succeeded:'计算完成',qa_failed:'计算完成，质量未通过',failed:'计算失败',cancelled:'已取消',interrupted:'已中断'};
  const status=run.status||workspace.background?.run?.status;
  let explanation=submitted?(statuses[status]||'已提交，需刷新确认真实状态'):'尚未提交计算任务';
  if(run.current_stage||run.stage) explanation+=` · ${taskStageNames[run.current_stage||run.stage]||run.current_stage||run.stage}`;
  if(Number.isFinite(run.progress_percent)) explanation+=` · ${run.progress_percent.toFixed(1)}%（后台报告）`;
  $('#task-run-explanation').textContent=explanation;
  const list=$('#task-stages');list.replaceChildren();
  const completed=new Set(run.completed_stages||[]);
  for(const stage of run.stages||[]) {
    const name=typeof stage==='string'?stage:stage.stage_id||stage.name||stage.stage||'';
    const row=document.createElement('li');row.textContent=`${taskStageNames[name]||name}：${completed.has(name)?'已完成':name===(run.current_stage||run.stage)?(['running','queued'].includes(status)?'处理中':'停止于此阶段'):'待处理'}`;list.append(row);
  }
  const error=run.error||(run.qa_errors?.length?run.qa_errors.join('\n'):null)||run.errors||workspace.last_error;
  $('#task-error').textContent=error?(typeof error==='string'?error:JSON.stringify(error)):'';
}
async function loadTaskRevisions() {
  if(!sessionId) return;const identifier=sessionId;
  try {
    const data=await api(`/api/v1/sessions/${encodeURIComponent(identifier)}/task`);
    if(identifier!==sessionId) return;
    if(revisionChoiceSession!==identifier) {revisionChoice=null;revisionChoiceSession=identifier;}
    const select=$('#task-revision'), selected=revisionChoice;select.replaceChildren();
    for(const row of data.revisions||[]) {const option=document.createElement('option');option.value=row.revision_id;
      option.textContent=`${row.label} · ${row.location||'位置未定'} · ${new Date(row.at).toLocaleString()}`;select.append(option);}
    if(!select.options.length) {const option=document.createElement('option');option.value='';option.textContent='尚无历史版本';select.append(option);}
    if([...select.options].some(o=>o.value===selected)) select.value=selected;
    renderTaskWorkspace();
  } catch(error) { $('#task-explanation').textContent=`版本读取失败：${error.message}`; }
}
async function taskMutation(action,body={}) {
  if(busy||!sessionId) return null;busy=true;sendButton.disabled=true;renderTaskWorkspace();
  try {
    const data=await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/task/${action}`,{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    renderReply(data.reply);updateSession(data.session);await loadTaskRevisions();return data;
  } catch(error) {message(error.message,'assistant');$('#task-settings-error').textContent=error.message;return null;}
  finally {busy=false;sendButton.disabled=false;renderTaskWorkspace();}
}
function openSavedDrawing(qa=null) {
  const location=taskWorkspace?.location||{}, map=location.map;
  if(!map?.url) {areaShow();return;}
  areaApply({...location,administrative_area:taskWorkspace.background.location.administrative_area,map});
  const draft=taskWorkspace.draft||{}, points=draft.coordinates??taskWorkspace.coordinates??[];
  $('#area-assessment-mode').value=draft.assessment_mode||taskWorkspace.assessment_mode||'regional_screening';areaPolicyNote();
  area.points=points.map(p=>[...p]);
  if(area.points.length>1&&area.points[0][0]===area.points.at(-1)[0]&&area.points[0][1]===area.points.at(-1)[1]) area.points.pop();
  area.closed=draft.coordinates!==undefined?draft.closed:area.points.length>=3;
  area.qa=qa||taskWorkspace.spatial_qa;areaDraw();
  areaHint('可拖动顶点修改边界，修改后重新提交检查。诊断点依据上次已检查的边界。');
}
function showTaskSettings(scope) {
  taskSettingsScope=scope;$('#task-settings').hidden=false;$('#task-settings-error').textContent='';
  for(const id of ['task-rainfall-label','task-rainfall','task-station-count-label','task-station-count']) $('#'+id).hidden=scope!=='rainfall';
  for(const id of ['task-preset-label','task-preset','task-overrides-label','task-overrides']) $('#'+id).hidden=scope!=='parameters';
  const rain=taskWorkspace?.rainfall||{};
  $('#task-rainfall').value=rain.date||(rain.start_datetime?`${rain.start_datetime} ~ ${rain.end_datetime}`:rain.mode==='historical_max'?'历史最大日':'');
  $('#task-station-count').value=rain.station_count??'';
  $('#task-preset').value=taskWorkspace?.parameters?.preset||'medium';
  $('#task-overrides').value=Object.entries(taskWorkspace?.parameters?.overrides||{}).map(([key,value])=>`${key}=${value}`).join('\n');
}
document.querySelector('#area-zoom-in').addEventListener('click',()=>areaZoom(area.zoom+.5));
function areaPolicyNote() {
  $('#area-policy-note').textContent=$('#area-assessment-mode').value==='regional_screening'
    ?'允许选择局部区域；仅计算框内降雨产流与坡面响应，不计框外上游入流。诊断仍会标记边界局限。'
    :'明显框外上游汇水或诊断不确定会阻止确认；通过仍只计算框内降雨，需人工核对，不能保证完整自然流域。';
}
document.querySelector('#area-assessment-mode').addEventListener('change',()=>{areaPolicyNote();saveAreaDraft();});
document.querySelector('#area-zoom-out').addEventListener('click',()=>areaZoom(area.zoom-.5));
document.querySelector('#area-zoom-reset').addEventListener('click',()=>areaZoom(1));
document.querySelector('#area-mapwrap').addEventListener('pointerdown',event=>{
  const vertex=event.target.closest('.editable-vertex');if(!vertex||busy||taskWorkspace?.actions?.edit_boundary===false) return;
  event.preventDefault();areaRemember();area.drag=Number(vertex.dataset.vertex);event.currentTarget.setPointerCapture(event.pointerId);
});
document.querySelector('#area-mapwrap').addEventListener('pointermove',event=>{
  if(area.drag===null||busy) return;const point=eventToMapPoint(event);if(!point) return;
  area.points[area.drag]=point;areaDraw();
});
for(const type of ['pointerup','pointercancel']) document.querySelector('#area-mapwrap').addEventListener(type,event=>{
  if(area.drag===null) return;area.drag=null;area.suppressClick=true;saveAreaDraft();
  if(event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
});
document.querySelector('#task-edit-location').addEventListener('click',()=>openSavedDrawing());
document.querySelector('#task-edit-boundary').addEventListener('click',async()=>{const qa=taskWorkspace?.spatial_qa;
  const data=await taskMutation('edit',{scope:'boundary'});if(data?.reply.ok) openSavedDrawing(qa);});
document.querySelector('#task-edit-rainfall').addEventListener('click',()=>showTaskSettings('rainfall'));
document.querySelector('#task-edit-parameters').addEventListener('click',()=>showTaskSettings('parameters'));
document.querySelector('#task-settings-close').addEventListener('click',()=>{$('#task-settings').hidden=true;});
document.querySelector('#task-settings-save').addEventListener('click',async()=>{
  try {
    let body={scope:taskSettingsScope};
    if(taskSettingsScope==='rainfall') {
      const health=await api('/api/v1/health');
      if(!health.adaptive_rainfall_stations) throw new Error('当前 WebUI 仍运行旧版本，请重启后再修改虚拟雨站数量。');
      const text=$('#task-rainfall').value.trim();
      const countText=$('#task-station-count').value.trim();
      const stationCount=countText===''?null:Number(countText);
      if(stationCount!==null&&(!Number.isInteger(stationCount)||stationCount<1||stationCount>100)) throw new Error('虚拟雨站数量须为1～100的整数，留空则按面积自动确定。');
      if(text==='历史最大日') body.rainfall={mode:'historical_max',lookback_years:30};
      else if(text.includes('~')) {const [start,end]=text.split('~').map(s=>s.trim());body.rainfall={mode:'explicit_period',start_datetime:start,end_datetime:end};}
      else body.rainfall={mode:'explicit_period',date:text};
      body.rainfall.station_count=stationCount;
    } else {
      const overrides={};for(const line of $('#task-overrides').value.split('\n').filter(s=>s.trim())) {
        const parts=line.split('=');if(parts.length!==2||!parts[0].trim()||!parts[1].trim()||!Number.isFinite(Number(parts[1]))) throw new Error('参数请逐行填写 名称=有限数值。');
        overrides[parts[0].trim()]=Number(parts[1]);
      }
      body={...body,parameter_preset:$('#task-preset').value,parameter_overrides:overrides};
    }
    $('#task-settings-error').textContent='';const data=await taskMutation('edit',body);
    if(data?.reply.ok) $('#task-settings').hidden=true;
  } catch(error) {$('#task-settings-error').textContent=error.message;}
});
document.querySelector('#task-revision').addEventListener('change',()=>{revisionChoice=$('#task-revision').value;renderTaskWorkspace();});
document.querySelector('#task-restore').addEventListener('click',async()=>{const data=await taskMutation('restore',{revision_id:$('#task-revision').value});if(data?.reply.ok) openSavedDrawing();});
document.querySelector('#task-confirm').addEventListener('click',()=>send('确认该边界'));
document.querySelector('#task-prepare').addEventListener('click',()=>taskMutation('prepare'));
document.querySelector('#task-plan').addEventListener('click',()=>taskMutation('plan'));
document.querySelector('#task-start').addEventListener('click',()=>send('开始运行'));
document.querySelector('#task-refresh').addEventListener('click',()=>send('查看当前任务进度'));
