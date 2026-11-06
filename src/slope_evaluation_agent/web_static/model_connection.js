/* Connection state stays in server memory; passwords never enter storage/history. */
(() => {
  const el = id => document.getElementById(id);
  const dialog = el('model-connection-dialog'), form = el('model-connection-form');
  let token = '', state = null, effortHints = {}, pending = false, candidates = [];
  const allEfforts = ['none','minimal','low','medium','high','xhigh','max','ultra'];
  const profiles = {
    'openai-compatible': {label:'4Router GPT', format:'responses'},
    anthropic: {label:'4Router Claude', format:'anthropic_messages'},
    grok: {label:'4Router Grok', format:'chat_completions'},
    gemini: {label:'4Router Gemini', format:'chat_completions'},
    qwen: {label:'4Router Qwen', format:'chat_completions'},
    deepseek: {label:'DeepSeek直连', format:'chat_completions'}
  };
  function status(text, error = false) {
    el('connection-status').textContent = text;
    el('connection-status').classList.toggle('error', error);
  }
  function effortOptions(value = el('connection-effort').value) {
    const provider = el('connection-provider').value;
    const anthropic = el('connection-format').value === 'anthropic_messages';
    const model = el('connection-model').value.trim();
    const hints = effortHints[el('connection-model').value.trim()];
    let allowed = hints?.length ? hints : (provider === 'deepseek' ? ['none','low','high','max'] : allEfforts);
    if (anthropic) {
      allowed = ['low','medium','high','xhigh','max'];
      if (/^claude-(haiku|sonnet)-4-5(?:-|$)/.test(model)) allowed = [];
      else if (/^claude-opus-4-5(?:-|$)/.test(model)) allowed = ['low','medium','high'];
      else if (/^claude-(opus|sonnet)-4-6(?:-|$)/.test(model)) allowed = ['low','medium','high','max'];
    } else if (provider === 'grok') {
      allowed = ['low','medium','high','xhigh'];
      if (/grok-3-mini/.test(model)) allowed = ['low','high'];
      else if (/non-reasoning|^grok-4(?:-0709)?$/.test(model)) allowed = [];
    } else if (provider === 'gemini') {
      allowed = ['minimal','low','medium','high'];
      if (/gemini-2\.5-flash/.test(model)) allowed.unshift('none');
    } else if (provider === 'qwen') {
      allowed = ['none','minimal','low','medium','high','xhigh','max'];
    }
    const select = el('connection-effort'); select.replaceChildren();
    select.add(new Option('默认，不额外指定', ''));
    for (const effort of allowed) select.add(new Option(effort, effort));
    select.value = allowed.includes(value) ? value : '';
    el('connection-thinking-label').hidden = provider !== 'deepseek';
    el('connection-thinking').disabled = provider !== 'deepseek' || !!select.value;
    if (provider === 'deepseek' && select.value) el('connection-thinking').value = select.value === 'none' ? 'disabled' : 'enabled';
    el('connection-anthropic-thinking-label').hidden = !anthropic;
    const manual = anthropic && el('connection-anthropic-thinking').value === 'enabled';
    el('connection-anthropic-budget-label').hidden = !manual;
    el('connection-anthropic-budget').disabled = !manual;
    el('connection-anthropic-budget').setCustomValidity(manual &&
      Number(el('connection-anthropic-budget').value) >= Number(el('connection-tokens').value)
      ? '思考预算必须小于生成token上限，为最终回答保留空间。' : '');
    el('connection-qwen-thinking-label').hidden = provider !== 'qwen';
    el('connection-qwen-budget-label').hidden = provider !== 'qwen';
    el('connection-gemini-budget-label').hidden = provider !== 'gemini';
    el('connection-qwen-budget').disabled = provider !== 'qwen';
    el('connection-gemini-budget').disabled = provider !== 'gemini';
    const qwenBudget = el('connection-qwen-budget'), geminiBudget = el('connection-gemini-budget');
    qwenBudget.setCustomValidity(provider === 'qwen' && qwenBudget.value !== '' &&
      (select.value || el('connection-qwen-thinking').value === 'disabled')
      ? '思考预算不能与effort同时指定，也不能在关闭思考时指定。' : '');
    geminiBudget.setCustomValidity(provider === 'gemini' && geminiBudget.value !== '' && select.value
      ? 'Gemini思考预算与推理强度请二选一。' : '');
    el('connection-qwen-thinking').setCustomValidity(provider === 'qwen' &&
      ((el('connection-qwen-thinking').value === 'disabled' && select.value && select.value !== 'none') ||
       (el('connection-qwen-thinking').value === 'enabled' && select.value === 'none'))
      ? 'Qwen思考开关与所选强度冲突。' : '');
    el('connection-stream-label').hidden = el('connection-format').value !== 'chat_completions';
    el('connection-help').textContent = anthropic
      ? 'Claude的思考模式与effort独立。默认不额外指定；旧模型与新模型支持项不同。JSON通过提示词约束并由程序校验，不发送OpenAI的JSON参数。'
      : '推理选项取决于所选模型。生成上限可能同时包含思考与最终回答；不支持的设置会在连接测试中报告。';
    if (provider === 'grok') el('connection-help').textContent = 'Grok使用4Router兼容接口，可按通道选择Chat Completions或Responses。effort支持范围随模型变化；默认不额外指定。';
    if (provider === 'gemini') el('connection-help').textContent = 'Gemini默认走4Router的Chat兼容入口；通道要求原生时可选GenerateContent。思考强度与预算二选一，关闭思考须模型支持。';
    if (provider === 'qwen') el('connection-help').textContent = 'Qwen默认使用4Router兼容Chat和流式传输。思考开关、预算与effort按模型支持选择；旧模型思考时可能无法保证JSON格式。可用模型以4Router分组为准。';
    if (manual) el('connection-anthropic-budget').max = Math.max(1024, Number(el('connection-tokens').value) - 1);
  }
  function clearCandidates() {
    candidates = []; effortHints = {}; el('connection-models').replaceChildren();
    el('connection-candidates').replaceChildren(new Option('请选择模型', ''));
    el('connection-candidates-label').hidden = true;
  }
  function syncCandidates() {
    const model = el('connection-model').value.trim();
    el('connection-candidates').value = candidates.includes(model) ? model : '';
    el('connection-model-note').textContent = model && !candidates.includes(model)
      ? `当前填写的 ${model} 不在本次返回列表中，请核对模型或选择候选项。`
      : '选择只更新表单，点击“应用连接”才会切换；列表不保证生成通道可用。';
  }
  function modelFamily(model) {
    const name = model.toLowerCase().replace(/^models\//, '').replace(/^(google|x-ai|xai|alibaba|anthropic|openai|qwen)\//, '');
    return name.startsWith('claude-') ? 'anthropic' : name.startsWith('gpt-') ? 'openai-compatible'
      : name.startsWith('grok-') ? 'grok' : name.startsWith('gemini-') ? 'gemini' : /^(qwen|qwq)/.test(name) ? 'qwen'
      : name.startsWith('deepseek') ? 'deepseek' : null;
  }
  function modelChanged() {
    const model = el('connection-model').value.trim();
    let fourRouter = false;
    try { fourRouter = new URL(el('connection-url').value).hostname === '4router.net'; } catch (_) {}
    const family = modelFamily(model);
    if (fourRouter && family && family !== 'deepseek' && el('connection-provider').value !== family) {
      el('connection-provider').value = family; el('connection-format').value = profiles[family].format;
      el('connection-temperature').value = ''; el('connection-stream').checked = family === 'qwen';
      el('connection-qwen-thinking').value = 'default'; el('connection-qwen-budget').value = '';
      el('connection-gemini-budget').value = ''; effortOptions('');
      status(`已匹配${profiles[family].label}及默认协议，请核对分组密钥并测试连接。`);
    }
    syncCandidates(); effortOptions();
  }
  function transportSupported() {
    const provider = el('connection-provider').value;
    if (['grok','gemini','qwen'].includes(provider) && !state?.supported_providers?.includes(provider)) {
      status('当前后端尚未加载Grok、Gemini、Qwen适配，请使用已更新的服务。', true); return false;
    }
    if (el('connection-format').value === 'anthropic_messages' && !state?.supported_api_formats?.includes('anthropic_messages')) {
      status('当前运行的后端尚未加载Claude接口支持，请使用已更新的服务后再测试或应用。', true); return false;
    }
    if (el('connection-format').value === 'gemini_generate_content' && !state?.supported_api_formats?.includes('gemini_generate_content')) {
      status('当前后端尚未加载Gemini原生协议，请使用已更新的服务。', true); return false;
    }
    return true;
  }
  function payload() {
    const body = {provider: el('connection-provider').value, base_url: el('connection-url').value.trim().replace(/\/+$/, ''),
      model: el('connection-model').value.trim(), api_key: el('connection-key').value || null, retain_key: true,
      api_format: el('connection-format').value, reasoning_effort: el('connection-effort').value || null,
      deepseek_thinking: el('connection-thinking').value,
      temperature: el('connection-temperature').value === '' ? null : Number(el('connection-temperature').value),
      maximum_tokens: Number(el('connection-tokens').value), request_timeout_s: Number(el('connection-timeout').value),
      max_retries: state?.max_retries ?? 1, json_mode: el('connection-json').checked};
    if (body.api_format === 'anthropic_messages') {
      body.anthropic_thinking = el('connection-anthropic-thinking').value;
      body.anthropic_thinking_budget_tokens = Number(el('connection-anthropic-budget').value);
    }
    if (state?.chat_streaming !== undefined || el('connection-stream').checked) body.chat_streaming = el('connection-stream').checked;
    if (body.provider === 'qwen') {
      body.qwen_thinking = el('connection-qwen-thinking').value;
      body.qwen_thinking_budget_tokens = el('connection-qwen-budget').value === '' ? null : Number(el('connection-qwen-budget').value);
    }
    if (body.provider === 'gemini') body.gemini_thinking_budget_tokens = el('connection-gemini-budget').value === '' ? null : Number(el('connection-gemini-budget').value);
    return body;
  }
  async function request(path, body) {
    const response = await fetch('/api/v1/model-connection' + path, {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Model-Connection-Token': token}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : '模型连接操作失败。';
      // Works with an already-running older backend, preserving its memory key.
      if (/http_503/i.test(detail)) throw new Error('中转站返回HTTP 503：本次生成请求未获服务。模型列表可读取不代表生成通道可用；请查看中转站该次请求的错误详情，确认模型通道、账户分组或上游状态。');
      throw new Error(detail);
    }
    return data;
  }
  async function action(message, callback) {
    if (pending) return;
    pending = true; status(message);
    form.querySelectorAll('input,select,button').forEach(x => x.disabled = true);
    try { await callback(); } catch (error) { status(error.message, true); }
    finally {
      pending = false; form.querySelectorAll('input,select,button').forEach(x => x.disabled = false); effortOptions();
    }
  }
  async function open() {
    try {
      const response = await fetch('/api/v1/model-connection');
      if (!response.ok) throw new Error('当前服务尚无模型设置入口，请重启为最新版本。');
      state = await response.json(); token = state.form_token; effortHints = {};
      el('connection-url').value = state.base_url || 'https://4router.net/v1';
      el('connection-provider').value = state.provider || 'openai-compatible';
      el('connection-format').value = state.api_format || 'chat_completions';
      el('connection-model').value = state.model || '';
      el('connection-key').value = '';
      el('connection-key').placeholder = state.credential_present ? '地址未变时可留空沿用；切换地址请提供新密钥' : '填写本服务的API密钥';
      el('connection-thinking').value = state.deepseek_thinking || 'disabled';
      el('connection-anthropic-thinking').value = state.anthropic_thinking || 'default';
      el('connection-anthropic-budget').value = state.anthropic_thinking_budget_tokens || 1024;
      el('connection-qwen-thinking').value = state.qwen_thinking || 'default';
      el('connection-qwen-budget').value = state.qwen_thinking_budget_tokens ?? '';
      el('connection-gemini-budget').value = state.gemini_thinking_budget_tokens ?? '';
      el('connection-stream').checked = !!state.chat_streaming;
      const temperature = state.temperature == null ? '' : String(state.temperature);
      if (![...el('connection-temperature').options].some(o => o.value === temperature)) el('connection-temperature').add(new Option(temperature, temperature));
      el('connection-temperature').value = temperature;
      el('connection-tokens').value = state.maximum_tokens || 8192;
      el('connection-timeout').value = state.request_timeout_s || 120;
      el('connection-json').checked = state.json_mode !== false;
      clearCandidates(); effortOptions(state.reasoning_effort || '');
      status('可先读取模型列表，再测试连接；当前设置尚未改变。');
      dialog.showModal(); el('connection-url').focus();
    } catch (error) { alert(error.message); }
  }
  function useFourRouter(anthropic = false) {
    useProfile(anthropic ? 'anthropic' : 'openai-compatible');
  }
  function useProfile(provider) {
    const endpoint = provider === 'deepseek' ? 'https://api.deepseek.com' : 'https://4router.net/v1';
    const sameEndpoint = el('connection-url').value.replace(/\/+$/, '') === endpoint;
    el('connection-url').value = endpoint;
    el('connection-provider').value = provider;
    el('connection-format').value = profiles[provider].format;
    el('connection-model').value = ''; if (!sameEndpoint) el('connection-key').value = '';
    el('connection-temperature').value = ''; el('connection-tokens').value = 8192; el('connection-timeout').value = 180;
    el('connection-anthropic-thinking').value = 'default';
    el('connection-qwen-thinking').value = 'default'; el('connection-qwen-budget').value = '';
    el('connection-gemini-budget').value = ''; el('connection-stream').checked = provider === 'qwen';
    clearCandidates(); effortOptions('');
    status(`已填入${profiles[provider].label}默认地址和协议。同地址可留空沿用内存密钥；切换分组或地址时请填写对应密钥。`);
  }
  el('model-chip').addEventListener('click', open);
  el('connection-close').addEventListener('click', () => { if (!pending) dialog.close(); });
  dialog.addEventListener('cancel', event => { if (pending) event.preventDefault(); });
  dialog.addEventListener('close', () => { el('connection-key').value = ''; token = ''; el('model-chip').focus(); });
  el('connection-fourrouter').addEventListener('click', () => useFourRouter(false));
  el('connection-fourrouter-claude').addEventListener('click', () => useFourRouter(true));
  for (const provider of ['grok','gemini','qwen']) el('connection-fourrouter-' + provider).addEventListener('click', () => useProfile(provider));
  el('connection-deepseek').addEventListener('click', () => useProfile('deepseek'));
  el('connection-provider').addEventListener('change', () => {
    useProfile(el('connection-provider').value);
  });
  el('connection-format').addEventListener('change', () => {
    if (el('connection-format').value === 'anthropic_messages') {
      el('connection-provider').value = 'anthropic'; el('connection-temperature').value = '';
    } else if (el('connection-format').value === 'gemini_generate_content') {
      el('connection-provider').value = 'gemini'; el('connection-temperature').value = '';
    } else if (el('connection-provider').value === 'anthropic' ||
      (el('connection-format').value === 'responses' && ['qwen','gemini'].includes(el('connection-provider').value))) el('connection-provider').value = 'openai-compatible';
    effortOptions('');
  });
  el('connection-model').addEventListener('input', () => { effortOptions(); syncCandidates(); });
  el('connection-model').addEventListener('change', modelChanged);
  el('connection-candidates').addEventListener('change', () => {
    if (el('connection-candidates').value) { el('connection-model').value = el('connection-candidates').value; modelChanged(); }
  });
  el('connection-key').addEventListener('input', clearCandidates);
  el('connection-anthropic-thinking').addEventListener('change', () => {
    if (['enabled','adaptive'].includes(el('connection-anthropic-thinking').value)) el('connection-temperature').value = '';
    effortOptions();
  });
  el('connection-tokens').addEventListener('input', () => effortOptions());
  el('connection-anthropic-budget').addEventListener('input', () => effortOptions());
  for (const id of ['connection-qwen-thinking','connection-qwen-budget','connection-gemini-budget']) {
    el(id).addEventListener(id.endsWith('thinking') ? 'change' : 'input', () => effortOptions());
  }
  el('connection-effort').addEventListener('change', () => effortOptions());
  el('connection-url').addEventListener('change', () => {
    clearCandidates();
    if (el('connection-url').value.replace(/\/+$/, '') !== (state?.base_url || '').replace(/\/+$/, '')) {
      el('connection-model').value = ''; effortOptions('');
      status('API地址已变更。请提供对应密钥，并重新选择模型。');
    }
  });
  el('connection-list').addEventListener('click', () => {
    if (!transportSupported()) return;
    const body = payload();
    action('正在读取模型列表…', async () => {
      const data = await request('/models', body); effortHints = data.reasoning_efforts || {};
      el('connection-models').replaceChildren();
      for (const id of data.models) { const option = document.createElement('option'); option.value = id; el('connection-models').append(option); }
      candidates = data.models;
      el('connection-candidates').replaceChildren(new Option('请选择模型', ''));
      for (const id of candidates) el('connection-candidates').add(new Option(id, id));
      el('connection-candidates-label').hidden = false; syncCandidates();
      const selected = el('connection-provider').value;
      const mismatch = candidates.length && !candidates.some(id => modelFamily(id) === selected);
      effortOptions();
      if (mismatch) status(`已读取${candidates.length}个候选模型，但未识别到${profiles[selected].label}条目。请核对密钥分组；特殊别名可按服务商提供的ID手动填写。`, true);
      else status(`已读取${data.models.length}个候选模型，请选择；是否可用以连接测试为准。`);
    });
  });
  el('connection-check').addEventListener('click', () => {
    if (!transportSupported() || !form.reportValidity()) return;
    const body = payload();
    action('正在检查连接和基础意图理解；不会启动计算…', async () => {
      const data = await request('/check', body);
      const model = data.response_model && data.response_model !== data.model
        ? `${data.model}（接口返回：${data.response_model}）` : (data.response_model || data.model);
      if (data.workflow_compatible === false) {
        const names = {greeting: '问候', engineering_question: '工程原理提问'};
        const failed = (data.routing_checks || []).find(check => !check.passed);
        const scope = names[failed?.name] || '基础意图';
        const reasons = {
          transport_error: '请求或响应处理发生异常，尚不能据此判断模型是否理解。',
          response_truncated: '模型回答未完整返回，请核对生成上限和通道输出限制。',
          empty_response: '模型没有返回可用正文。',
          invalid_json: '模型正文未能解析为JSON对象。',
          invalid_response: '接口响应结构不符合所选协议。',
          invalid_route_schema: 'JSON可解析，但意图字段不符合本项目的格式要求。',
          unexpected_conversation_route: '模型给出的意图与本次检查的预期不符。',
          unexpected_tool_call: '模型返回了工具请求，未执行该请求。',
          model_decision_refused: '模型提出了只能由用户确认的操作，已被拒绝。',
          request_failed: '模型服务未正常完成这次请求。'
        };
        const reason = reasons[failed?.reason] || (/^http_\d{3}$/.test(failed?.reason || '')
          ? `意图请求返回HTTP ${failed.reason.slice(5)}，请核对服务商记录。`
          : '未获得符合要求的模型意图输出。');
        status(`基础JSON请求成功：${model}；${scope}检查未通过。${reason}本次测试未切换当前连接。`, true);
      } else if (data.workflow_compatible === true) {
        status(`连接和基础意图检查通过：${model}。具体工作流仍需任务验证；尚未切换当前连接。`);
      } else {
        status(`基础JSON连接测试通过：${model}。当前运行版本尚未检查意图理解；应用新版服务后可进一步验证，当前连接尚未改变。`);
      }
    });
  });
  form.addEventListener('submit', event => {
    event.preventDefault(); if (!transportSupported() || !form.reportValidity()) return;
    const body = payload();
    action('正在应用连接…', async () => {
      state = await request('', body); el('connection-key').value = '';
      status(`已应用：${state.model}。后续消息使用此连接；配置和密钥仅保留在当前服务内存。`);
      if (typeof health === 'function') await health();
    });
  });
})();
