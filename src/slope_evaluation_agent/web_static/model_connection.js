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
    deepseek: {label:'DeepSeek direct', format:'chat_completions'}
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
    select.add(new Option('Provider default', ''));
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
      ? 'The thinking budget must leave room for the final answer.' : '');
    el('connection-qwen-thinking-label').hidden = provider !== 'qwen';
    el('connection-qwen-budget-label').hidden = provider !== 'qwen';
    el('connection-gemini-budget-label').hidden = provider !== 'gemini';
    el('connection-qwen-budget').disabled = provider !== 'qwen';
    el('connection-gemini-budget').disabled = provider !== 'gemini';
    const qwenBudget = el('connection-qwen-budget'), geminiBudget = el('connection-gemini-budget');
    qwenBudget.setCustomValidity(provider === 'qwen' && qwenBudget.value !== '' &&
      (select.value || el('connection-qwen-thinking').value === 'disabled')
      ? 'A thinking budget requires thinking to be enabled and cannot be combined with effort.' : '');
    geminiBudget.setCustomValidity(provider === 'gemini' && geminiBudget.value !== '' && select.value
      ? 'Choose either a Gemini thinking budget or reasoning effort.' : '');
    el('connection-qwen-thinking').setCustomValidity(provider === 'qwen' &&
      ((el('connection-qwen-thinking').value === 'disabled' && select.value && select.value !== 'none') ||
       (el('connection-qwen-thinking').value === 'enabled' && select.value === 'none'))
      ? 'The selected Qwen thinking mode conflicts with reasoning effort.' : '');
    el('connection-stream-label').hidden = el('connection-format').value !== 'chat_completions';
    el('connection-help').textContent = anthropic
      ? 'Claude thinking and effort are independent and model-dependent. JSON is requested through the prompt and validated by the agent.'
      : 'Options vary by model. The token limit may cover reasoning and the final answer. Test the connection to check your settings.';
    if (provider === 'grok') el('connection-help').textContent = 'For Grok, choose Chat Completions or Responses according to your endpoint. Effort support varies by model.';
    if (provider === 'gemini') el('connection-help').textContent = 'Gemini supports compatible Chat Completions and native GenerateContent. Choose either effort or a budget.';
    if (provider === 'qwen') el('connection-help').textContent = 'Qwen defaults to streaming Chat Completions. Available models, thinking options and JSON support depend on your endpoint.';
    if (manual) el('connection-anthropic-budget').max = Math.max(1024, Number(el('connection-tokens').value) - 1);
  }
  function clearCandidates() {
    candidates = []; effortHints = {}; el('connection-models').replaceChildren();
    el('connection-candidates').replaceChildren(new Option('Select a model', ''));
    el('connection-candidates-label').hidden = true;
  }
  function syncCandidates() {
    const model = el('connection-model').value.trim();
    el('connection-candidates').value = candidates.includes(model) ? model : '';
    el('connection-model-note').textContent = model && !candidates.includes(model)
      ? `${model} is absent from the returned list. Check the ID or choose a listed model.`
      : 'Select Apply connection to use your changes. Test the model before relying on it.';
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
      status(`Selected ${profiles[family].label} and its default format. Check the key and test the connection.`);
    }
    syncCandidates(); effortOptions();
  }
  function transportSupported() {
    const provider = el('connection-provider').value;
    if (['grok','gemini','qwen'].includes(provider) && !state?.supported_providers?.includes(provider)) {
      status('Restart the updated service to use Grok, Gemini or Qwen.', true); return false;
    }
    if (el('connection-format').value === 'anthropic_messages' && !state?.supported_api_formats?.includes('anthropic_messages')) {
      status('Restart the updated service before using the Claude API.', true); return false;
    }
    if (el('connection-format').value === 'gemini_generate_content' && !state?.supported_api_formats?.includes('gemini_generate_content')) {
      status('Restart the updated service before using native Gemini GenerateContent.', true); return false;
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
      const detail = typeof data.detail === 'string' ? data.detail : 'Model connection failed.';
      // Works with an already-running older backend, preserving its memory key.
      if (/http_503/i.test(detail)) throw new Error('The endpoint returned HTTP 503. Check the provider log, model access and upstream status.');
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
      if (!response.ok) throw new Error('Model settings are unavailable. Restart the updated service.');
      state = await response.json(); token = state.form_token; effortHints = {};
      el('connection-url').value = state.base_url || 'https://4router.net/v1';
      el('connection-provider').value = state.provider || 'openai-compatible';
      el('connection-format').value = state.api_format || 'chat_completions';
      el('connection-model').value = state.model || '';
      el('connection-key').value = '';
      el('connection-key').placeholder = state.credential_present ? 'Leave blank to reuse the key for this URL' : 'Enter the API key for this endpoint';
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
      status('Fetch the models, then test the connection. Your current connection is unchanged.');
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
    status(`Loaded the defaults for ${profiles[provider].label}. Provide a key when changing URL or access group.`);
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
      status('The API URL changed. Enter the corresponding key and select a model.');
    }
  });
  el('connection-list').addEventListener('click', () => {
    if (!transportSupported()) return;
    const body = payload();
    action('Fetching models…', async () => {
      const data = await request('/models', body); effortHints = data.reasoning_efforts || {};
      el('connection-models').replaceChildren();
      for (const id of data.models) { const option = document.createElement('option'); option.value = id; el('connection-models').append(option); }
      candidates = data.models;
      el('connection-candidates').replaceChildren(new Option('Select a model', ''));
      for (const id of candidates) el('connection-candidates').add(new Option(id, id));
      el('connection-candidates-label').hidden = false; syncCandidates();
      const selected = el('connection-provider').value;
      const mismatch = candidates.length && !candidates.some(id => modelFamily(id) === selected);
      effortOptions();
      if (mismatch) status(`Fetched ${candidates.length} models. No ${profiles[selected].label} entry was identified. Check your access group or enter a provider model ID.`, true);
      else status(`Fetched ${data.models.length} models. Select one and test the connection.`);
    });
  });
  el('connection-check').addEventListener('click', () => {
    if (!transportSupported() || !form.reportValidity()) return;
    const body = payload();
    action('Testing the connection and basic intents…', async () => {
      const data = await request('/check', body);
      const model = data.response_model && data.response_model !== data.model
        ? `${data.model} (API returned ${data.response_model})` : (data.response_model || data.model);
      if (data.workflow_compatible === false) {
        const names = {greeting: 'greeting', engineering_question: 'engineering question'};
        const failed = (data.routing_checks || []).find(check => !check.passed);
        const scope = names[failed?.name] || 'basic intent';
        const reasons = {
          transport_error: 'A transport or response error prevented the intent check.',
          response_truncated: 'The response was truncated. Check the token limit and endpoint limits.',
          empty_response: 'The model returned no usable text.',
          invalid_json: 'The response could not be parsed as a JSON object.',
          invalid_response: 'The response did not match the selected API format.',
          invalid_route_schema: 'The intent fields did not match the expected schema.',
          unexpected_conversation_route: 'The model returned an unexpected intent.',
          unexpected_tool_call: 'The check declined an unexpected tool request.',
          model_decision_refused: 'The check declined an operation requiring user confirmation.',
          request_failed: 'The model request failed.'
        };
        const reason = reasons[failed?.reason] || (/^http_\d{3}$/.test(failed?.reason || '')
          ? `Intent request returned HTTP ${failed.reason.slice(5)}. Check the provider log.`
          : 'The model returned no valid intent.');
        status(`JSON request passed for ${model}. The ${scope} check failed. ${reason} The current connection is unchanged.`, true);
      } else if (data.workflow_compatible === true) {
        status(`Connection and basic intent checks passed for ${model}. Apply the connection to use it.`);
      } else {
        status(`JSON connection test passed for ${model}. This service did not test intent routing.`);
      }
    });
  });
  form.addEventListener('submit', event => {
    event.preventDefault(); if (!transportSupported() || !form.reportValidity()) return;
    const body = payload();
    action('Applying connection…', async () => {
      state = await request('', body); el('connection-key').value = '';
      status(`Connected to ${state.model}. The settings and key are held in memory.`);
      if (typeof health === 'function') await health();
    });
  });
})();
