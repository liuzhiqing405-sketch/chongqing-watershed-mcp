const $ = (selector, root = document) => root.querySelector(selector);
const conversation = $('#conversation');
const composer = $('#composer');
const input = $('#message-input');
const sendButton = $('#send-button');
const historyList = $('#history-list');
const storageKey = 'slope-evaluation-session-v1';
let sessionId = localStorage.getItem(storageKey);
let busy = false;
let pollTimer = null;
let taskWorkspace = null;
let revisionSession = null;
let taskSettingsScope = null;
let historyItems = [];
let historyManaging = false;
let historyDeleting = false;
const selectedHistory = new Set();

const stateLabels = {
  new: 'New session', needs_location: 'Waiting for location', outlet_candidates_ready: 'Waiting for map selection',
  watershed_delineated: 'Boundary saved', awaiting_watershed_confirmation: 'Waiting for confirmation',
  watershed_qa_failed: 'Spatial checks failed',
  watershed_rejected: 'Boundary rejected', watershed_confirmed: 'Boundary confirmed', rainfall_ready: 'Rainfall ready',
  model_inputs_ready: 'Model inputs ready', hydro_input_imported: 'Hydrology inputs imported',
  hydro_input_validated: 'Hydrology inputs validated', hydro_planned: 'Run plan ready for review', hydro_running: 'Running',
  hydro_succeeded: 'Run completed', hydro_qa_failed: 'Quality checks failed', hydro_failed: 'Run failed',
  hydro_cancelled: 'Run cancelled', assessment_ready: 'Assessment ready'
};
const stateProgress = {
  new: 0, needs_location: 4, outlet_candidates_ready: 12, watershed_delineated: 25,
  awaiting_watershed_confirmation: 30, watershed_rejected: 12, watershed_confirmed: 40,
  watershed_qa_failed: 12,
  rainfall_ready: 52, model_inputs_ready: 60, hydro_input_imported: 65,
  hydro_input_validated: 70, hydro_planned: 76, hydro_running: 84,
  hydro_succeeded: 92, hydro_qa_failed: 92, hydro_failed: 92, hydro_cancelled: 92,
  assessment_ready: 100
};

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let data = {};
  try { data = await response.json(); } catch { /* non-json error */ }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`);
  return data;
}

function message(text, role = 'assistant', kind = 'message', meta = null) {
  const template = role === 'user' ? $('#user-template') : $('#assistant-template');
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(kind);
  $('.bubble', node).textContent = text;
  if (meta && role !== 'user' && meta.spatial_qa_passed === false) {
    const tags = document.createElement('div'); tags.className = 'reply-meta';
    const quality = document.createElement('span'); quality.textContent = 'Spatial quality checks failed';
    tags.append(quality);
    $('.bubble', node).append(tags);
  }
  conversation.append(node);
  // The welcome lockup steps back once a real exchange has started. The opening
  // greeting alone does not count — it is part of the welcome, and shrinking
  // there would mean the full-size lockup is never seen at all.
  const card = conversation.closest('.chat-card');
  if (card) card.classList.toggle('chat-active', conversation.children.length > 1);
  node.scrollIntoView({behavior: 'smooth', block: 'end'});
  return node;
}

function typing() {
  const node = message('', 'assistant');
  const bubble = $('.bubble', node); bubble.classList.add('typing');
  bubble.innerHTML = '<span class="typing-label">Reading your message…</span><span class="typing-dots"><i></i><i></i><i></i></span>';
  const label = $('.typing-label', bubble);
  // Two phases, and the second one only says what is actually true: the wait
  // is longer than reading a sentence takes, so it may involve a computation.
  // Nothing here claims a tool ran when the turn was conversation.
  const slow = setTimeout(() => {
    label.textContent = 'Working… Calculations and maps may take longer.';
  }, 2000);
  return () => { clearTimeout(slow); node.remove(); };
}

function addImages(node, images, skipUrls = null) {
  for (const image of images || []) {
    if (skipUrls && skipUrls.has(image.url)) continue;
    const figure = document.createElement('figure'); figure.className = 'reply-figure';
    const element = document.createElement('img'); element.src = image.url; element.alt = 'Workflow figure'; element.loading = 'lazy';
    figure.append(element); $('.bubble', node).append(figure);
  }
}

// Formats are named the way a person reads them, not the way the registry
// stores them. The .out label says what the file is, because "SWMM 二进制"
// is the difference between opening it in a viewer and opening it in a engine.
const FORMAT_LABELS = {
  geotiff: 'GeoTIFF', csv: 'CSV', tsv: 'TSV', xlsx: 'Excel', zip: 'ZIP',
  inp: 'INP', rpt: 'RPT', out: 'OUT (SWMM binary)', png: 'PNG', pdf: 'PDF',
  json: 'JSON', geojson: 'GeoJSON', shp: 'Shapefile', markdown: 'Documentation',
  text: 'Text', binary: 'File'
};

const ROLE_LABELS = {
  source_raster: 'Source raster', source_table: 'Source table', model_input: 'SWMM model input',
  native_report: 'SWMM run report', native_result: 'Native SWMM result', package: 'Output bundle',
  figure: 'Figure file', vector: 'Vector data', metadata: 'Metadata', spreadsheet: 'Derived table'
};

function formatLabel(format) { return FORMAT_LABELS[format] || String(format || 'File'); }

function artifactDetail(artifact) {
  const shape = artifact.rows != null
    ? `${artifact.rows} rows × ${artifact.column_count ?? '—'} columns` : null;
  const rows = [
    ['Role', ROLE_LABELS[artifact.role] || artifact.role || '—'],
    ['Size', artifact.size_text || '—'],
    ['Units', artifact.units || 'N/A'],
    ['CRS', artifact.crs || 'N/A'],
    ['Time column', artifact.time_column || 'N/A'],
    ['Table dimensions', shape || 'N/A'],
    ['SHA-256', artifact.sha256_short ? `${artifact.sha256_short}…` : '—'],
    ['Quality checks', artifact.qa_status === 'passed' ? 'Passed' : (artifact.diagnostic_only ? 'Failed' : 'Not recorded')],
    ['Source stage', artifact.stage_id || '—']
  ];
  const list = document.createElement('dl'); list.className = 'artifact-detail';
  for (const [term, value] of rows) {
    const dt = document.createElement('dt'); dt.textContent = term;
    const dd = document.createElement('dd'); dd.textContent = value;
    list.append(dt, dd);
  }
  return list;
}

function artifactRow(artifact) {
  const row = document.createElement('li'); row.className = 'artifact-row';

  const link = document.createElement('a');
  link.className = 'artifact-download';
  link.href = artifact.download_url; link.setAttribute('download', '');
  link.textContent = `Download original ${formatLabel(artifact.format)} · ${artifact.size_text || 'Unknown size'}`;
  link.title = artifact.title || '';
  link.setAttribute('aria-label',
    `Download ${artifact.title || artifact.artifact_id}, format ${formatLabel(artifact.format)}`);
  row.append(link);

  // Only formats the service says it can produce are offered. A greyed-out
  // button for a file that does not exist reads as a broken feature.
  for (const format of artifact.export_formats || []) {
    const generate = document.createElement('button');
    generate.type = 'button'; generate.className = 'artifact-generate';
    generate.textContent = `Generate ${formatLabel(format)}`;
    generate.addEventListener('click', () => generateThenDownload(
      generate, artifact, format));
    row.append(generate);
  }

  const info = document.createElement('button');
  info.type = 'button'; info.className = 'artifact-info-toggle';
  info.textContent = 'Details'; info.setAttribute('aria-expanded', 'false');
  const detail = artifactDetail(artifact); detail.hidden = true;
  info.addEventListener('click', () => {
    detail.hidden = !detail.hidden;
    info.setAttribute('aria-expanded', String(!detail.hidden));
  });
  row.append(info);

  const details = document.createElement('div'); details.className = 'artifact-more';
  if (artifact.diagnostic_only) {
    const warn = document.createElement('p'); warn.className = 'artifact-warning';
    warn.textContent = 'Diagnostics only (quality checks failed)';
    details.append(warn);
  }
  details.append(detail);
  row.append(details);
  return row;
}

function addArtifactList(node, artifacts) {
  const items = artifacts || [];
  if (!items.length) return null;
  const list = document.createElement('ul'); list.className = 'artifact-list';
  for (const artifact of items) list.append(artifactRow(artifact));
  $('.bubble', node).append(list);
  return list;
}

async function generateThenDownload(button, artifact, format) {
  const label = button.textContent;
  button.disabled = true; button.textContent = 'Generating…';
  try {
    const data = await api(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/artifacts/${encodeURIComponent(artifact.artifact_id)}/exports`,
      {method: 'POST', headers: {'Content-Type': 'application/json'},
       body: JSON.stringify({format})});
    const made = data.artifact;
    if (!made) throw new Error('The export returned no file');
    triggerDownload(made.download_url);
    // The generated file replaces the button: it is a normal download from now
    // on, and re-deriving it would only repeat work the service already cached.
    const link = document.createElement('a');
    link.className = 'artifact-download';
    link.href = made.download_url; link.setAttribute('download', '');
    link.textContent = `${formatLabel(made.format)} · ${made.size_text || 'Unknown size'}`;
    link.title = made.title || '';
    link.setAttribute('aria-label',
      `Download generated ${made.title || made.artifact_id}, format ${formatLabel(made.format)}`);
    button.replaceWith(link);
  } catch (error) {
    button.disabled = false; button.textContent = label;
    const note = document.createElement('p'); note.className = 'artifact-error';
    note.textContent = `Export failed: ${error.message}`;
    button.closest('.artifact-row').append(note);
  }
}

function triggerDownload(url) {
  const link = document.createElement('a');
  link.href = url; link.setAttribute('download', '');
  document.body.append(link); link.click(); link.remove();
}

function addVisualizations(node, visualizations) {
  const previewUrls = new Set();
  for (const figure of visualizations || []) {
    const wrapper = document.createElement('figure'); wrapper.className = 'reply-figure';
    if (figure.preview && figure.preview.url) {
      const image = document.createElement('img');
      image.src = figure.preview.url;
      image.alt = `Figure: ${figure.title || figure.kind || 'Workflow figure'}`;
      image.loading = 'lazy';
      wrapper.append(image);
      previewUrls.add(figure.preview.url);
    }
    const caption = document.createElement('figcaption');
    const name = document.createElement('strong');
    name.textContent = figure.title || figure.kind || 'Figure';
    caption.append(name);
    if (figure.qa_failed) {
      const badge = document.createElement('span'); badge.className = 'qa-badge';
      badge.textContent = 'Quality checks failed (diagnostics only)';
      caption.append(badge);
    }
    wrapper.append(caption);
    const details = document.createElement('details'); details.className = 'figure-details';
    const detailsTitle = document.createElement('summary'); detailsTitle.textContent = 'Figure details and downloads';
    details.append(detailsTitle);
    for (const limitation of figure.limitations || []) {
      const line = document.createElement('small'); line.textContent = limitation;
      details.append(line);
    }

    if ((figure.artifacts || []).length) {
      const heading = document.createElement('p'); heading.className = 'artifact-heading';
      heading.textContent = 'Source outputs for this figure:';
      details.append(heading);
      const list = document.createElement('ul'); list.className = 'artifact-list';
      for (const artifact of figure.artifacts) list.append(artifactRow(artifact));
      details.append(list);
    } else {
      const none = document.createElement('p'); none.className = 'artifact-heading';
      none.textContent = 'This figure has no registered files to download.';
      details.append(none);
    }
    wrapper.append(details);
    $('.bubble', node).append(wrapper);
  }
  return previewUrls;
}

function addSwmmSection(node, swmm) {
  if (!swmm || (!(swmm.artifacts || []).length && !swmm.bundle)) return;
  const details = document.createElement('details'); details.className = 'swmm-section';
  const summary = document.createElement('summary');
  summary.textContent = `SWMM model outputs (${(swmm.artifacts || []).length} files)`;
  details.append(summary);

  if (swmm.complete === false) {
    const warn = document.createElement('p'); warn.className = 'artifact-warning';
    warn.textContent = `Required files missing: ${(swmm.missing_required || []).join(', ')}. The outputs are incomplete.`;
    details.append(warn);
  }
  const list = document.createElement('ul'); list.className = 'artifact-list';
  for (const artifact of swmm.artifacts || []) {
    const row = artifactRow(artifact);
    if (artifact.format === 'out') {
      const note = document.createElement('p'); note.className = 'artifact-note';
      note.textContent = 'Native SWMM binary output. Open it with SWMM or a compatible reader.';
      row.querySelector('.artifact-more').prepend(note);
    }
    list.append(row);
  }
  details.append(list);

  const bundleRow = document.createElement('div'); bundleRow.className = 'bundle-row';
  if (swmm.bundle) {
    const link = document.createElement('a');
    link.className = 'artifact-download';
    link.href = swmm.bundle.download_url; link.setAttribute('download', '');
    const bundleLabel = swmm.bundle_complete === false ? 'Diagnostic model ZIP' : 'Complete model ZIP';
    link.textContent = `${bundleLabel} · ${swmm.bundle.size_text || 'Unknown size'}`;
    if (swmm.bundle_complete === false) {
      const note = document.createElement('span'); note.className = 'qa-badge';
      note.textContent = 'Incomplete';
      link.append(note);
    }
    bundleRow.append(link);
  } else {
    const make = document.createElement('button');
    make.type = 'button'; make.className = 'artifact-generate';
    make.textContent = swmm.complete === false ? 'Generate diagnostic model ZIP' : 'Generate complete model ZIP';
    make.addEventListener('click', () => bundleThenDownload(make, swmm.run_id));
    bundleRow.append(make);
  }
  details.append(bundleRow);
  $('.bubble', node).append(details);
}

async function bundleThenDownload(button, runId) {
  const label = button.textContent;
  button.disabled = true; button.textContent = 'Packaging…';
  try {
    const data = await api(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/runs/${encodeURIComponent(runId)}/swmm-bundle`,
      {method: 'POST'});
    const artifact = data.artifact;
    if (!artifact) throw new Error('The model bundle returned no file');
    triggerDownload(artifact.download_url);
    const complete = data.swmm ? data.swmm.bundle_complete !== false : true;
    button.textContent = `${complete ? 'Complete model bundle ready' : 'Diagnostic model bundle ready'} · ${artifact.size_text || ''}`;
    button.disabled = false;
  } catch (error) {
    button.disabled = false; button.textContent = label;
    const note = document.createElement('p'); note.className = 'artifact-error';
    note.textContent = `Packaging failed: ${error.message}`;
    button.closest('.bubble').append(note);
  }
}

function addProcessArtifacts(node, artifacts) {
  const items = artifacts || [];
  if (!items.length) return;
  const details = document.createElement('details');
  details.className = 'swmm-section process-artifacts-section';
  const summary = document.createElement('summary');
  summary.textContent = `All workflow files (${items.length} files in original formats)`;
  details.append(summary);
  const note = document.createElement('p');
  note.className = 'artifact-note';
  note.textContent = 'Files retain their original formats, including CSV, GeoTIFF, JSON, reports and models.';
  details.append(note);
  const list = document.createElement('ul');
  list.className = 'artifact-list';
  for (const artifact of items) list.append(artifactRow(artifact));
  details.append(list);
  $('.bubble', node).append(details);
}

function addSuggestions(node, suggestions) {
  const items = (suggestions || []).slice(0, 3);
  if (!items.length) return;
  const box = document.createElement('div'); box.className = 'reply-suggestions';
  const hint = document.createElement('span'); hint.className = 'reply-suggestions-hint';
  hint.textContent = 'Suggested replies:'; box.append(hint);
  for (const item of items) {
    const chip = document.createElement('button');
    chip.type = 'button'; chip.textContent = item;
    chip.addEventListener('click', () => send(item));
    box.append(chip);
  }
  $('.bubble', node).append(box);
}

function renderReply(reply) {
  if (!reply) return;
  const node = message(reply.text, 'assistant', reply.ok ? reply.kind : 'error', reply);
  if (reply.route) node.dataset.route = reply.route;
  // A figure brings its own preview; what is left over is the older shape
  // (``images``) that this release still answers, so nothing that used to be
  // shown disappears.
  const previews = addVisualizations(node, reply.visualizations);
  addImages(node, reply.images, previews);
  if (reply.prepared_artifact) addArtifactList(node, [reply.prepared_artifact]);
  addProcessArtifacts(node, reply.process_artifacts);
  addSwmmSection(node, reply.swmm);
  // Examples are only offered where the agent is waiting for an answer.
  if (reply.needs_user_input) addSuggestions(node, reply.suggestions);
  if (reply.outlet_map) {
    const open = document.createElement('button');
    open.type = 'button'; open.textContent = reply.outlet_map.selection_mode === 'drawn_study_area'
      ? 'Draw a study area on the DEM' : 'Select an outlet on the map';
    open.addEventListener('click', () => pickerApply(reply.outlet_map));
    $('.bubble', node).append(open);
    pickerApply(reply.outlet_map);
  }
}

async function restoreArtifacts() {
  if (!sessionId) return;
  let data;
  try {
    data = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/artifacts`);
  } catch (error) {
    console.warn('Could not load outputs', error);
    return;
  }
  const figures = data.visualizations || [];
  if (!figures.length && !(data.process_artifacts || []).length && !data.swmm) return;
  const node = message('Figures and files from this session:', 'assistant', 'result');
  addVisualizations(node, figures);
  addProcessArtifacts(node, data.process_artifacts);
  addSwmmSection(node, data.swmm);
}

function updateSession(session) {
  if (!session) return;
  sessionId = session.session_id; localStorage.setItem(storageKey, sessionId);
  $('#session-id').textContent = sessionId;
  const failedWatershed = session.watershed_qa_passed === false
    && session.state === 'awaiting_watershed_confirmation';
  $('#session-state').textContent = failedWatershed ? stateLabels.watershed_qa_failed
    : (stateLabels[session.state] || session.state);
  $('#pending-action').textContent = session.pending_action || '—';
  $('#run-status').textContent = session.run_status || (session.has_run ? 'Waiting for status' : 'Not submitted');
  $('#session-summary').textContent = session.conversation_summary || '—';
  $('#cancel-run').hidden = session.state !== 'hydro_running';
  taskWorkspace=session.workspace||null;
  renderTaskWorkspace();
  if (revisionSession!==`${sessionId}:${session.state}:${session.updated_at}`) {
    revisionSession=`${sessionId}:${session.state}:${session.updated_at}`;
    loadTaskRevisions();
  }
  const progress = stateProgress[session.state] ?? 0;
  const thresholds = [1, 30, 52, 84, 100];
  const step=Math.max(1,thresholds.findIndex(value=>progress<=value)+1);
  $('#progress-value').textContent = `Step ${step}/5`; $('#progress-bar').style.width = `${step/5*100}%`;
  document.querySelectorAll('#workflow-steps li').forEach((item, index) => {
    item.classList.toggle('done', progress > thresholds[index]);
    item.classList.toggle('active', progress <= thresholds[index] && (index === 0 || progress > thresholds[index - 1]));
  });
  if (session.state === 'hydro_running') schedulePoll(); else stopPoll();
}

function historyDate(value) {
  if (!value) return 'Unknown time';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Unknown time';
  return new Intl.DateTimeFormat('en-GB', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'
  }).format(date);
}

function renderHistory(items) {
  historyItems = items;
  const visible = new Set(items.map(item => item.session_id));
  for (const identifier of selectedHistory) if (!visible.has(identifier)) selectedHistory.delete(identifier);
  historyList.replaceChildren();
  updateHistoryControls();
  if (!items.length) {
    const empty = document.createElement('p'); empty.className = 'history-empty';
    empty.textContent = 'No past conversations'; historyList.append(empty); return;
  }
  for (const item of items) {
    const row = document.createElement('div'); row.className = 'history-item';
    row.classList.toggle('current', item.session_id === sessionId);
    row.classList.toggle('selectable', historyManaging);
    row.classList.toggle('selected', selectedHistory.has(item.session_id));
    let checkbox = null;
    if (historyManaging) {
      checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.className = 'history-checkbox';
      checkbox.checked = selectedHistory.has(item.session_id); checkbox.disabled = historyDeleting;
      checkbox.dataset.sessionId = item.session_id;
      checkbox.setAttribute('aria-label', `Select conversation: ${item.title || 'New session'}`);
      checkbox.addEventListener('change', () => {
        if (checkbox.checked) selectedHistory.add(item.session_id); else selectedHistory.delete(item.session_id);
        row.classList.toggle('selected', checkbox.checked); updateHistoryControls();
      });
      row.append(checkbox);
    }

    const open = document.createElement('button'); open.type = 'button'; open.className = 'history-open';
    const title = document.createElement('strong'); title.textContent = item.title || 'New session';
    const meta = document.createElement('small');
    meta.textContent = `${historyDate(item.updated_at)} · ${stateLabels[item.state] || item.state || 'Unknown status'}`;
    open.append(title, meta);
    open.addEventListener('click', async () => {
      if (busy || historyDeleting) return;
      if (historyManaging) { checkbox.click(); return; }
      if (item.session_id === sessionId) return;
      stopPoll(); sessionId = item.session_id; localStorage.setItem(storageKey, sessionId);
      await restoreSession(); input.focus();
    });

    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'history-delete';
    remove.textContent = 'Delete'; remove.setAttribute('aria-label', `Delete conversation: ${item.title || item.session_id}`);
    remove.addEventListener('click', async event => {
      event.stopPropagation(); await deleteHistory(item.session_id, item.title || 'this conversation');
    });
    row.append(open);
    if (!historyManaging) row.append(remove);
    historyList.append(row);
  }
}

function historyStatus(text, isError = false) {
  const status = $('#history-status'); status.textContent = text;
  status.hidden = !text; status.classList.toggle('is-error', isError);
}

function updateHistoryControls() {
  $('#history-toolbar').hidden = !historyManaging;
  const manage = $('#history-manage');
  manage.textContent = historyManaging ? 'Done' : 'Select conversations';
  manage.setAttribute('aria-expanded', String(historyManaging));
  manage.disabled = historyDeleting || (!historyManaging && !historyItems.length);
  $('#history-selected-count').textContent = `${selectedHistory.size} selected`;
  $('#history-delete-selected').disabled = historyDeleting || !selectedHistory.size;
  const all = $('#history-select-all');
  all.checked = historyItems.length > 0 && selectedHistory.size === historyItems.length;
  all.indeterminate = selectedHistory.size > 0 && selectedHistory.size < historyItems.length;
  all.disabled = historyDeleting || !historyItems.length;
}

async function loadHistory() {
  try {
    const data = await api('/api/v1/sessions?limit=200');
    renderHistory(data.sessions || []);
    return true;
  } catch (error) {
    historyStatus(`Could not load history: ${error.message}`, true); return false;
  }
}

async function deleteSelectedHistory() {
  if (historyDeleting || !selectedHistory.size) return;
  if (busy) { historyStatus('Wait for the current operation to finish before deleting.'); return; }
  try {
    const capabilities = await api('/api/v1/health');
    if (!capabilities.batch_session_product_cleanup) {
      historyStatus('Restart the web service and refresh the page to enable bulk deletion.', true); return;
    }
  } catch (error) { historyStatus(error.message, true); return; }
  const selected = historyItems.filter(item => selectedHistory.has(item.session_id));
  const identifiers = selected.map(item => item.session_id);
  if (busy || !identifiers.length || !confirm(`Permanently delete the ${identifiers.length} selected conversations and their models, results, figures and revisions? Shared files and original inputs are kept.`)) return;
  historyDeleting = true; busy = true; sendButton.disabled = true;
  renderHistory(historyItems); renderTaskWorkspace();
  historyStatus(`Deleting ${identifiers.length} conversations and their outputs…`);
  try {
    const result = await api('/api/v1/sessions/batch-delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_ids: identifiers})
    });
    for (const identifier of result.deleted) selectedHistory.delete(identifier);
    const removed = new Set(result.deleted);
    historyItems = historyItems.filter(item => !removed.has(item.session_id));
    if (result.deleted.includes(sessionId)) {
      stopPoll(); localStorage.removeItem(storageKey); sessionId = null;
      try { await createSession(); }
      catch { conversation.replaceChildren(); message('The selected conversations were deleted. Refresh the page to create a new session.', 'assistant', 'error'); }
    } else await loadHistory();
    const failed = result.failed || [];
    const retained = result.cleanup?.retained_ids?.length || 0;
    let text = `Deleted ${result.deleted.length} conversations and their outputs.`;
    if (retained) text += ` Kept ${retained} files shared with other tasks.`;
    if (failed.length) {
      text += `\n${failed.length} conversations could not be deleted. You can retry:`;
      const names = new Map(selected.map(item => [item.session_id, item.title || 'New session']));
      text += '\n' + failed.slice(0, 3).map(item => `${names.get(item.session_id) || 'Conversation'}: ${item.detail}`).join('\n');
      if (failed.length > 3) text += `\n${failed.length - 3} other conversations remain selected for retry.`;
    } else { historyManaging = false; selectedHistory.clear(); }
    historyStatus(text, failed.length > 0);
  } catch (error) {
    const loaded = await loadHistory();
    if (loaded && identifiers.includes(sessionId) && !historyItems.some(item => item.session_id === sessionId)) {
      stopPoll(); localStorage.removeItem(storageKey); sessionId = null;
      try { await createSession(); } catch { /* A refresh can recreate the session. */ }
    }
    historyStatus(`Deletion interrupted: ${error.message}\nReview the list and retry any conversations still present.`, true);
  } finally {
    historyDeleting = false; busy = false; sendButton.disabled = false;
    renderHistory(historyItems); renderTaskWorkspace();
  }
}

async function deleteHistory(identifier, title) {
  if (busy) return;
  try {
    const capabilities = await api('/api/v1/health');
    if (!capabilities.session_product_cleanup) {
      message('Restart the web service and refresh the page before deleting a conversation.', 'assistant', 'error');
      return;
    }
  } catch (error) {
    message(error.message, 'assistant', 'error'); return;
  }
  if (busy || !confirm(`Permanently delete “${title}” and its models, results, figures and revisions? Shared files and original inputs are kept.`)) return;
  busy = true; sendButton.disabled = true;
  try {
    const result = await api(`/api/v1/sessions/${encodeURIComponent(identifier)}`, {method: 'DELETE'});
    if (identifier === sessionId) {
      stopPoll(); localStorage.removeItem(storageKey); sessionId = null;
      await createSession();
    } else {
      await loadHistory();
    }
    const retained = result.cleanup?.retained_ids?.length || 0;
    if (retained) message(`Conversation deleted. Kept ${retained} files shared with other tasks.`, 'assistant');
  } catch (error) {
    message(error.message, 'assistant', 'error');
  } finally {
    busy = false; sendButton.disabled = false;renderTaskWorkspace();
  }
}

async function health() {
  try {
    const data = await api('/api/v1/health');
    $('#service-state').classList.add('ready'); $('#service-state span').textContent = 'Web service ready';
    const chip = $('#model-chip');
    const llm = data.llm || {};
    const name = llm.response_model || llm.model;
    const effort = llm.reasoning_effort ? ` · reasoning ${llm.reasoning_effort}` : (llm.deepseek_thinking ? ` · thinking ${llm.deepseek_thinking === 'enabled' ? 'on' : 'off'}` : '');
    chip.textContent = llm.configured && name ? `${name}${effort} · Settings` : 'Model disconnected · Settings';
    chip.title = llm.configured ? `Configured model: ${llm.model || 'unspecified'}\n${llm.response_model ? `Model reported by API: ${llm.response_model}` : 'Showing configuration before the first API response'}\nOpen model settings` : (llm.reason || 'Open model settings');
    chip.classList.toggle('ready', !!llm.configured);
  } catch (error) {
    $('#service-state').classList.remove('ready'); $('#service-state span').textContent = error.message;
    $('#model-chip').classList.remove('ready'); $('#model-chip').textContent = 'Model status unknown · Settings';
  }
}

async function createSession() {
  const session = await api('/api/v1/sessions', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({language: 'en'})
  });
  conversation.replaceChildren(); updateSession(session);
  message('Ask an engineering question or start an assessment. Select a district and draw the study area on the DEM, or enter outlet coordinates. Review the boundary and rainfall before running the model.');
  await loadHistory();
}

async function restoreSession() {
  if (!sessionId) return createSession();
  try {
    const session = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}`);
    updateSession(session); conversation.replaceChildren();
    for (const turn of session.conversation || []) {
      const meta = ['general_model_knowledge', 'builtin_knowledge', 'saved_task_context'].includes(turn.answer_source)
        ? {answer_source: turn.answer_source, audit_scope: turn.answer_source === 'saved_task_context' ? 'task_context' : 'general_knowledge'} : null;
      message(turn.text, turn.role === 'user' ? 'user' : 'assistant', 'message', meta);
    }
    if (!(session.conversation || []).length) message('Session restored. Ask an engineering question or enter a river name or outlet coordinates.');
    // The transcript is text; the figures and their downloads are restored from
    // the session's own record, so a reloaded page can still hand over the files.
    await restoreArtifacts();
    await loadHistory();
  } catch { localStorage.removeItem(storageKey); sessionId = null; await createSession(); }
}

async function send(text) {
  text = String(text || '').trim(); if (!text || busy || !sessionId) return;
  message(text, 'user'); input.value = ''; resizeInput(); busy = true; sendButton.disabled = true;
  renderTaskWorkspace();
  const removeTyping = typing();
  try {
    const data = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})
    });
    removeTyping(); renderReply(data.reply); updateSession(data.session); await loadHistory();
  } catch (error) { removeTyping(); message(error.message, 'assistant', 'error'); }
  finally { busy = false; sendButton.disabled = false; input.focus();renderTaskWorkspace(); void health(); }
}

function stopPoll() { if (pollTimer) clearTimeout(pollTimer); pollTimer = null; }
function schedulePoll() { stopPoll(); pollTimer = setTimeout(reconcile, 4000); }
async function reconcile() {
  if (!sessionId || busy) { schedulePoll(); return; }
  try {
    const data = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/reconcile`, {method: 'POST'});
    if (data.reply) renderReply(data.reply); updateSession(data.session); await loadHistory();
  } catch (error) { console.warn('Status refresh failed', error); schedulePoll(); }
}

function resizeInput() { input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight,150)}px`; }
composer.addEventListener('submit', event => { event.preventDefault(); send(input.value); });
input.addEventListener('input', resizeInput);
input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); composer.requestSubmit(); } });
$('#new-session').addEventListener('click', async () => { if (!busy && confirm('Create a new session? Existing conversations will be kept.')) await createSession(); });
$('#history-manage').addEventListener('click', () => {
  if (busy || historyDeleting) return;
  historyManaging = !historyManaging; selectedHistory.clear();
  renderHistory(historyItems); historyStatus('');
});
$('#history-select-all').addEventListener('change', event => {
  if (historyDeleting) return;
  selectedHistory.clear();
  if (event.target.checked) for (const item of historyItems) selectedHistory.add(item.session_id);
  renderHistory(historyItems);
});
$('#history-delete-selected').addEventListener('click', deleteSelectedHistory);
$('#cancel-run').addEventListener('click', async () => {
  if (!sessionId || busy || !confirm('Cancel the current run?')) return;
  busy = true;
  renderTaskWorkspace();
  try { const data = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/cancel`, {method:'POST'}); renderReply(data.reply); updateSession(data.session); }
  catch (error) { message(error.message, 'assistant', 'error'); }
  finally { busy = false;renderTaskWorkspace(); }
});
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => send(button.dataset.prompt)));
document.querySelectorAll('[data-action="scope"]').forEach(button => button.addEventListener('click', async () => {
  try { const data = await api('/api/v1/scope'); message(data.text); }
  catch (error) { message(error.message, 'assistant', 'error'); }
}));

/* ---- 交互式选址: 用户在地图上点选出水口 ---------------------------------- */
/* 底图由空间 MCP 渲染，接口一并返回 bbox_wgs84（绘图区四边经纬度)与 axes_rect
   （绘图区在整张图里的比例位置)。点击先归一化成整图比例，再裁到绘图区，
   最后线性插值成经纬度——没有底图瓦片，也不需要联网。 */
const picker = {
  geo: null,
  selected: null,
  river: localStorage.getItem('slope-picker-river') || ''
};

function pickerShow() {
  $('#outlet-picker').hidden = false;
  const field = $('#picker-river');
  if (!field.value) field.value = picker.river;
  field.focus(); field.select();
}
function pickerHide() { $('#outlet-picker').hidden = true; }
function pickerHint(text, isError = false) {
  const node = $('#picker-hint');
  node.textContent = text;
  node.classList.toggle('is-error', Boolean(isError));
}
function pickerMark(position) {
  const marker = $('#picker-marker');
  if (!position) { marker.hidden = true; return; }
  marker.style.left = `${position.fx * 100}%`;
  marker.style.top = `${position.fy * 100}%`;
  marker.hidden = false;
}
function pickerClear() {
  /* 载入失败时必须把上一张图撤掉: 留着旧底图会让用户在一块已经不是当前河流的
     图幅上点选，取回来的经纬度看着合理却对不上。 */
  picker.geo = null;
  picker.selected = null;
  const image = $('#picker-map');
  image.hidden = true;
  image.removeAttribute('src');
  $('#picker-mapwrap').classList.remove('has-map');
  $('#picker-selected-text').textContent = 'No point selected';
  $('#picker-selected-note').textContent = '';
  $('#picker-use').disabled = true;
  pickerMark(null);
}
function pickerGeoToFraction(longitude, latitude) {
  if (!picker.geo) return null;
  const [west, south, east, north] = picker.geo.bbox_wgs84;
  const [left, bottom, width, height] = picker.geo.axes_rect;
  const ax = (longitude - west) / (east - west);
  const ay = (latitude - south) / (north - south);
  return {fx: left + ax * width, fy: 1 - (bottom + ay * height)};
}
function pickerFractionToGeo(fx, fy) {
  if (!picker.geo) return null;
  const [west, south, east, north] = picker.geo.bbox_wgs84;
  const [left, bottom, width, height] = picker.geo.axes_rect;
  const ax = (fx - left) / width;
  const ay = (1 - fy - bottom) / height;
  if (ax < -0.002 || ax > 1.002 || ay < -0.002 || ay > 1.002) return null;
  return {
    longitude: west + Math.min(Math.max(ax, 0), 1) * (east - west),
    latitude: south + Math.min(Math.max(ay, 0), 1) * (north - south)
  };
}
function pickerSelect(longitude, latitude, note) {
  picker.selected = {longitude, latitude};
  $('#picker-selected-text').textContent =
    `${longitude.toFixed(5)}°E, ${latitude.toFixed(5)}°N`;
  $('#picker-selected-note').textContent = note;
  $('#picker-use').disabled = false;
  pickerMark(pickerGeoToFraction(longitude, latitude));
}
function pickerSummarize(data) {
  const matched = (data.matched_names || []).join(', ');
  if (!matched) {
    if ((data.map || data).url && data.location_hint) {
      return 'The river name did not match the local dataset. This suggested location is unverified. '
        + 'Check the river reach and choose its outlet on the blue line.';
    }
    // 名字对不上就没有底图可点，所以这里必须给出下一步: 坐标兜底或换名字。
    return data.message || 'The river could not be located. Enter a district, nearby landmark or outlet coordinates.';
  }
  return `Matched: ${matched}. Choose an outlet on the blue line and check its coordinates before delineation.`;
}
function pickerApply(data) {
  if (data.selection_mode === 'drawn_study_area') { areaApply(data); return; }
  pickerClear();
  picker.river = data.river_name || '';
  $('#picker-river').value = picker.river;
  localStorage.setItem('slope-picker-river', picker.river);
  const map = data.map || data;
  picker.geo = map.url ? map : null;
  if (picker.geo) {
    $('#picker-map').src = map.url;
    $('#picker-map').hidden = false;
    $('#picker-mapwrap').classList.add('has-map');
  }
  pickerHint(pickerSummarize(data));
  pickerShow();
}
async function pickerLoad() {
  const name = $('#picker-river').value.trim();
  if (!name) { pickerHint('Enter a river name.', true); return; }
  if (!sessionId) { pickerHint('The session is starting. Try again shortly.', true); return; }
  if (busy) { pickerHint('Wait for the previous message before loading the map.', true); return; }
  const button = $('#picker-load');
  button.disabled = true;
  pickerClear();
  pickerHint('Matching the river and rendering the map…');
  try {
    const data = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/outlet-candidates`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({river_name: name})
    });
    pickerApply(data);
    if (data.session) updateSession(data.session);
  } catch (error) {
    pickerHint(error.message, true);
  } finally {
    button.disabled = false;
  }
}
function pickerParseCoordinates(text) {
  /* 河名查不到水系时，这是唯一的出路: 给一个大概坐标，走与手动输入相同的划分路径。
     只接受能明确读成"经度在前, 纬度在后"的两数。 */
  const numbers = String(text || '').match(/-?\d+(?:\.\d+)?/g) || [];
  if (numbers.length !== 2) return null;
  const longitude = Number(numbers[0]);
  const latitude = Number(numbers[1]);
  if (!(-180 <= longitude <= 180) || !(-90 <= latitude <= 90)) return null;
  if (longitude <= latitude) return null;      // 只可能是"经度, 纬度"
  return {longitude, latitude};
}
function pickerUseCoordinates() {
  if (busy) { pickerHint('Wait for the previous message before delineation.', true); return; }
  const point = pickerParseCoordinates($('#picker-coord').value);
  if (!point) {
    pickerHint('Enter longitude and latitude as two numbers, for example 105.90, 29.35.', true);
    return;
  }
  pickerHide();
  send(`${point.longitude.toFixed(5)}, ${point.latitude.toFixed(5)}`);
}
$('#picker-coord-use').addEventListener('click', pickerUseCoordinates);
$('#picker-coord').addEventListener('keydown', event => {
  if (event.key === 'Enter') { event.preventDefault(); pickerUseCoordinates(); }
});
$('#picker-map').addEventListener('click', event => {
  const rect = event.currentTarget.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const point = pickerFractionToGeo((event.clientX - rect.left) / rect.width,
                                    (event.clientY - rect.top) / rect.height);
  if (!point) { pickerHint('Click inside the terrain map.', true); return; }
  pickerSelect(point.longitude, point.latitude, 'Map selection');
  pickerHint(`Selected ${point.longitude.toFixed(5)}°E, ${point.latitude.toFixed(5)}°N. `
    + 'Check the point and select “Delineate from this point”.');
});
$('#picker-use').addEventListener('click', () => {
  if (!picker.selected) return;
  if (busy) { pickerHint('Wait for the previous message before delineating from this point.', true); return; }
  const {longitude, latitude} = picker.selected;
  pickerHide();
  send(`${longitude.toFixed(5)}, ${latitude.toFixed(5)}`);
});
$('#picker-load').addEventListener('click', pickerLoad);
$('#picker-close').addEventListener('click', pickerHide);
$('#picker-river').addEventListener('keydown', event => {
  if (event.key === 'Enter') { event.preventDefault(); pickerLoad(); }
});
$('#outlet-picker').addEventListener('click', event => {
  if (event.target === event.currentTarget) pickerHide();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('#outlet-picker').hidden) pickerHide();
});
document.querySelectorAll('[data-action="outlet-picker"]').forEach(button =>
  button.addEventListener('click',()=>{areaSetMode('river');areaShow();}));

/* User-drawn domains: coordinates come from the DEM's WGS84 display grid. */
const area = {geo:null,points:[],closed:false,district:'',session:null,context:null,undo:[],redo:[],zoom:1,drag:null,suppressClick:false,qa:null};
function areaSetMode(mode) {
  $('#area-mode').value=mode;
  $('#area-district').hidden=mode!=='district';$('#area-district-label').hidden=mode!=='district';
  $('#area-river').hidden=mode!=='river';$('#area-coordinates').hidden=mode!=='coordinates';
  if (!area.geo) areaHint(mode==='district'?'The map is clipped to the district. Calculations use the boundary you draw and confirm.'
    :'Load the DEM by river name or coordinates, then draw a closed boundary.');
}
function areaHint(text, error=false) {
  $('#area-hint').textContent=text; $('#area-hint').classList.toggle('is-error',error);
}
function areaDraw() {
  const locked=busy||taskWorkspace?.actions?.edit_boundary===false;
  $('#area-assessment-mode').disabled=locked;
  const svg=$('#area-lines'); svg.replaceChildren();
  if (area.geo && area.points.length) {
    const [west,south,east,north]=area.geo.bbox_wgs84;
    const [left,bottom,width,height]=area.geo.axes_rect;
    const points=area.points.map(([x,y])=>[
      (left+(x-west)/(east-west)*width)*1000,
      (1-bottom-(y-south)/(north-south)*height)*1000]);
    const line=document.createElementNS('http://www.w3.org/2000/svg',area.closed?'polygon':'polyline');
    line.setAttribute('points',points.map(p=>p.join(',')).join(' '));
    line.setAttribute('fill',area.closed?'rgba(237,94,32,.18)':'none');
    line.setAttribute('stroke','#ed5e20'); line.setAttribute('stroke-width','3'); svg.append(line);
    points.forEach(([x,y],index)=>{const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');
      dot.dataset.vertex=String(index);dot.classList.add('editable-vertex');
      dot.setAttribute('cx',x);dot.setAttribute('cy',y);dot.setAttribute('r',6);
      dot.setAttribute('fill','#fff');dot.setAttribute('stroke','#ed5e20');svg.append(dot);});
  }
  $('#area-status').textContent=area.points.length?`${area.points.length} vertices · ${area.closed?'Closed':'Drawing'}`:'No boundary drawn';
  drawDiagnosticMarkers(svg);
  $('#area-undo').disabled=!area.undo.length || locked;
  $('#area-redo').disabled=!area.redo.length || locked;
  $('#area-clear').disabled=!area.points.length||locked;
  $('#area-finish').disabled=area.points.length<3 || area.closed||locked;
  $('#area-use').disabled=!area.closed || locked;
}
function areaClearMap() {
  clearTimeout(draftTimer);area.undo=[];area.redo=[];area.qa=null;areaZoom(1);
  area.geo=null;area.points=[];area.closed=false;area.district='';area.context=null;
  $('#area-map').hidden=true;$('#area-map').removeAttribute('src');
  $('#area-mapwrap').classList.remove('has-map');areaDraw();
}
async function areaShow() {
  if (!sessionId || busy) return;
  $('#area-picker').hidden=false;
  if (area.session!==sessionId) {areaClearMap();area.session=sessionId;$('#area-assessment-mode').value=taskWorkspace?.assessment_mode||'regional_screening';areaPolicyNote();}
  const identifier=sessionId;
  try {
    const data=await api(`/api/v1/sessions/${encodeURIComponent(identifier)}/administrative-areas`);
    if (identifier!==sessionId) return;
    const select=$('#area-district');const selected=area.district||select.value||'永川区';
    select.replaceChildren();
    (data.areas||[]).forEach(item=>{const option=document.createElement('option');
      option.value=item.name;option.textContent=item.name;select.append(option);});
    if ([...select.options].some(option=>option.value===selected)) select.value=selected;
  } catch(error) {areaHint(error.message,true);}
}
function areaApply(data) {
  areaClearMap();area.session=sessionId;
  area.district=data.administrative_area;
  area.context=data;
  areaSetMode(data.location_mode|| (area.district?'district':'river'));
  if (data.river_name) $('#area-river').value=data.river_name;
  if (data.map_center) $('#area-coordinates').value=`${data.map_center.longitude}, ${data.map_center.latitude}`;
  area.geo=data.map||data;
  $('#area-map').src=area.geo.url+(area.geo.url.includes('?')?'&':'?')+'reload='+Date.now();
  $('#area-map').hidden=false;
  $('#area-mapwrap').classList.add('has-map');
  $('#area-picker').hidden=false;
  const select=$('#area-district');
  if (area.district && ![...select.options].some(option=>option.value===area.district)) {
    const option=document.createElement('option');option.value=area.district;option.textContent=area.district;select.append(option);
  }
  select.value=area.district;
  areaHint(data.message||'Add vertices along the boundary, then select “Close boundary”.');areaDraw();
}
$('#area-load').addEventListener('click',async()=>{
  if (!sessionId||busy) return;
  const mode=$('#area-mode').value;let location;
  if (mode==='district') location={administrative_area:$('#area-district').value};
  else if (mode==='river') {
    const name=$('#area-river').value.trim();if (!name) {areaHint('Enter the river name.',true);return;}
    location={river_name:name};
  } else {
    const point=pickerParseCoordinates($('#area-coordinates').value);
    if (!point) {areaHint('Enter two numbers in longitude, latitude order.',true);return;}
    location=point;
  }
  areaClearMap();busy=true;sendButton.disabled=true;$('#area-load').disabled=true;
  renderTaskWorkspace();
  for (const id of ['area-mode','area-district','area-river','area-coordinates']) $('#'+id).disabled=true;
  areaHint(mode==='district'?'Clipping and rendering the district DEM…':'Locating and loading the DEM and rivers…');
  try {
    const data=await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/study-area-map`,{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(location)});
    if (data.map?.url) areaApply(data);else areaHint(data.message||'The location could not be resolved. Enter a district or map-centre coordinates.',true);
    updateSession(data.session);
  } catch(error) {areaHint(error.message,true);}
  finally {busy=false;sendButton.disabled=false;$('#area-load').disabled=false;
    for (const id of ['area-mode','area-district','area-river','area-coordinates']) $('#'+id).disabled=false;
    areaDraw();renderTaskWorkspace();}
});
$('#area-mode').addEventListener('change',()=>{areaClearMap();areaSetMode($('#area-mode').value);areaHint('Choose a location and reload the map before drawing the boundary.');});
for (const id of ['area-river','area-coordinates']) $('#'+id).addEventListener('input',()=>{areaClearMap();areaHint('Location changed. Reload the map.');});
$('#area-district').addEventListener('change',()=>{areaClearMap();areaHint('District changed. Reload the map.');});
$('#area-mapwrap').addEventListener('click',event=>{
  if (area.suppressClick) {area.suppressClick=false;return;}
  if (event.target.closest('.editable-vertex,.diagnostic-marker')) return;
  if (!area.geo||area.closed||busy) return;
  if (area.points.length>=499) {areaHint('The boundary has too many vertices. Simplify it.',true);return;}
  const rect=$('#area-map').getBoundingClientRect();
  const fx=(event.clientX-rect.left)/rect.width,fy=(event.clientY-rect.top)/rect.height;
  const [left,bottom,width,height]=area.geo.axes_rect;
  const ax=(fx-left)/width,ay=(1-fy-bottom)/height;
  if (ax<0||ax>1||ay<0||ay>1) {areaHint('Click inside the DEM map.',true);return;}
  const [west,south,east,north]=area.geo.bbox_wgs84;
  const point=[west+ax*(east-west),south+ay*(north-south)];
  const previous=area.points.at(-1);
  if (previous&&Math.hypot(point[0]-previous[0],point[1]-previous[1])<1e-7) return;
  areaRemember();area.points.push(point);areaDraw();saveAreaDraft();areaHint('Add at least three vertices before closing the boundary.');
});
$('#area-undo').addEventListener('click',()=>{if (busy||!area.undo.length) return;area.redo.push(areaGeometry());Object.assign(area,area.undo.pop());areaDraw();saveAreaDraft();});
$('#area-redo').addEventListener('click',()=>{if (busy||!area.redo.length) return;area.undo.push(areaGeometry());Object.assign(area,area.redo.pop());areaDraw();saveAreaDraft();});
$('#area-clear').addEventListener('click',()=>{if (busy) return;areaRemember();area.points=[];area.closed=false;areaDraw();saveAreaDraft();});
$('#area-finish').addEventListener('click',()=>{if (busy||area.points.length<3) return;areaRemember();area.closed=true;areaDraw();saveAreaDraft();areaHint('Boundary closed. Submit it for coverage and drainage checks.');});
$('#area-use').addEventListener('click',async()=>{
  if (busy||!area.closed||area.session!==sessionId||!area.geo) return;
  clearTimeout(draftTimer);
  busy=true;sendButton.disabled=true;areaDraw();areaHint('Checking coverage, potential outlets and upstream inflow…');
  try {
    const coordinates=[...area.points,area.points[0]];
    const data=await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/study-area`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({administrative_area:area.district||null,map_id:area.context?.map_id,coordinates,assessment_mode:$('#area-assessment-mode').value})});
    message(`Check the drawn boundary at ${area.context?.location_label||area.district}.`,'user');
    renderReply(data.reply);updateSession(data.session);$('#area-picker').hidden=true;await loadHistory();
  } catch(error) {areaHint(error.message,true);}
  finally {busy=false;sendButton.disabled=false;areaDraw();renderTaskWorkspace();}
});
$('#area-close').addEventListener('click',()=>{$('#area-picker').hidden=true;});
$('#area-picker').addEventListener('click',event=>{if (event.target===event.currentTarget) event.currentTarget.hidden=true;});
document.addEventListener('keydown',event=>{if (event.key==='Escape') $('#area-picker').hidden=true;});
document.querySelectorAll('[data-action="area-picker"]').forEach(button=>button.addEventListener('click',areaShow));

(async () => { await health(); await restoreSession(); input.focus(); })();
