const origin = (import.meta.env?.VITE_API_BASE_URL || '').replace(/\/$/, '');
export const apiUrl = path => `${origin}${path}`;
export const pending = status => status === 'queued' || status === 'running';
export function errorMessage(body) {
  if (typeof body?.detail === 'string') return body.detail;
  if (Array.isArray(body?.detail)) return body.detail.map(item => item.msg).join('; ');
  return body?.error || 'The request could not be completed.';
}
export async function api(path, options = {}) {
  let response;
  try { response = await fetch(apiUrl(path), options); }
  catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Cannot reach the backend. Check that it is running, then retry.');
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(body) || `Request failed (${response.status}).`);
  return body;
}
export function analyzeForm(slots, query) {
  const form = new FormData();
  const active = slots.filter(slot => slot.file);
  active.forEach(slot => form.append('files', slot.file));
  form.append('query', query);
  form.append('modality_overrides', JSON.stringify(active.map(slot => slot.override || null)));
  return form;
}
export const modelNames = {rs_vlm: 'RS-VLM', grounding: 'Grounding DINO', change: 'Chg2Cap', fusion: 'SEN12MS Fusion'};

export async function downloadReport(requestId) {
  let response;
  try { response = await fetch(apiUrl(`/api/report/${requestId}`)); }
  catch { throw new Error('Cannot reach the backend to download the report.'); }
  if (!response.ok) throw new Error(errorMessage(await response.json().catch(() => null)));
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url; link.download = `satquery-${requestId}.html`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
