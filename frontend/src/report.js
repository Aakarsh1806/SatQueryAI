// Present existing job/trace fields; never manufacture stage timings or confidence.
export const stageNames = ['Validate', 'Route', 'Preprocess', 'Run', 'Integrate'];
const stageIndex = stage => ({validation:0, routing:1, preprocessing:2, model_load:3, model:3, integration:4})[stage];
export function pipelineSteps(job, trace) {
  const result = job.result || trace?.output?.result;
  const playback = result?.metadata.execution_type === 'pre-computed' || trace?.execution_type === 'pre-computed';
  const stub = result?.metadata.mode === 'stub';
  const completed = [trace?.validation?.valid === true, !!trace?.routing, !!trace?.preprocessing_steps?.length, !!result, trace?.output?.success === true];
  const error = trace?.errors?.at(-1);
  const failedAt = stageIndex(error?.stage);
  let current = -1;
  if (job.status === 'running') {
    const message = job.progress_message || '';
    if (/integrat/i.test(message)) current = 4;
    else if (/loading|download|loaded|LoRA|running|exclusive|specialist|synthetic/i.test(message)) current = 3;
    else if (/prepar|preprocess/i.test(message)) current = 2;
    else if (/rout/i.test(message)) current = 1;
    else if (/validat|detecting modalit/i.test(message)) current = 0;
  }
  return stageNames.map((name, index) => {
    let status = 'Not reported';
    if (job.status === 'queued') status = 'Waiting';
    else if (job.status === 'running') status = current === index ? 'In progress' : index < current ? 'Reached' : 'Waiting';
    else if (index === failedAt) status = job.status === 'cancelled' ? 'Cancelled' : 'Failed';
    else if (failedAt !== undefined && index > failedAt) status = 'Not run';
    else if (completed[index]) status = index === 3 && playback ? 'Playback' : index === 3 && stub ? 'Synthetic' : 'Complete';
    else if (job.status === 'cancelled') status = 'Not confirmed';
    const runtime = index === 3 && !playback ? result?.metadata.runtime_ms : null;
    return {name, status, duration: Number.isFinite(runtime) ? `${Math.round(runtime)} ms · ${stub ? 'synthetic runtime' : 'inference only'}` : 'Not recorded'};
  });
}
export function validationLabel(job) {
  if (job.model_id === 'grounding') return 'Unvalidated';
  if (job.model_id === 'fusion') return 'Not verified on a real pair';
  const status = job.result?.metadata.validation_status || '';
  if (/small|sanity|sample/i.test(status)) return 'Small-sample check';
  if (/test|BLEU|VRSBench|measured/i.test(status)) return 'Reported evaluation';
  return 'Validation not available';
}
const key = text => text.toLowerCase().replace(/[\s.,;:]+/g, ' ').trim();
export function reportNotes(job) {
  const metadata = job.result?.metadata;
  const warnings = [...(job.warnings || []), ...(metadata?.warnings || [])];
  if (metadata?.fallback_reason) warnings.push(`Using stub fallback: ${metadata.fallback_reason}`);
  const calibration = new Map(), notes = new Map();
  if (job.result?.confidence?.note) calibration.set(key(job.result.confidence.note), job.result.confidence.note);
  for (const warning of warnings) {
    if (typeof warning !== 'string') continue;
    if (job.result?.confidence?.note && key(warning) === key(job.result.confidence.note)) continue;
    // Move only calibration clauses to the score explanation; keep other clauses.
    for (const text of warning.split(/(?<=[.!?])\s+(?=[A-Z])|;\s*/).map(value => value.trim()).filter(Boolean)) {
      if (job.result && /uncalibrated|not calibrated|mean (?:generated-)?token probability|softmax score|aggregate (?:answer )?confidence/i.test(text)) {calibration.set(key(text), text); continue;}
      if ((job.routing?.reasons || []).some(reason => key(reason) === key(text))) continue; // Already displayed under Routing.
      const group = /georefer|\bCRS\b|footprint|resiz|resampl|\bbands?\b|nodata|no valid pixels|input format|image size/i.test(text) ? 'Input' : /domain gap|train|model|stub|placeholder|pre-computed|benchmark|caption|masks|verified|normalization|hallucinat/i.test(text) ? 'Model limits' : 'Interpretation';
      notes.set(key(text), {text, group});
    }
  }
  return {calibration: [...calibration.values()], notes: [...notes.values()].sort((a,b) => Number(/domain gap/i.test(b.text)) - Number(/domain gap/i.test(a.text)))};
}
export function traceParameters(trace, name) {
  if (!trace) return 'Not available';
  if (name === 'Validate') return JSON.stringify({valid:trace.validation?.valid, images:trace.validation?.images?.map(image => ({bands:image.band_count,size:[image.width,image.height],crs:image.crs})),code:trace.validation?.code});
  if (name === 'Route') return JSON.stringify(trace.routing || {});
  if (name === 'Preprocess') return JSON.stringify(trace.preprocessing_steps || []);
  if (name === 'Run') return JSON.stringify({chosen_model:trace.chosen_model,device:trace.device,parameters:trace.parameters,lifecycle:trace.lifecycle,execution_type:trace.execution_type});
  return JSON.stringify({success:trace.output?.success,warnings:trace.warnings,errors:trace.errors,cancelled:trace.cancelled});
}
export function downloadTrace(trace, id) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(trace,null,2)],{type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = `satquery-${id}.json`; link.click(); URL.revokeObjectURL(url);
}
