// UI guidance only. The Agent remains the authority for routing and pair validation.
export const inputTypes = [
  {value: 'single', label: 'Single image'},
  {value: 'change', label: 'Before / After'},
  {value: 'fusion', label: 'Optical + SAR'},
];
export const modality = slot => slot.override || slot.info?.modality;
export function obviousInputType(slots) {
  const ready = slots.filter(slot => slot.file && slot.info && !slot.loading && !slot.error);
  if (ready.length === 2) {
    const types = ready.map(modality);
    if (types.every(type => type === 'optical')) return 'change';
    if (types.includes('optical') && types.includes('sar')) return 'fusion';
  }
  if (ready.length === 1 && slots.filter(slot => slot.file).length === 1 && modality(ready[0]) === 'optical') return 'single';
  return null;
}
export function fusionProblems(slot) {
  if (!slot.file || !slot.info) return [];
  const info = slot.info, expected = modality(slot) === 'sar' ? 2 : 13;
  const problems = [];
  if (!/\.tiff?$/i.test(slot.file.name)) problems.push('Model 4 needs a GeoTIFF; PNG/JPG are display/demo inputs only.');
  if (info.band_count !== expected) problems.push(`Model 4 needs a ${expected}-band GeoTIFF; this file has ${info.band_count} bands.`);
  if (info.width !== 256 || info.height !== 256) problems.push(`Model 4 needs 256 × 256 pixels; this file is ${info.width} × ${info.height}.`);
  if (!info.crs) problems.push('Model 4 needs a non-null CRS; this file has no georeferencing.');
  return problems;
}
export function exampleReason(kind, slots) {
  const ready = slots.filter(slot => slot.file && slot.info && !slot.loading && !slot.error);
  if (slots.some(slot => slot.loading)) return 'Wait for image inspection.';
  const types = ready.map(modality);
  if (kind === 'single') return ready.length === 1 && types[0] === 'optical' ? '' : 'Needs one optical image.';
  if (kind === 'change') return ready.length === 2 && types.every(type => type === 'optical') ? '' : 'Needs two optical images of the same area.';
  return ready.length === 2 && types.includes('optical') && types.includes('sar') && ready.every(slot => !fusionProblems(slot).length) ? '' : 'Needs an aligned 13-band optical + 2-band SAR GeoTIFF pair, 256 × 256.';
}
export function inputIssue(slots, query, inputType, realFusion) {
  if (!slots[0].file) return 'Add Image 1 to begin.';
  if (slots.some(slot => slot.loading)) return 'Wait for image inspection.';
  if (slots.some(slot => slot.file && (slot.error || !slot.info))) return 'Replace the invalid image or retry inspection.';
  const active = slots.filter(slot => slot.file), types = active.map(modality);
  if (types.every(type => type === 'sar')) return 'SAR-only input is unsupported. Add an optical image.';
  if (active.length < 2 && (inputType === 'change' || /\b(changed|changes|change analysis|before\s*(?:\/|and|&)\s*after|between (?:these|the) images)\b/i.test(query))) return 'Add a second image for change analysis.';
  if (active.length < 2 && inputType === 'fusion') return 'Add an optical + SAR pair for land-cover fusion.';
  const fusionPair = types.includes('sar') && types.includes('optical');
  if (realFusion && (fusionPair || inputType === 'fusion')) {
    const problems = active.flatMap(fusionProblems);
    if (problems.length) return problems[0];
    if (active.length === 2 && types[0] === types[1]) return 'Fusion needs one optical image and one SAR image; check the modality overrides.';
    if (active.length === 2 && active[0].info.crs !== active[1].info.crs) return 'Model 4 needs identical CRS for both GeoTIFFs.';
  }
  if (!query.trim()) return 'Enter a question or choose an example.';
  return '';
}
