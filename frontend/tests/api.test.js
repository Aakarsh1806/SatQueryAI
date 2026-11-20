import assert from 'node:assert/strict';
import test from 'node:test';
import { analyzeForm, errorMessage, pending } from '../src/api.js';

test('form preserves before/after order and explicit overrides', () => {
  const first = new File(['a'], 'before.png');
  const second = new File(['b'], 'after.png');
  const form = analyzeForm([{file: first, override: ''}, {file: second, override: 'optical'}], 'What changed?');
  assert.deepEqual(form.getAll('files').map(file => file.name), ['before.png', 'after.png']);
  assert.equal(form.get('modality_overrides'), '[null,"optical"]');
  assert.equal(form.get('query'), 'What changed?');
});
test('empty second slot is omitted', () => {
  const form = analyzeForm([{file: new File(['a'], 'demo.png'), override: ''}, {file: null}], 'Describe');
  assert.equal(form.getAll('files').length, 1);
  assert.equal(form.get('modality_overrides'), '[null]');
});
test('API validation messages and terminal states', () => {
  assert.equal(errorMessage({detail: 'Two images required'}), 'Two images required');
  assert.equal(errorMessage({detail: [{msg: 'Missing file'}, {msg: 'Invalid modality'}]}), 'Missing file; Invalid modality');
  assert.equal(pending('queued'), true);
  assert.equal(pending('running'), true);
  assert.equal(pending('done'), false);
  assert.equal(pending('failed'), false);
});
