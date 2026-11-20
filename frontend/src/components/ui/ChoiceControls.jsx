import React, { useId, useRef } from 'react';

export function SegmentedControl({label, options, value, onChange, disabled = false}) {
  const id = useId();
  return <fieldset className="ui-segmented" disabled={disabled}><legend className="sr-only">{label}</legend>
    {options.map(option => <label key={option.value} className={value === option.value ? 'is-selected' : ''}>
      <input type="radio" name={id} value={option.value} checked={value === option.value} disabled={option.disabled} onChange={() => onChange(option.value)}/><span>{option.label}</span>
    </label>)}
  </fieldset>;
}

export function Tabs({label, items, value, onChange}) {
  const id = useId(); const buttons = useRef([]);
  function navigate(event, index) {
    const enabled = items.map((item, i) => !item.disabled ? i : null).filter(i => i !== null);
    let next;
    if (event.key === 'Home') next = enabled[0];
    else if (event.key === 'End') next = enabled.at(-1);
    else if (['ArrowRight', 'ArrowLeft'].includes(event.key)) next = enabled[(enabled.indexOf(index) + (event.key === 'ArrowRight' ? 1 : -1) + enabled.length) % enabled.length];
    if (next !== undefined) {event.preventDefault(); buttons.current[next]?.focus(); onChange(items[next].value);}
  }
  return <div className="ui-tabs"><div role="tablist" aria-label={label}>
    {items.map((item, index) => <button key={item.value} type="button" ref={element => {buttons.current[index] = element;}} role="tab" id={`${id}-tab-${index}`} aria-controls={`${id}-panel-${index}`} aria-selected={value === item.value} tabIndex={value === item.value ? 0 : -1} disabled={item.disabled} onClick={() => onChange(item.value)} onKeyDown={event => navigate(event, index)}>{item.label}</button>)}
  </div>{items.map((item, index) => <div key={item.value} role="tabpanel" id={`${id}-panel-${index}`} aria-labelledby={`${id}-tab-${index}`} tabIndex={0} hidden={value !== item.value}>{item.content}</div>)}</div>;
}
