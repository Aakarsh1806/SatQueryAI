import React, { cloneElement, useId, useState } from 'react';
export function Tooltip({content, children}) {
  const id = useId(); const [open, setOpen] = useState(false);
  const child = cloneElement(children, {
    'aria-describedby': open ? [children.props['aria-describedby'], id].filter(Boolean).join(' ') : children.props['aria-describedby'],
    onFocus: event => {setOpen(true); children.props.onFocus?.(event);},
    onBlur: event => {setOpen(false); children.props.onBlur?.(event);},
    onKeyDown: event => {if (event.key === 'Escape') {event.preventDefault(); setOpen(false);} children.props.onKeyDown?.(event);},
  });
  return <span className="ui-tooltip-root" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>{child}{open && <span className="ui-tooltip" id={id} role="tooltip">{content}</span>}</span>;
}
