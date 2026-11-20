import React, { forwardRef } from 'react';
export const Card = forwardRef(function Card({as: Tag = 'section', className = '', children, ...props}, ref) {
  return <Tag ref={ref} className={`ui-card ${className}`} {...props}>{children}</Tag>;
});
