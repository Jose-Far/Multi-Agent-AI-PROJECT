import React from 'react';

export default function StatusBadge({ status }) {
  const normStatus = (status || 'UNAVAILABLE').toUpperCase();
  
  let bgColor = 'rgba(148, 163, 184, 0.15)';
  let color = '#94a3b8';
  let border = 'rgba(148, 163, 184, 0.3)';
  
  if (normStatus === 'SUCCESS') {
    bgColor = 'rgba(16, 185, 129, 0.15)';
    color = '#34d399';
    border = 'rgba(16, 185, 129, 0.3)';
  } else if (normStatus === 'PARTIAL') {
    bgColor = 'rgba(251, 191, 36, 0.15)';
    color = '#fbbf24';
    border = 'rgba(251, 191, 36, 0.3)';
  } else if (normStatus === 'ERROR') {
    bgColor = 'rgba(244, 63, 94, 0.15)';
    color = '#fb7185';
    border = 'rgba(244, 63, 94, 0.3)';
  }

  return (
    <span style={{
      fontSize: '10px',
      fontWeight: 700,
      padding: '2px 7px',
      borderRadius: '4px',
      backgroundColor: bgColor,
      color: color,
      border: `1px solid `,
      display: 'inline-flex',
      alignItems: 'center',
      gap: '4px'
    }}>
      <span style={{ fontSize: '10px' }}>●</span> {normStatus}
    </span>
  );
}
