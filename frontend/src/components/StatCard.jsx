import React from 'react';

export default function StatCard({ title, value, subtitle, icon: Icon, color = 'var(--cyber-blue)', badge }) {
  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      padding: '20px 24px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      position: 'relative',
      overflow: 'hidden',
      transition: 'transform 0.2s ease, border-color 0.2s ease',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)'
    }}>
      {/* Top accent strip */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '2px',
        backgroundColor: color
      }} />

      {/* Header with Title & Icon */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <span style={{
          fontSize: '12px',
          fontWeight: 600,
          color: 'var(--text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '0.6px'
        }}>
          {title}
        </span>
        {Icon && (
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '6px',
            backgroundColor: 'rgba(255, 255, 255, 0.04)',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Icon size={16} color={color} />
          </div>
        )}
      </div>

      {/* Main Metric Value */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '6px' }}>
        <span style={{
          fontSize: '26px',
          fontWeight: 700,
          color: '#ffffff',
          fontFamily: 'ui-monospace, monospace',
          letterSpacing: '-0.5px'
        }}>
          {value}
        </span>
        {badge && (
          <span style={{
            fontSize: '10px',
            fontWeight: 700,
            padding: '2px 6px',
            borderRadius: '4px',
            backgroundColor: 'rgba(56, 189, 248, 0.15)',
            color: 'var(--cyber-blue)'
          }}>
            {badge}
          </span>
        )}
      </div>

      {/* Subtitle / Context */}
      {subtitle && (
        <div style={{ fontSize: '12px', color: 'var(--text-faint)' }}>
          {subtitle}
        </div>
      )}
    </div>
  );
}
