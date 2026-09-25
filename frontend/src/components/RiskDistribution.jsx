import React from 'react';

export default function RiskDistribution({ score }) {
  const numScore = Number(score);
  if (isNaN(numScore)) return null;

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      padding: '24px',
      marginTop: '24px'
    }}>
      <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc', marginBottom: '16px' }}>
        Operational Risk Calibration (0-100)
      </h3>
      
      <div style={{ position: 'relative', height: '24px', borderRadius: '4px', display: 'flex', overflow: 'hidden', marginBottom: '8px' }}>
        <div style={{ flex: 3, backgroundColor: '#34d399', borderRight: '1px solid var(--bg-card)' }} /> {/* 0-29 Low */}
        <div style={{ flex: 3, backgroundColor: '#fbbf24', borderRight: '1px solid var(--bg-card)' }} /> {/* 30-59 Med */}
        <div style={{ flex: 2, backgroundColor: '#fb7185', borderRight: '1px solid var(--bg-card)' }} /> {/* 60-79 High */}
        <div style={{ flex: 2, backgroundColor: '#f43f5e' }} /> {/* 80-100 Crit */}

        
        {/* Dimming Overlay for unused portion */}
        <div style={{
          position: 'absolute',
          top: 0, bottom: 0, right: 0,
          width: `${100 - Math.max(0, Math.min(100, numScore))}%`,
          backgroundColor: 'var(--bg-dark)',
          opacity: 0.85,
          zIndex: 5,
          transition: 'width 0.5s ease-out'
        }} />

        {/* Marker */}
        <div style={{
          position: 'absolute',
          top: 0, bottom: 0,
          left: `${Math.max(0, Math.min(100, numScore))}%`,
          width: '2px',
          backgroundColor: '#fff',
          boxShadow: '0 0 8px rgba(255,255,255,0.8)',
          transform: 'translateX(-50%)',
          zIndex: 10
        }} />
      </div>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-faint)' }}>
        <span>0</span>
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
        <span>Critical</span>
        <span>100</span>
      </div>
    </div>
  );
}
