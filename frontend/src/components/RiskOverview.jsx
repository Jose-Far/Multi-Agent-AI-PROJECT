import React from 'react';
import RiskBadge from './RiskBadge';

export default function RiskOverview({ finalAssessment }) {
  const { verdict, risk_score, risk_level, confidence } = finalAssessment || {};
  
  const borderColor = verdict === 'phishing' || verdict === 'malicious' ? '#f43f5e' : 
                      verdict === 'legitimate' ? '#34d399' : 
                      verdict === 'unknown' ? '#94a3b8' : '#fbbf24';

  const formatScore = (score) => {
    if (score === null || score === undefined) return '-';
    return Number(score).toFixed(0);
  };

  const formatConfidence = (conf) => {
    if (conf === null || conf === undefined) return '0%';
    const pct = conf <= 1.0 ? conf * 100 : conf;
    return `%`;
  };

  if (verdict === 'unknown') {
    return (
      <div style={{
        backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
        borderTop: '3px solid ' + borderColor, borderRadius: '8px', padding: '24px',
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px',
        textAlign: 'center', height: '100%', justifyContent: 'center'
      }}>
        <h3 style={{ fontSize: '12px', textTransform: 'uppercase', color: 'var(--text-faint)', margin: 0, letterSpacing: '1px' }}>
          Overall Security Assessment
        </h3>
        <div style={{ fontSize: '42px', fontWeight: 700, fontFamily: 'monospace', color: '#f8fafc', lineHeight: 1 }}>-</div>
        <RiskBadge type="verdict" value="UNKNOWN" size="large" />
        <div style={{ fontSize: '14px', color: 'var(--text-muted)', marginTop: '8px' }}>
          Insufficient evidence for a reliable assessment.
        </div>
      </div>
    );
  }

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderTop: `3px solid `,
      borderRadius: '8px',
      padding: '24px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '16px',
      textAlign: 'center',
      height: '100%',
      justifyContent: 'center'
    }}>
      <h3 style={{ fontSize: '12px', textTransform: 'uppercase', color: 'var(--text-faint)', margin: 0, letterSpacing: '1px' }}>
        Overall Security Assessment
      </h3>
      
      <div style={{ fontSize: '42px', fontWeight: 700, fontFamily: 'monospace', color: '#f8fafc', lineHeight: 1 }}>
        {formatScore(risk_score)} <span style={{ fontSize: '20px', color: 'var(--text-faint)' }}>{risk_score !== null && risk_score !== undefined ? '/ 100' : ''}</span>
      </div>
      
      <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
        <RiskBadge type="verdict" value={verdict} size="large" />
        <RiskBadge type="level" value={risk_level} size="large" />
      </div>
      
      <div style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
        Confidence <strong style={{ color: '#f8fafc' }}>{formatConfidence(confidence)}</strong>
      </div>
    </div>
  );
}
