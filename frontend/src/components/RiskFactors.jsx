import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function RiskFactors({ riskFactors }) {
  if (!riskFactors || riskFactors.length === 0) {
    return (
      <div style={{
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '8px',
        padding: '24px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <AlertTriangle size={16} color="#fbbf24" />
          <h4 style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc', margin: 0 }}>Identified Risk Indicators</h4>
        </div>
        <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
          No high-severity anomalous risk indicators detected for this target.
        </div>
      </div>
    );
  }

  const formatFactor = (factor) => {
    if (typeof factor === 'string') return factor;
    if (typeof factor === 'object' && factor !== null) {
      if (factor.factor) {
        // Handle inner JSON string if necessary
        if (typeof factor.factor === 'string' && factor.factor.startsWith('{')) {
          try {
            // It might use single quotes instead of double quotes, try to handle it roughly
            const innerStr = factor.factor.replace(/'/g, '"');
            const innerObj = JSON.parse(innerStr);
            if (innerObj.feature) {
               return `${innerObj.feature.toUpperCase().replace(/_/g, ' ')}: ${innerObj.value} (Impact: ${Number(innerObj.impact).toFixed(2)})`;
            }
          } catch(e) {
             return factor.factor;
          }
        }
        return factor.factor;
      }
      if (factor.title) return factor.title;
      if (factor.feature) {
        return `${factor.feature}: ${factor.value} (${factor.impact})`;
      }
      // If we don't know what it is, just stringify it neatly
      return JSON.stringify(factor);
    }
    return String(factor);
  };

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      padding: '24px',
      display: 'flex',
      flexDirection: 'column',
      gap: '12px',
      height: '100%'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <AlertTriangle size={16} color="#fbbf24" />
        <h4 style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
          Identified Risk Indicators ({riskFactors.length})
        </h4>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1, overflowY: 'auto', paddingRight: '4px' }}>
        {riskFactors.map((rf, idx) => {
          let severity = rf.severity;
          // Extract severity if hidden in string
          if (!severity && typeof rf.factor === 'string' && rf.factor.includes("'impact':")) {
             severity = 'high';
          }
          
          const isHigh = severity === 'high' || severity === 'critical' || severity === 'High';
          
          return (
            <div
              key={idx}
              style={{
                padding: '10px 12px',
                borderRadius: '4px',
                backgroundColor: isHigh ? 'rgba(244, 63, 94, 0.05)' : 'rgba(6, 9, 15, 0.4)',
                border: `1px solid ${isHigh ? 'rgba(244, 63, 94, 0.2)' : 'var(--border-subtle)'}`,
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '10px'
              }}
            >
              <span style={{ color: isHigh ? '#fb7185' : '#f8fafc', lineHeight: 1.4 }}>
                {formatFactor(rf)}
              </span>
              {severity && (
                <span style={{
                  fontSize: '10px',
                  fontWeight: 700,
                  padding: '2px 6px',
                  borderRadius: '3px',
                  backgroundColor: isHigh ? 'rgba(244, 63, 94, 0.15)' : 'rgba(251, 191, 36, 0.15)',
                  color: isHigh ? '#fb7185' : '#fbbf24',
                  textTransform: 'uppercase'
                }}>
                  {severity}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}