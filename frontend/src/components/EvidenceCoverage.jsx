import React from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function EvidenceCoverage({ consensus, analysisId }) {
  if (!consensus) return null;
  
  const available = consensus.available_agents || 0;
  const total = consensus.total_agents || 6;
  const percentage = total > 0 ? (available / total) * 100 : 0;
  
  const isLimited = consensus.limited_evidence || available < total;

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      padding: '24px',
      display: 'flex',
      flexDirection: 'column',
      gap: '16px',
      height: '100%',
      justifyContent: 'center'
    }}>
      <h3 style={{ fontSize: '12px', textTransform: 'uppercase', color: 'var(--text-muted)', margin: 0, letterSpacing: '1px' }}>
        Evidence Coverage
      </h3>
      
      <div style={{ fontSize: '18px', color: '#f8fafc', fontWeight: 600 }}>
        {available} / {total} Agents Available
      </div>
      
      {/* Visual Progress Bar instead of Xs */}
      <div style={{ width: '100%', height: '8px', backgroundColor: 'var(--bg-dark)', borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{ 
          width: `${percentage}%`, 
          height: '100%', 
          backgroundColor: isLimited ? '#fbbf24' : '#34d399',
          transition: 'width 0.5s ease-out'
        }}></div>
      </div>
      
      {/* Respective Investigation ID */}
      {analysisId && (
        <div style={{ fontSize: '13px', color: 'var(--text-faint)', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '6px' }}>
          ID: <span style={{ color: 'var(--text-muted)' }}>{analysisId}</span>
        </div>
      )}
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {isLimited ? (
          <>
            <AlertTriangle size={18} color="#fbbf24" />
            <span style={{ fontSize: '15px', color: '#fbbf24', fontWeight: 600 }}>Limited Evidence</span>
          </>
        ) : (
          <>
            <CheckCircle2 size={18} color="#34d399" />
            <span style={{ fontSize: '15px', color: '#34d399', fontWeight: 600 }}>Complete Evidence</span>
          </>
        )}
      </div>
    </div>
  );
}