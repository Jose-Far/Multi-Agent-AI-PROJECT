import React from 'react';
import { AlertTriangle, GitMerge } from 'lucide-react';

export default function FusionSummary({ consensus, finalAssessment, agents }) {
  if (!consensus) return null;
  
  const { 
    available_agents, total_agents, 
    consensus_satisfied, evidence_state, 
    conflict_detected, limited_evidence 
  } = consensus;
  
  const { risk_score, confidence } = finalAssessment || {};
  
  const formatScore = (score) => score !== null && score !== undefined ? Number(score).toFixed(1) : '-';
  const formatConfidence = (conf) => conf !== null && conf !== undefined ? `%` : '0%';

  // Calculate Agreement
  let suspiciousCount = 0;
  let legitimateCount = 0;
  let unknownCount = 0;
  
  if (agents) {
    Object.values(agents).forEach(ag => {
      const pred = (ag.prediction || '').toLowerCase();
      if (['phishing', 'suspicious', 'malicious'].includes(pred)) suspiciousCount++;
      else if (['legitimate', 'benign'].includes(pred)) legitimateCount++;
      else unknownCount++;
    });
  }

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      padding: '24px',
      display: 'flex',
      flexDirection: 'column',
      gap: '20px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <GitMerge size={20} color="var(--cyber-blue)" />
        <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#f8fafc', margin: 0, textTransform: 'uppercase' }}>
          Decision Fusion
        </h3>
      </div>
      
      {/* Step 24: Visual bars for agent contributions */}
      {agents && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '8px' }}>
          {Object.entries(agents).map(([key, data]) => {
            const score = data.risk_score || 0;
            const isMissing = data.status === 'unavailable' || data.status === 'error';
            return (
              <div key={key === 'threat_intelligence' ? 'Threat' : key} style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12px' }}>
                <div style={{ width: '80px', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                  {key === 'threat_intelligence' ? 'Threat' : key}
                </div>
                <div style={{ flex: 1, backgroundColor: 'rgba(6, 9, 15, 0.5)', height: '12px', borderRadius: '2px', overflow: 'hidden', position: 'relative' }}>
                  {!isMissing && (
                    <div style={{
                      position: 'absolute', top: 0, bottom: 0, left: 0, width: `${Math.round(score)}%`,
                      backgroundColor: score > 79 ? '#f43f5e' : score > 59 ? '#fb7185' : score > 29 ? '#fbbf24' : '#34d399'
                    }} />
                  )}
                </div>
                <div style={{ width: '30px', textAlign: 'right', color: isMissing ? 'var(--text-faint)' : '#f8fafc', fontFamily: 'monospace' }}>
                  {isMissing ? '—' : Math.round(score)}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Consensus Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '13px', borderTop: '1px solid var(--border-subtle)', paddingTop: '16px' }}>
        <div style={{ color: 'var(--text-muted)' }}>Agents Usable</div>
        <div style={{ color: '#f8fafc', fontWeight: 600 }}>{available_agents || 0} / {total_agents || 6}</div>
        
        <div style={{ color: 'var(--text-muted)' }}>Consensus Satisfied</div>
        <div style={{ color: '#f8fafc', fontWeight: 600 }}>{consensus_satisfied ? 'Yes' : 'No'}</div>
        
        <div style={{ color: 'var(--text-muted)' }}>Agent Agreement</div>
        <div style={{ color: '#f8fafc', fontWeight: 600 }}>
          <span style={{ color: '#fb7185' }}>{suspiciousCount} Suspicious</span> / <span style={{ color: '#34d399' }}>{legitimateCount} Legitimate</span>
        </div>
      </div>

      {limited_evidence && (
        <div style={{ color: '#fbbf24', fontSize: '12px', padding: '10px', backgroundColor: 'rgba(251, 191, 36, 0.05)', borderRadius: '4px', border: '1px solid rgba(251, 191, 36, 0.2)' }}>
          <strong>⚠ LIMITED EVIDENCE:</strong> Only a subset of the six agents produced usable results. The assessment should be interpreted with caution.
        </div>
      )}

      {conflict_detected && (
        <div style={{
          padding: '16px',
          backgroundColor: 'rgba(251, 191, 36, 0.05)',
          border: '1px solid rgba(251, 191, 36, 0.2)',
          borderRadius: '6px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', color: '#fbbf24', fontWeight: 600, fontSize: '13px' }}>
            <AlertTriangle size={16} />
            ⚠ CONFLICTING EVIDENCE
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            Evidence from the agents is not fully consistent. Different security signals disagree on the classification of this target.
          </div>
        </div>
      )}
    </div>
  );
}
