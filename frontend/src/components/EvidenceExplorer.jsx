import React, { useState } from 'react';
import { Database, ChevronDown, ChevronRight, FileText } from 'lucide-react';

export default function EvidenceExplorer({ agents }) {
  const [expandedAgent, setExpandedAgent] = useState(null);

  if (!agents || Object.keys(agents).length === 0) return null;

  const getEvidence = (data) => {
    if (data.features && Object.keys(data.features).length > 0) return data.features;
    if (data.evidence && Object.keys(data.evidence).length > 0) return data.evidence;
    if (data.raw_features && Object.keys(data.raw_features).length > 0) return data.raw_features;
    if (data.extracted_data && Object.keys(data.extracted_data).length > 0) return data.extracted_data;
    return null;
  };

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      marginTop: '24px',
      overflow: 'hidden',
      height: '100%',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: 'rgba(6, 9, 15, 0.5)' }}>
        <Database size={18} color="var(--cyber-blue)" />
        <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#f8fafc', margin: 0, textTransform: 'uppercase' }}>
          Evidence Explorer
        </h3>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflowY: 'auto' }}>
        {Object.entries(agents).map(([key, data], i) => {
          const isExpanded = expandedAgent === key;
          const evidence = getEvidence(data);
          const hasEvidence = evidence !== null;
          
          return (
            <div key={key} style={{ borderBottom: i < Object.keys(agents).length - 1 ? '1px solid var(--border-subtle)' : 'none' }}>
              <button 
                onClick={() => setExpandedAgent(isExpanded ? null : key)}
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', gap: '10px',
                  padding: '16px 24px', backgroundColor: isExpanded ? 'rgba(56, 189, 248, 0.05)' : 'transparent',
                  border: 'none', color: '#f8fafc', cursor: 'pointer', textAlign: 'left',
                  transition: 'all 0.2s ease'
                }}
              >
                {isExpanded ? <ChevronDown size={16} color="var(--cyber-blue)" /> : <ChevronRight size={16} color="var(--text-muted)" />}
                <span style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', color: isExpanded ? 'var(--cyber-blue)' : '#f8fafc' }}>
                  {key.replace('_', ' ')} Agent
                </span>
                
                {hasEvidence && !isExpanded && (
                  <span style={{ marginLeft: '12px', fontSize: '10px', padding: '2px 6px', borderRadius: '4px', backgroundColor: 'rgba(52, 211, 153, 0.1)', color: '#34d399' }}>
                    {Object.keys(evidence).length} metrics
                  </span>
                )}

                <span style={{ marginLeft: 'auto', fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                  Risk: <span style={{ color: data.risk_score > 70 ? '#fb7185' : data.risk_score > 30 ? '#fbbf24' : '#34d399' }}>
                    {data.risk_score != null ? Number(data.risk_score).toFixed(1) : '-'}
                  </span>
                </span>
              </button>

              {isExpanded && (
                <div style={{ padding: '0 24px 24px 50px', animation: 'fadeIn 0.3s ease-out' }}>
                  {hasEvidence ? (
                    <div style={{ 
                      display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px',
                      backgroundColor: 'rgba(6, 9, 15, 0.3)', padding: '16px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)'
                    }}>
                      {Object.entries(evidence).map(([eKey, eVal], idx) => {
                        let valStr = String(eVal);
                        if (typeof eVal === 'object' && eVal !== null) {
                           valStr = JSON.stringify(eVal);
                        } else if (typeof eVal === 'number') {
                           valStr = Number.isInteger(eVal) ? eVal : Number(eVal).toFixed(3);
                        }
                        
                        return (
                          <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '4px', overflow: 'hidden' }}>
                            <span style={{ fontSize: '11px', color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.5px', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
                              {eKey.replace(/_/g, ' ')}
                            </span>
                            <span style={{ fontSize: '13px', color: '#f8fafc', fontFamily: 'monospace', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>
                              {valStr}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div style={{ padding: '16px', backgroundColor: 'rgba(6, 9, 15, 0.3)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      {data.reason && (
                        <div>
                          <span style={{ fontSize: '11px', color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Primary Reason</span>
                          <div style={{ fontSize: '13px', color: '#f8fafc', marginTop: '4px' }}>{data.reason}</div>
                        </div>
                      )}
                      {data.operational_meaning && (
                        <div>
                          <span style={{ fontSize: '11px', color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Operational Meaning</span>
                          <div style={{ fontSize: '13px', color: '#f8fafc', marginTop: '4px' }}>{data.operational_meaning}</div>
                        </div>
                      )}
                      {!data.reason && !data.operational_meaning && (
                        <div style={{ fontSize: '13px', color: 'var(--text-faint)', fontStyle: 'italic' }}>
                          No granular numerical features exposed by this agent for this specific scan.
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}