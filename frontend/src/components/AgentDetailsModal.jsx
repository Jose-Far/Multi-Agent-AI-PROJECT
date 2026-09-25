import React from 'react';
import { X, Shield, Globe, Database, FileText, Cpu, AlertTriangle, ShieldAlert, FileCode } from 'lucide-react';
import StatusBadge from './StatusBadge';

export default function AgentDetailsModal({ isOpen, agentKey, agentData, onClose }) {
  if (!isOpen || !agentKey || !agentData) return null;

  const ICONS = {
    url: Globe,
    html: FileCode,
    ssl: Shield,
    dns: Database,
    visual: FileText,
    threat_intelligence: ShieldAlert
  };

  const Icon = ICONS[agentKey] || Cpu;
  const nameMap = {
    url: 'URL Analysis',
    html: 'HTML DOM Analysis',
    ssl: 'SSL/TLS Analysis',
    dns: 'DNS Intelligence',
    visual: 'Visual/Brand Analysis',
    threat_intelligence: 'Threat Intelligence'
  };

  const getEvidence = (data) => {
    if (data.features && Object.keys(data.features).length > 0) return data.features;
    if (data.evidence && Object.keys(data.evidence).length > 0) return data.evidence;
    if (data.raw_features && Object.keys(data.raw_features).length > 0) return data.raw_features;
    if (data.extracted_data && Object.keys(data.extracted_data).length > 0) return data.extracted_data;
    return null;
  };

  const renderDynamicEvidence = () => {
    const evidence = getEvidence(agentData);
    if (!evidence || Object.keys(evidence).length === 0) {
      if (agentData.reason || agentData.operational_meaning) {
        return (
          <div style={{ backgroundColor: 'rgba(6, 9, 15, 0.3)', padding: '16px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', flexDirection: 'column', gap: '12px' }}>
             {agentData.reason && (
               <div>
                 <span style={{ fontSize: '11px', color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Primary Reason</span>
                 <div style={{ fontSize: '13px', color: '#f8fafc', marginTop: '4px' }}>{agentData.reason}</div>
               </div>
             )}
             {agentData.operational_meaning && (
               <div>
                 <span style={{ fontSize: '11px', color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Operational Meaning</span>
                 <div style={{ fontSize: '13px', color: '#f8fafc', marginTop: '4px' }}>{agentData.operational_meaning}</div>
               </div>
             )}
          </div>
        );
      }
      return <div style={{ color: 'var(--text-muted)' }}>No explicit evidence available in the database record.</div>;
    }

    return (
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
    );
  };

  const renderRiskFactor = (rf) => {
    if (typeof rf === 'string') return rf;
    if (typeof rf === 'object' && rf !== null) {
      if (rf.factor) {
        if (typeof rf.factor === 'string' && rf.factor.startsWith('{')) {
          try {
            const innerStr = rf.factor.replace(/'/g, '"');
            const innerObj = JSON.parse(innerStr);
            if (innerObj.feature) {
               return `${innerObj.feature.toUpperCase().replace(/_/g, ' ')}: ${innerObj.value} (Impact: ${Number(innerObj.impact).toFixed(2)})`;
            }
          } catch(e) {
             return rf.factor;
          }
        }
        return rf.factor;
      }
      return rf.title || rf.feature || JSON.stringify(rf);
    }
    return String(rf);
  };

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
    }}>
      <div style={{
        backgroundColor: 'var(--bg-main)', border: '1px solid var(--border-subtle)',
        borderRadius: '8px', width: '90%', maxWidth: '600px', maxHeight: '90vh',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.2)'
      }}>
        {/* Header */}
        <div style={{ padding: '20px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Icon size={20} color="var(--cyber-blue)" />
            <h3 style={{ margin: 0, fontSize: '16px', color: '#fff', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              {nameMap[agentKey] || agentKey}
            </h3>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Top Stats */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', fontSize: '13px', backgroundColor: 'rgba(6, 9, 15, 0.5)', padding: '16px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
            <div><span style={{ color: 'var(--text-muted)' }}>Status:</span> <StatusBadge status={agentData.status} /></div>
            <div><span style={{ color: 'var(--text-muted)' }}>Prediction:</span> <strong style={{ color: '#fff', textTransform: 'uppercase' }}>{agentData.prediction || 'Unknown'}</strong></div>
            <div><span style={{ color: 'var(--text-muted)' }}>Risk Score:</span> <strong style={{ color: '#fff' }}>{agentData.risk_score != null ? Number(agentData.risk_score).toFixed(1) + ' / 100' : 'N/A'}</strong></div>
            <div><span style={{ color: 'var(--text-muted)' }}>Confidence:</span> <strong style={{ color: '#fff' }}>{agentData.confidence != null ? (Number(agentData.confidence) <= 1.0 ? (Number(agentData.confidence) * 100).toFixed(1) + '%' : Number(agentData.confidence).toFixed(1) + '%') : 'N/A'}</strong></div>
            <div><span style={{ color: 'var(--text-muted)' }}>Source:</span> <strong style={{ color: '#fff' }}>{agentData.prediction_source || 'ML Model'}</strong></div>
          </div>

          {/* Risk Factors */}
          {agentData.risk_factors && agentData.risk_factors.length > 0 && (
            <div>
              <h4 style={{ fontSize: '12px', color: 'var(--text-faint)', textTransform: 'uppercase', marginBottom: '12px' }}>Risk Factors</h4>
              <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', color: '#fbbf24', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {agentData.risk_factors.map((rf, i) => (
                  <li key={i}>{renderRiskFactor(rf)}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Extracted Evidence */}
          <div>
            <h4 style={{ fontSize: '12px', color: 'var(--text-faint)', textTransform: 'uppercase', marginBottom: '12px' }}>Observed Evidence</h4>
            <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              {renderDynamicEvidence()}
            </div>
          </div>
          
          {/* Explanation */}
          {agentData.explanation && (
            <div style={{ backgroundColor: 'rgba(56, 189, 248, 0.05)', padding: '16px', borderRadius: '6px', border: '1px solid rgba(56, 189, 248, 0.1)' }}>
              <h4 style={{ fontSize: '12px', color: 'var(--cyber-blue)', textTransform: 'uppercase', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Cpu size={14} /> Model Interpretation
              </h4>
              <div style={{ fontSize: '13px', color: '#e2e8f0', lineHeight: 1.5 }}>
                {agentData.explanation}
              </div>
            </div>
          )}
          
        </div>
      </div>
    </div>
  );
}