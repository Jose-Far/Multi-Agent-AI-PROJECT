import React from 'react';
import { Lightbulb, CheckCircle, BrainCircuit } from 'lucide-react';

export default function ExplainabilityPanel({ result }) {
  if (!result) return null;

  const getPrimaryEvidence = () => {
    const factors = result.explanation?.risk_factors || result.risk_factors || [];
    if (factors.length === 0) return ["No high-severity anomalous risk indicators detected for this target."];
    
    return factors.map(factor => {
      if (typeof factor === 'string') return factor;
      if (typeof factor === 'object' && factor !== null) {
        if (factor.factor) {
          if (typeof factor.factor === 'string' && factor.factor.startsWith('{')) {
            try {
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
        if (factor.feature) return `${factor.feature}: ${factor.value}`;
        return JSON.stringify(factor);
      }
      return String(factor);
    }).slice(0, 4);
  };

  const getModelInterpretation = () => {
    if (result.explanation?.summary) return result.explanation.summary;
    
    // Dynamic summary generator
    const total = result.consensus?.available_agents || 0;
    const verdict = result.final_assessment?.verdict || 'unknown';
    const score = result.final_assessment?.risk_score || 0;
    
    let suspiciousAgents = [];
    let safeAgents = [];
    
    if (result.agents) {
      Object.entries(result.agents).forEach(([key, ag]) => {
        if (!ag || ag.status === 'unavailable') return;
        const pred = (ag.prediction || '').toLowerCase();
        if (['phishing', 'suspicious', 'malicious'].includes(pred)) suspiciousAgents.push(key);
        else if (['legitimate', 'benign'].includes(pred)) safeAgents.push(key);
      });
    }
    
    let summary = `PhishDec's orchestration engine aggregated signals from ${total} active security models to reach a final calibration score of ${score}/100 (${verdict.toUpperCase()}). `;
    
    if (suspiciousAgents.length > 0) {
      summary += `The ${suspiciousAgents.map(a => a.replace('_', ' ')).join(', ')} agent(s) identified high-risk anomalies that heavily influenced the severity calibration. `;
    }
    
    if (safeAgents.length > 0 && suspiciousAgents.length > 0) {
      summary += `However, mitigating baseline signals were detected by the ${safeAgents.map(a => a.replace('_', ' ')).join(', ')} agent(s). `;
    } else if (safeAgents.length > 0) {
      summary += `The ${safeAgents.map(a => a.replace('_', ' ')).join(', ')} agent(s) unanimously observed benign patterns aligned with standard legitimate infrastructure. `;
    }
    
    summary += "The OSRH-v2 fusion algorithm mathematically calibrated these conflicting outputs to produce the final confidence projection.";
    
    return summary;
  };

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      padding: '24px',
      marginTop: '24px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '20px' }}>
        <Lightbulb size={20} color="#fbbf24" />
        <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#f8fafc', margin: 0, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Why did PhishDec reach this result?
        </h3>
      </div>
      
      <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '24px' }}>
        PhishDec combined evidence from multiple independent security agents.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
        
        {/* Observed Evidence */}
        <div style={{ backgroundColor: 'rgba(6, 9, 15, 0.4)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <CheckCircle size={16} color="var(--cyber-blue)" />
            <h4 style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc', margin: 0 }}>Observed Evidence</h4>
          </div>
          <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {getPrimaryEvidence().map((ev, i) => (
              <li key={i}>{ev}</li>
            ))}
          </ul>
        </div>

        {/* Model Interpretation */}
        <div style={{ backgroundColor: 'rgba(6, 9, 15, 0.4)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '20px', flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <BrainCircuit size={16} color="#fbbf24" />
            <h4 style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc', margin: 0 }}>Model Interpretation</h4>
          </div>
          <div style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: '1.6' }}>
            {getModelInterpretation()}
          </div>
        </div>

      </div>
    </div>
  );
}