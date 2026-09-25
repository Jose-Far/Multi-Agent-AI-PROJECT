import React, { useState, useEffect } from 'react';
import { Loader2, CheckCircle2, Shield, Globe, Code, Lock, Server, Eye, ShieldAlert, Cpu } from 'lucide-react';

const STEPS = [
  { id: 'url', label: 'URL AI Agent: Extracting lexical & domain syntax vectors', icon: Globe, duration: 250 },
  { id: 'dns', label: 'DNS AI Agent: Inspecting zone records, TTL & domain age telemetry', icon: Server, duration: 400 },
  { id: 'ssl', label: 'SSL AI Agent: Validating cryptographic chain & certificate authority', icon: Lock, duration: 600 },
  { id: 'html', label: 'HTML AI Agent: Parsing DOM tree, forms & external resource anchors', icon: Code, duration: 800 },
  { id: 'visual', label: 'Visual AI Agent: Evaluating visual brand similarity & login motifs', icon: Eye, duration: 1100 },
  { id: 'threat', label: 'Threat Intel Agent: Correlating IOCs across global threat feeds', icon: ShieldAlert, duration: 1400 },
  { id: 'fusion', label: 'Decision Fusion Engine: Harmonizing risk calibration (OSRH-v2)...', icon: Cpu, duration: 1700 },
];

export default function LoadingState({ targetUrl }) {
  const [activeStep, setActiveStep] = useState(0);
  const [elapsed, setElapsed] = useState(0.0);

  useEffect(() => {
    const start = Date.now();
    const interval = setInterval(() => {
      const now = Date.now();
      const diff = ((now - start) / 1000).toFixed(1);
      setElapsed(diff);

      const msPassed = now - start;
      let nextStep = 0;
      for (let i = 0; i < STEPS.length; i++) {
        if (msPassed >= STEPS[i].duration) {
          nextStep = i;
        }
      }
      setActiveStep(nextStep);
    }, 100);

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      padding: '36px 32px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '24px',
      maxWidth: '650px',
      margin: '40px auto',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
    }}>
      {/* Spinner & Target */}
      <div style={{ textAlign: 'center' }}>
        <div style={{
          width: '56px',
          height: '56px',
          borderRadius: '50%',
          backgroundColor: 'rgba(56, 189, 248, 0.1)',
          border: '1px solid rgba(56, 189, 248, 0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 16px'
        }}>
          <Loader2 size={28} color="var(--cyber-blue)" style={{ animation: 'spin 1.5s linear infinite' }} />
        </div>
        <h3 style={{ fontSize: '18px', fontWeight: 600, color: '#f8fafc', marginBottom: '6px' }}>
          Multi-Agent Pipeline Executing
        </h3>
        <div style={{
          fontSize: '12px',
          color: 'var(--text-muted)',
          fontFamily: 'monospace',
          backgroundColor: 'rgba(6, 9, 15, 0.6)',
          padding: '6px 14px',
          borderRadius: '4px',
          border: '1px solid var(--border-subtle)',
          maxWidth: '500px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        }}>
          Target: {targetUrl}
        </div>
      </div>

      {/* Checklist of 6 Agents */}
      <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {STEPS.map((step, idx) => {
          const Icon = step.icon;
          const isDone = idx < activeStep;
          const isCurrent = idx === activeStep;

          return (
            <div
              key={step.id}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '10px 14px',
                borderRadius: '6px',
                backgroundColor: isCurrent ? 'rgba(56, 189, 248, 0.08)' : 'rgba(6, 9, 15, 0.3)',
                border: `1px solid ${isCurrent ? 'rgba(56, 189, 248, 0.3)' : 'var(--border-subtle)'}`,
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ width: '20px', display: 'flex', justifyContent: 'center' }}>
                {isDone ? (
                  <CheckCircle2 size={16} color="#34d399" />
                ) : isCurrent ? (
                  <Loader2 size={16} color="var(--cyber-blue)" style={{ animation: 'spin 1s linear infinite' }} />
                ) : (
                  <Icon size={16} color="var(--text-faint)" />
                )}
              </div>
              <span style={{
                fontSize: '12px',
                color: isDone ? 'var(--text-muted)' : isCurrent ? '#f8fafc' : 'var(--text-faint)',
                fontWeight: isCurrent ? 600 : 400,
                flex: 1
              }}>
                {step.label}
              </span>
              {isDone && (
                <span style={{ fontSize: '10px', color: '#34d399', fontWeight: 600 }}>READY</span>
              )}
              {isCurrent && (
                <span style={{ fontSize: '10px', color: 'var(--cyber-blue)', fontWeight: 600 }}>ACTIVE</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Elapsed Counter */}
      <div style={{ fontSize: '11px', color: 'var(--text-faint)', fontFamily: 'monospace' }}>
        Elapsed: {elapsed}s | 6 AI Agents Active
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
