import React from 'react';
import { formatIST } from '../utils/dateUtils';
import { Clock, CheckCircle2, Circle } from 'lucide-react';

export default function InvestigationTimeline({ timestamp }) {
  if (!timestamp) return null;

  const dateObj = new Date(timestamp);
  
  const formatTime = (offsetMs) => {
    const d = new Date(dateObj.getTime() - offsetMs);
    return formatIST(d).split(' ')[1];
  };

  const timelineSteps = [
    { label: 'Ingestion Triggered', offset: 1200, isFinal: false, active: true },
    { label: 'URL Lexical Scan', offset: 900, isFinal: false, active: true },
    { label: 'DOM/HTML Sandbox', offset: 750, isFinal: false, active: true },
    { label: 'SSL Cert Validation', offset: 600, isFinal: false, active: true },
    { label: 'DNS Graph Resolution', offset: 450, isFinal: false, active: true },
    { label: 'Visual OCR Render', offset: 300, isFinal: false, active: true },
    { label: 'Global Threat Intel Query', offset: 150, isFinal: false, active: true },
    { label: 'OSRH-v2 Consensus Fusion', offset: 50, isFinal: false, active: true },
    { label: 'Final Matrix Compiled', offset: 0, isFinal: true, active: true },
  ];

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      padding: '24px',
      marginTop: '24px',
      height: '100%',
      position: 'relative',
      overflow: 'hidden'
    }}>
      <style>
        {`
          @keyframes lineGlow {
            0% { background-position: 0% 0%; }
            50% { background-position: 0% 100%; }
            100% { background-position: 0% 0%; }
          }
          @keyframes pulseDot {
            0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.4); }
            70% { box-shadow: 0 0 0 6px rgba(56, 189, 248, 0); }
            100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
          }
        `}
      </style>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '28px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '16px' }}>
        <Clock size={18} color="var(--cyber-blue)" />
        <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#f8fafc', margin: 0, textTransform: 'uppercase' }}>
          Processing Sequence
        </h3>
      </div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', position: 'relative' }}>
        
        {/* Animated Background Line */}
        <div style={{
          position: 'absolute',
          top: '10px',
          bottom: '20px',
          left: '87px',
          width: '2px',
          background: 'linear-gradient(180deg, rgba(56,189,248,0.1) 0%, rgba(56,189,248,0.8) 50%, rgba(56,189,248,0.1) 100%)',
          backgroundSize: '100% 200%',
          animation: 'lineGlow 3s ease-in-out infinite',
          zIndex: 1
        }} />

        {timelineSteps.map((step, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', zIndex: 2 }}>
            <div style={{ width: '65px', fontSize: '11px', color: 'var(--cyber-blue)', fontFamily: 'monospace', paddingTop: '3px', textAlign: 'right', opacity: 0.8 }}>
              {formatTime(step.offset)}
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', paddingTop: '2px' }}>
              {step.isFinal ? (
                <div style={{ 
                  width: '14px', height: '14px', borderRadius: '50%', 
                  backgroundColor: 'var(--cyber-blue)', border: '2px solid var(--bg-card)',
                  animation: 'pulseDot 2s infinite'
                }} />
              ) : (
                <div style={{ 
                  width: '10px', height: '10px', borderRadius: '50%', 
                  backgroundColor: 'var(--bg-card)', border: '2px solid #34d399',
                  boxShadow: '0 0 6px rgba(52, 211, 153, 0.4)'
                }} />
              )}
            </div>
            
            <div style={{ 
              fontSize: '13px', 
              color: step.isFinal ? '#fff' : 'var(--text-muted)', 
              fontWeight: step.isFinal ? 600 : 400,
              textShadow: step.isFinal ? '0 0 8px rgba(255,255,255,0.3)' : 'none',
              transform: step.isFinal ? 'scale(1.02)' : 'none',
              transformOrigin: 'left center',
              transition: 'all 0.3s ease'
            }}>
              {step.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
