import React, { useState } from 'react';
import { Database, ChevronDown, ChevronRight } from 'lucide-react';

export default function EvidencePanel({ evidence }) {
  const [expanded, setExpanded] = useState(false);

  if (!evidence || evidence.length === 0) return null;

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      overflow: 'hidden',
      marginTop: '24px'
    }}>
      <button 
        onClick={() => setExpanded(!expanded)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '16px 24px',
          backgroundColor: 'transparent',
          border: 'none',
          color: '#f8fafc',
          cursor: 'pointer',
          textAlign: 'left'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Database size={16} color="var(--cyber-blue)" />
          <h4 style={{ fontSize: '14px', fontWeight: 600, margin: 0 }}>
            Raw Evidence Chain ({evidence.length} blocks)
          </h4>
        </div>
        {expanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
      </button>

      {expanded && (
        <div style={{
          padding: '0 24px 24px 24px',
          borderTop: '1px solid var(--border-subtle)',
        }}>
          <div style={{
            backgroundColor: 'rgba(6, 9, 15, 0.7)',
            padding: '16px',
            borderRadius: '6px',
            border: '1px solid var(--border-subtle)',
            maxHeight: '300px',
            overflowY: 'auto',
            marginTop: '16px',
            fontSize: '12px',
            fontFamily: 'ui-monospace, monospace',
            color: 'var(--text-muted)'
          }}>
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(evidence, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
