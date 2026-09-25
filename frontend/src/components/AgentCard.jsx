import React from 'react';
import { 
  Globe, 
  Code, 
  Lock, 
  Server, 
  Eye, 
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  TrendingUp,
  Cpu
} from 'lucide-react';
import RiskBadge from './RiskBadge';
import StatusBadge from './StatusBadge';

const AGENT_CONFIGS = {
  url: {
    label: 'URL AI Agent',
    icon: Globe,
    tech: 'Lexical & Structural ML (XGBoost)',
    color: '#38bdf8'
  },
  html: {
    label: 'HTML AI Agent',
    icon: Code,
    tech: 'DOM Tree & Form Heuristics',
    color: '#a78bfa'
  },
  ssl: {
    label: 'SSL AI Agent',
    icon: Lock,
    tech: 'X.509 Cryptographic Validation',
    color: '#34d399'
  },
  dns: {
    label: 'DNS AI Agent',
    icon: Server,
    tech: 'Zone Telemetry & Infrastructure',
    color: '#f59e0b'
  },
  visual: {
    label: 'Visual AI Agent',
    icon: Eye,
    tech: 'Brand & Layout Perception',
    color: '#ec4899'
  },
  threat_intelligence: {
    label: 'Threat Intelligence Agent',
    icon: ShieldAlert,
    tech: 'Global Threat Feeds & IOCs',
    color: '#f43f5e'
  }
};

export default function AgentCard({ agentKey, data, onViewDetails }) {
  const config = AGENT_CONFIGS[agentKey] || {
    label: agentKey,
    icon: Cpu,
    tech: 'AI Analysis Subsystem',
    color: '#94a3b8'
  };

  const Icon = config.icon;
  const isAvailable = data && data.status === 'success' && data.signal !== false;
  const riskScore = data?.risk_score !== null && data?.risk_score !== undefined 
    ? Number(data.risk_score).toFixed(1) 
    : 'N/A';
  
  const rawScore = data?.raw_risk_score !== null && data?.raw_risk_score !== undefined
    ? Number(data.raw_risk_score).toFixed(1)
    : null;

  const confidence = data?.confidence !== null && data?.confidence !== undefined
    ? (Number(data.confidence) <= 1.0 ? (Number(data.confidence) * 100).toFixed(1) + '%' : Number(data.confidence).toFixed(1) + '%')
    : 'N/A';

  // Calculate score bar width
  const scoreNum = Number(riskScore);
  const barWidth = !isNaN(scoreNum) ? Math.min(Math.max(scoreNum, 0), 100) : 0;
  
  // Color bar based on score
  let barColor = '#34d399';
  if (scoreNum >= 75) barColor = '#f43f5e';
  else if (scoreNum >= 60) barColor = '#fb7185';
  else if (scoreNum >= 40) barColor = '#fbbf24';

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      padding: '18px 20px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      gap: '14px',
      position: 'relative',
      overflow: 'hidden',
      transition: 'border-color 0.2s ease',
      boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
    }}>
      {/* Top Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '34px',
            height: '34px',
            borderRadius: '6px',
            backgroundColor: 'rgba(255, 255, 255, 0.04)',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Icon size={18} color={config.color} />
          </div>
          <div>
            <div style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc' }}>
              {config.label}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-faint)' }}>
              {config.tech}
            </div>
          </div>
        </div>

        {/* Status Pill */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '4px' }}>
          <span style={{
            fontSize: '10px',
            fontWeight: 700,
            padding: '2px 7px',
            borderRadius: '4px',
            backgroundColor: isAvailable ? 'rgba(16, 185, 129, 0.15)' : 'rgba(148, 163, 184, 0.15)',
            color: isAvailable ? '#34d399' : '#94a3b8',
            border: `1px solid ${isAvailable ? 'rgba(16, 185, 129, 0.3)' : 'rgba(148, 163, 184, 0.3)'}`
          }}>
            {data?.status?.toUpperCase() || 'UNAVAILABLE'}
          </span>
          {data?.signal === false && (
            <span style={{ fontSize: '9px', color: '#fbbf24', fontWeight: 600 }}>
              No Direct Signal
            </span>
          )}
        </div>
      </div>

      {/* Main Metric: Operational Risk Score & Prediction */}
      <div style={{
        backgroundColor: 'rgba(6, 9, 15, 0.5)',
        borderRadius: '6px',
        padding: '12px 14px',
        border: '1px solid var(--border-subtle)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div>
            <div style={{ fontSize: '10px', textTransform: 'uppercase', color: 'var(--text-faint)', letterSpacing: '0.5px' }}>
              Calibrated Risk Score
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', marginTop: '2px' }}>
              <span style={{
                fontSize: '22px',
                fontWeight: 700,
                fontFamily: 'ui-monospace, monospace',
                color: riskScore !== 'N/A' ? '#ffffff' : 'var(--text-faint)'
              }}>
                {riskScore}
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-faint)' }}>/ 100</span>
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '10px', textTransform: 'uppercase', color: 'var(--text-faint)', letterSpacing: '0.5px', marginBottom: '4px' }}>
              Prediction
            </div>
            <RiskBadge type="verdict" value={data?.prediction || 'unknown'} size="small" />
          </div>
        </div>

        {/* Risk Progress Bar */}
        <div style={{
          height: '4px',
          borderRadius: '2px',
          backgroundColor: 'rgba(255, 255, 255, 0.06)',
          overflow: 'hidden',
          marginBottom: '8px'
        }}>
          <div style={{
            height: '100%',
            width: `${barWidth}%`,
            backgroundColor: barColor,
            transition: 'width 0.5s ease-out'
          }} />
        </div>

        {/* Secondary Metrics Row */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '11px',
          color: 'var(--text-muted)',
          paddingTop: '4px'
        }}>
          <span>Confidence: <strong style={{ color: '#f8fafc', fontFamily: 'monospace' }}>{confidence}</strong></span>
          {rawScore !== null && (
            <span>Raw: <strong style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>{rawScore}</strong></span>
          )}
        </div>
      </div>

      {/* Decision Factor / Reason */}
      <div style={{
        fontSize: '12px',
        color: 'var(--text-muted)',
        lineHeight: 1.4,
        minHeight: '34px'
      }}>
        {data?.reason || data?.operational_meaning || 'Standard baseline telemetry acquired without anomalous triggers.'}
      </div>

      
      {/* Prediction Source & View Details */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '11px',
        color: 'var(--text-faint)',
        marginTop: 'auto',
        borderTop: '1px solid var(--border-subtle)',
        paddingTop: '10px'
      }}>
        <span>Source: <strong style={{ color: '#f8fafc' }}>{data?.prediction_source || 'ML Model'}</strong></span>
        <button onClick={onViewDetails} style={{
          background: 'none',
          border: 'none',
          color: 'var(--cyber-blue)',
          fontSize: '11px',
          cursor: 'pointer',
          padding: 0
        }}>
          [ View Details ]
        </button>
      </div>

      {/* Calibration Metadata Footer */}
      {data?.calibration_curve && (
        <div style={{
          borderTop: '1px solid var(--border-subtle)',
          paddingTop: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '10px',
          color: 'var(--text-faint)'
        }}>
          <span>Curve: <span style={{ fontFamily: 'monospace', color: 'var(--cyber-blue)' }}>{data.calibration_curve}</span></span>
          <span style={{ textTransform: 'uppercase', fontWeight: 600 }}>OSRH-v2</span>
        </div>
      )}
    </div>
  );
}
