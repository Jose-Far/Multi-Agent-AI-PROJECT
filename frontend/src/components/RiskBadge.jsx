import React from 'react';
import { AlertTriangle, CheckCircle, HelpCircle, ShieldAlert, Shield } from 'lucide-react';

export default function RiskBadge({ type = 'verdict', value, size = 'medium' }) {
  if (!value) return null;

  const normalized = String(value).toLowerCase().trim();

  let color = '#94a3b8';
  let bg = 'rgba(148, 163, 184, 0.12)';
  let border = 'rgba(148, 163, 184, 0.25)';
  let Icon = HelpCircle;
  let label = String(value).toUpperCase();

  if (type === 'verdict') {
    if (normalized === 'phishing') {
      color = '#f43f5e';
      bg = 'rgba(244, 63, 94, 0.15)';
      border = 'rgba(244, 63, 94, 0.4)';
      Icon = ShieldAlert;
      label = 'PHISHING';
    } else if (normalized === 'suspicious') {
      color = '#fbbf24';
      bg = 'rgba(251, 191, 36, 0.15)';
      border = 'rgba(251, 191, 36, 0.4)';
      Icon = AlertTriangle;
      label = 'SUSPICIOUS';
    } else if (normalized === 'legitimate') {
      color = '#34d399';
      bg = 'rgba(52, 211, 153, 0.15)';
      border = 'rgba(52, 211, 153, 0.4)';
      Icon = CheckCircle;
      label = 'LEGITIMATE';
    } else {
      color = '#94a3b8';
      bg = 'rgba(148, 163, 184, 0.15)';
      border = 'rgba(148, 163, 184, 0.3)';
      Icon = HelpCircle;
      label = 'UNKNOWN';
    }
  } else {
    // Risk Level
    if (normalized === 'critical') {
      color = '#f43f5e';
      bg = 'rgba(244, 63, 94, 0.15)';
      border = 'rgba(244, 63, 94, 0.4)';
      Icon = ShieldAlert;
      label = 'CRITICAL RISK';
    } else if (normalized === 'high') {
      color = '#fb7185';
      bg = 'rgba(251, 113, 133, 0.15)';
      border = 'rgba(251, 113, 133, 0.4)';
      Icon = AlertTriangle;
      label = 'HIGH RISK';
    } else if (normalized === 'medium') {
      color = '#fbbf24';
      bg = 'rgba(251, 191, 36, 0.15)';
      border = 'rgba(251, 191, 36, 0.4)';
      Icon = AlertTriangle;
      label = 'MEDIUM RISK';
    } else if (normalized === 'low') {
      color = '#34d399';
      bg = 'rgba(52, 211, 153, 0.15)';
      border = 'rgba(52, 211, 153, 0.4)';
      Icon = Shield;
      label = 'LOW RISK';
    } else {
      color = '#94a3b8';
      bg = 'rgba(148, 163, 184, 0.15)';
      border = 'rgba(148, 163, 184, 0.3)';
      Icon = HelpCircle;
      label = 'UNKNOWN RISK';
    }
  }

  const sizes = {
    small: { padding: '2px 8px', fontSize: '10px', iconSize: 11, gap: '4px' },
    medium: { padding: '4px 10px', fontSize: '11px', iconSize: 13, gap: '6px' },
    large: { padding: '6px 14px', fontSize: '13px', iconSize: 16, gap: '8px' },
  };

  const s = sizes[size] || sizes.medium;

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: s.gap,
      padding: s.padding,
      borderRadius: '4px',
      backgroundColor: bg,
      border: `1px solid ${border}`,
      color: color,
      fontSize: s.fontSize,
      fontWeight: 700,
      letterSpacing: '0.5px',
      whiteSpace: 'nowrap'
    }}>
      <Icon size={s.iconSize} color={color} />
      <span>{label}</span>
    </span>
  );
}
