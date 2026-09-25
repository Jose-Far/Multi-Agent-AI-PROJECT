import React from 'react';
import { Server, Database, GitMerge, Settings2, CheckCircle2, XCircle, AlertTriangle, Activity, Zap, HardDrive, ShieldCheck } from 'lucide-react';

export default function AdminHealthPage({ healthData, apiOnline }) {
  if (!healthData && !apiOnline) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh', color: 'var(--text-faint)', flexDirection: 'column', gap: '15px' }}>
        <XCircle size={48} color="#f43f5e" />
        <h3>System Health data unavailable (API offline).</h3>
      </div>
    );
  }

  const { api, database, fusion, system } = healthData || {};

  const cardStyle = {
    backgroundColor: 'var(--bg-card)',
    border: '1px solid var(--border-subtle)',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    transition: 'all 0.3s ease',
  };

  const StatusRow = ({ label, isOk, subtext }) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
      <div>
        <div style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '15px' }}>{label}</div>
        {subtext && <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>{subtext}</div>}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: isOk ? '#10b981' : '#f43f5e', fontWeight: 600, fontSize: '14px', backgroundColor: isOk ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)', padding: '6px 12px', borderRadius: '20px' }}>
        {isOk ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
        {isOk ? 'Online' : 'Offline'}
      </div>
    </div>
  );

  return (
    <div style={{ paddingBottom: '40px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '30px' }}>
        <Activity size={28} style={{ color: 'var(--cyber-blue)' }} />
        <h2 style={{ margin: 0, color: '#fff', fontSize: '24px' }}>System Health Checks</h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '30px', marginBottom: '30px' }}>
        
        {/* API Health */}
        <div style={{ ...cardStyle, borderTop: `4px solid ${api?.status === 'online' ? '#38bdf8' : '#ef4444'}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Server size={24} style={{ color: '#38bdf8' }} />
            <h2 style={{ margin: 0, fontSize: '20px', color: '#fff' }}>API Health Gateway</h2>
          </div>
          
          <StatusRow label="Backend Service" isOk={api?.status === 'online'} subtext="Main Flask REST API (:5050)" />
          <StatusRow label="Analysis Endpoints" isOk={api?.status === 'online'} subtext="POST /analyze" />
          <StatusRow label="History & Reports" isOk={api?.status === 'online'} subtext="GET /analyses, GET /report" />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 0' }}>
            <div>
              <div style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '15px' }}>Health Checks</div>
              <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>GET /health, GET /admin/system</div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: api?.status === 'online' ? '#10b981' : '#f43f5e', fontWeight: 600, fontSize: '14px', backgroundColor: api?.status === 'online' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)', padding: '6px 12px', borderRadius: '20px' }}>
              {api?.status === 'online' ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
              {api?.status === 'online' ? 'Online' : 'Offline'}
            </div>
          </div>
        </div>

        {/* Database Health */}
        <div style={{ ...cardStyle, borderTop: `4px solid ${database?.status === 'connected' ? '#a78bfa' : '#ef4444'}`, display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Database size={24} style={{ color: '#a78bfa' }} />
            <h2 style={{ margin: 0, fontSize: '20px', color: '#fff' }}>Database Status</h2>
          </div>
          
          <StatusRow label="Connection Status" isOk={database?.status === 'connected'} subtext="SQLite Persistent Storage" />
          
          <div style={{ marginTop: 'auto', paddingTop: '20px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div style={{ padding: '20px', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '14px', marginBottom: '10px', fontWeight: 600 }}>
                Total Investigations <HardDrive size={16} color="#a78bfa" />
              </div>
              <div style={{ color: '#fff', fontSize: '32px', fontWeight: 'bold' }}>{database?.investigations || 0}</div>
            </div>
            <div style={{ padding: '20px', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '14px', marginBottom: '10px', fontWeight: 600 }}>
                Total Reports <ShieldCheck size={16} color="#38bdf8" />
              </div>
              <div style={{ color: '#fff', fontSize: '32px', fontWeight: 'bold' }}>{database?.reports || 0}</div>
            </div>
          </div>
        </div>

      </div>

      {/* Orchestrator / Fusion Health */}
      <div style={{ ...cardStyle, borderTop: '4px solid #f59e0b' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '25px' }}>
          <GitMerge size={24} style={{ color: '#f59e0b' }} />
          <h2 style={{ margin: 0, fontSize: '20px', color: '#fff' }}>Orchestrator & Fusion Engine</h2>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '25px' }}>
           <div style={{ padding: '25px', backgroundColor: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '10px', transition: 'transform 0.2s' }} onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseLeave={e => e.currentTarget.style.transform = 'none'}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <Settings2 style={{ color: '#38bdf8' }} size={24} />
                <div style={{ fontWeight: 600, color: '#fff', fontSize: '18px' }}>Orchestrator</div>
              </div>
              <div style={{ fontSize: '14px', color: 'var(--text-muted)', marginBottom: '20px' }}>Multi-Agent Lifecycle Manager</div>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', color: system?.status === 'healthy' ? '#10b981' : '#fbbf24', fontWeight: 600, backgroundColor: system?.status === 'healthy' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(251, 191, 36, 0.1)', padding: '6px 16px', borderRadius: '20px', fontSize: '14px' }}>
                {system?.status === 'healthy' ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
                {system?.status === 'healthy' ? 'Healthy' : 'Degraded'}
              </div>
           </div>

           <div style={{ padding: '25px', backgroundColor: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '10px', transition: 'transform 0.2s' }} onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseLeave={e => e.currentTarget.style.transform = 'none'}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <GitMerge style={{ color: '#a78bfa' }} size={24} />
                <div style={{ fontWeight: 600, color: '#fff', fontSize: '18px' }}>Decision Fusion</div>
              </div>
              <div style={{ fontSize: '14px', color: 'var(--text-muted)', marginBottom: '20px' }}>Version {fusion?.version || '17.0.0'}</div>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', color: fusion?.status === 'operational' ? '#10b981' : '#f43f5e', fontWeight: 600, backgroundColor: fusion?.status === 'operational' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)', padding: '6px 16px', borderRadius: '20px', fontSize: '14px' }}>
                {fusion?.status === 'operational' ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                {fusion?.status === 'operational' ? 'Operational' : 'Offline'}
              </div>
           </div>

           <div style={{ padding: '25px', backgroundColor: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '10px', display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '25px' }}>
                <Zap style={{ color: '#fbbf24' }} size={24} />
                <div style={{ fontWeight: 600, color: '#fff', fontSize: '18px' }}>Fusion Parameters</div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px', fontSize: '15px', paddingBottom: '10px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                 <span style={{ color: 'var(--text-muted)' }}>Active Agents:</span>
                 <span style={{ color: '#fff', fontWeight: 'bold' }}>{fusion?.active_agents || 0} / 6</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '15px' }}>
                 <span style={{ color: 'var(--text-muted)' }}>Min. Consensus:</span>
                 <span style={{ color: '#fff', fontWeight: 'bold' }}>{fusion?.minimum_consensus || 2}</span>
              </div>
           </div>
        </div>
      </div>

    </div>
  );
}
