import React from 'react';
import { Activity, Server, Database, Cpu } from 'lucide-react';

export default function AdminOverviewPage({ healthData, apiOnline }) {
  // Fix: /admin/system returns system: { status: 'operational' | 'degraded' }
  // And agents: { available: 6, total: 6 }
  const isHealthy = apiOnline && (healthData?.system?.status === 'operational' || healthData?.system?.status === 'healthy' || healthData?.status === 'operational');
  const activeAgents = healthData?.agents?.available || healthData?.active_agents || 0;
  const totalAgents = healthData?.agents?.total || 6;

  return (
    <div className="admin-overview">
      <div className="admin-stat-grid">
        
        <div style={{
          backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 600 }}>API STATUS</div>
            <Server size={20} style={{ color: 'var(--cyber-blue)' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: apiOnline ? '#10b981' : '#f43f5e' }}>
            {apiOnline ? 'ONLINE' : 'OFFLINE'}
          </div>
        </div>

        <div style={{
          backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 600 }}>DATABASE</div>
            <Database size={20} style={{ color: 'var(--cyber-blue)' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: apiOnline ? '#10b981' : '#f43f5e' }}>
            {apiOnline ? 'ONLINE' : 'OFFLINE'}
          </div>
        </div>

        <div style={{
          backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 600 }}>AI AGENTS</div>
            <Cpu size={20} style={{ color: 'var(--cyber-blue)' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--text-main)' }}>
            {activeAgents} / {totalAgents}
          </div>
        </div>

        <div style={{
          backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 600 }}>SYSTEM</div>
            <Activity size={20} style={{ color: 'var(--cyber-blue)' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: isHealthy ? '#10b981' : '#fbbf24' }}>
            {isHealthy ? 'HEALTHY' : (apiOnline ? 'DEGRADED' : 'DOWN')}
          </div>
        </div>

      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
        <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px' }}>
          <h3 style={{ margin: '0 0 15px 0', fontSize: '16px', color: 'var(--text-main)' }}>System Components</h3>
          <div style={{ color: 'var(--text-faint)' }}>Detailed component breakdown will be built in upcoming steps.</div>
        </div>
        
        <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px' }}>
          <h3 style={{ margin: '0 0 15px 0', fontSize: '16px', color: 'var(--text-main)' }}>Recent Activity</h3>
          <div style={{ color: 'var(--text-faint)' }}>Activity logs will be built in Step 14.</div>
        </div>
      </div>
    </div>
  );
}
