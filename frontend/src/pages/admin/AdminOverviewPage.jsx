import React, { useState, useEffect } from 'react';
import { Activity, Server, Database, Cpu, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { getAuditLogs } from '../../services/api';

export default function AdminOverviewPage({ healthData, apiOnline }) {
  const [recentLogs, setRecentLogs] = useState([]);
  const [loadingLogs, setLoadingLogs] = useState(true);

  useEffect(() => {
    let mounted = true;
    const fetchLogs = async () => {
      try {
        const res = await getAuditLogs(5);
        if (mounted && res?.audit_logs) {
          setRecentLogs(res.audit_logs);
        }
      } catch (err) {
        console.error("Failed to fetch recent activity:", err);
      } finally {
        if (mounted) setLoadingLogs(false);
      }
    };
    fetchLogs();
    return () => { mounted = false; };
  }, []);

  const isHealthy = apiOnline && (healthData?.fusion?.status === 'operational' || healthData?.status === 'operational' || healthData?.system?.status === 'operational');
  const activeAgents = healthData?.agents?.available || healthData?.active_agents || 0;
  const totalAgents = healthData?.agents?.total || 6;

  // Render Component Row
  const renderComponent = (name, data) => {
    if (!data) return null;
    const isUp = data.status === 'online' || data.status === 'connected' || data.status === 'operational';
    return (
      <div className="system-component-row">
        <div>
          <div style={{ color: 'var(--text-main)', fontWeight: 600 }}>{name}</div>
          <div style={{ color: 'var(--text-muted)', fontSize: '12px', marginTop: '4px' }}>
            {data.engine || data.version || 'Core Component'}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isUp ? <CheckCircle size={16} color="#10b981" /> : <AlertCircle size={16} color="#f43f5e" />}
          <span style={{ color: isUp ? '#10b981' : '#f43f5e', fontSize: '13px', fontWeight: 600 }}>
            {data.status?.toUpperCase() || 'UNKNOWN'}
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="admin-overview">
      <style>{`
        .system-component-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 15px;
          background-color: var(--bg-card);
          border: 1px solid var(--border-subtle);
          border-radius: 6px;
          margin-bottom: 8px;
          transition: all 0.3s ease;
        }
        .system-component-row:hover {
          background-color: var(--bg-card-hover);
          border-color: var(--cyber-blue);
          box-shadow: 0 0 10px var(--cyber-glow);
          transform: translateX(4px);
        }
        .activity-log-row {
          display: flex;
          gap: 12px;
          align-items: flex-start;
          padding: 10px;
          border-radius: 6px;
          transition: all 0.2s ease;
        }
        .activity-log-row:hover {
          background-color: var(--bg-card-hover);
        }
      `}</style>
      <div className="admin-stat-grid">
        <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 600 }}>API STATUS</div>
            <Server size={20} style={{ color: 'var(--cyber-blue)' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: apiOnline ? '#10b981' : '#f43f5e' }}>
            {apiOnline ? 'ONLINE' : 'OFFLINE'}
          </div>
        </div>
        
        <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 600 }}>DATABASE</div>
            <Database size={20} style={{ color: 'var(--cyber-blue)' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: (apiOnline && healthData?.database?.status === 'connected') ? '#10b981' : '#f43f5e' }}>
            {(apiOnline && healthData?.database?.status === 'connected') ? 'ONLINE' : 'OFFLINE'}
          </div>
        </div>
        
        <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 600 }}>AI AGENTS</div>
            <Cpu size={20} style={{ color: 'var(--cyber-blue)' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--text-main)' }}>
            {activeAgents} / {totalAgents}
          </div>
        </div>
        
        <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 600 }}>SYSTEM</div>
            <Activity size={20} style={{ color: 'var(--cyber-blue)' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: isHealthy ? '#10b981' : '#fbbf24' }}>
            {isHealthy ? 'HEALTHY' : (apiOnline ? 'DEGRADED' : 'DOWN')}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', marginTop: '24px' }}>
        <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px' }}>
          <h3 style={{ margin: '0 0 15px 0', fontSize: '16px', color: 'var(--text-main)', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px' }}>System Components</h3>
          {healthData ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {renderComponent('Flask API Server', healthData.api)}
              {renderComponent('PostgreSQL/SQLite Database', healthData.database)}
              {renderComponent('Decision Fusion Engine', healthData.fusion)}
              {renderComponent('Agent Orchestrator', { status: 'operational', version: '2.0.0' })}
            </div>
          ) : (
            <div style={{ color: 'var(--text-faint)', fontStyle: 'italic', padding: '10px 0' }}>Component telemetry unavailable.</div>
          )}
        </div>
        
        <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px' }}>
          <h3 style={{ margin: '0 0 15px 0', fontSize: '16px', color: 'var(--text-main)', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px' }}>Recent Activity</h3>
          {loadingLogs ? (
            <div style={{ color: 'var(--text-faint)', fontStyle: 'italic', padding: '10px 0' }}>Loading logs...</div>
          ) : recentLogs.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '5px', marginTop: '10px' }}>
              {recentLogs.map((log) => (
                <div key={log.id} className="activity-log-row">
                  <Clock size={16} color="var(--cyber-blue)" style={{ marginTop: '2px', flexShrink: 0 }} />
                  <div>
                    <div style={{ color: 'var(--text-main)', fontSize: '14px', lineHeight: '1.4' }}>
                      <span style={{ fontWeight: 600 }}>{log.username || 'System'}</span> {(log.event_type || log.action || 'Unknown Event').replace(/_/g, ' ').toLowerCase()}
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '12px', marginTop: '4px' }}>
                      {new Date(log.created_at || Date.now()).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', dateStyle: 'medium', timeStyle: 'short' })}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: 'var(--text-faint)', fontStyle: 'italic', padding: '10px 0' }}>No recent activity found.</div>
          )}
        </div>
      </div>
    </div>
  );
}
