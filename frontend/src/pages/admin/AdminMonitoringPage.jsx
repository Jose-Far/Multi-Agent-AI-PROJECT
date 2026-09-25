import React, { useState, useEffect } from 'react';
import { getAdminMonitoring } from '../../services/api';
import { Activity, Server, Database, Clock, HardDrive, Cpu, CheckCircle2, XCircle, AlertTriangle, RefreshCw } from 'lucide-react';

export default function AdminMonitoringPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchMonitoring = async () => {
    setIsRefreshing(true);
    try {
      const res = await getAdminMonitoring();
      setData(res);
      setError(null);
    } catch (err) {
      setError("Monitoring data unavailable");
    } finally {
      setLoading(false);
      setTimeout(() => setIsRefreshing(false), 500); // Visual feedback
    }
  };

  useEffect(() => {
    fetchMonitoring();
    const interval = setInterval(fetchMonitoring, 30000); // auto refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const cardStyle = {
    backgroundColor: 'var(--bg-card)',
    border: '1px solid var(--border-subtle)',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    transition: 'all 0.3s ease',
  };

  if (loading && !data) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh', color: 'var(--cyber-blue)' }}>
        <Activity className="animate-spin" size={48} />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div style={{ ...cardStyle, borderColor: '#f43f5e', textAlign: 'center', marginTop: '40px' }}>
        <XCircle size={48} color="#f43f5e" style={{ margin: '0 auto 15px' }} />
        <h3 style={{ color: '#fff', marginBottom: '10px' }}>{error}</h3>
        <button onClick={fetchMonitoring} className="btn-primary" style={{ padding: '8px 24px' }}>Retry Connection</button>
      </div>
    );
  }

  const isOperational = data?.status === 'operational';

  return (
    <div className="admin-monitoring" style={{ paddingBottom: '40px' }}>
      {/* HEADER SECTION */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <Activity size={28} style={{ color: 'var(--cyber-blue)' }} />
          <h2 style={{ margin: 0, color: '#fff', fontSize: '24px' }}>Real-Time System Monitoring</h2>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', color: 'var(--text-muted)', fontSize: '14px' }}>
          <span>Auto-refresh: 30s</span>
          <button 
            onClick={fetchMonitoring} 
            style={{ background: 'transparent', border: 'none', color: 'var(--cyber-blue)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}
          >
            <RefreshCw size={16} className={isRefreshing ? 'animate-spin' : ''} />
            Refresh Now
          </button>
        </div>
      </div>

      {/* SYSTEM STATUS GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px', marginBottom: '30px' }}>
        
        <div style={{ ...cardStyle, borderTop: `4px solid ${isOperational ? '#10b981' : '#ef4444'}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px', color: 'var(--text-muted)' }}>
            <span style={{ fontSize: '14px', fontWeight: 600 }}>System Status</span>
            <Server size={18} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '24px', fontWeight: 'bold', color: '#fff', textTransform: 'capitalize' }}>
            {isOperational ? <CheckCircle2 size={24} color="#10b981" /> : <AlertTriangle size={24} color="#ef4444" />}
            {data?.status}
          </div>
        </div>

        <div style={{ ...cardStyle, borderTop: `4px solid ${data?.api.status === 'online' ? '#38bdf8' : '#ef4444'}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px', color: 'var(--text-muted)' }}>
            <span style={{ fontSize: '14px', fontWeight: 600 }}>API Gateway</span>
            <Activity size={18} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '24px', fontWeight: 'bold', color: '#fff', textTransform: 'capitalize' }}>
            {data?.api.status === 'online' ? <CheckCircle2 size={24} color="#38bdf8" /> : <XCircle size={24} color="#ef4444" />}
            {data?.api.status}
          </div>
        </div>

        <div style={{ ...cardStyle, borderTop: `4px solid ${data?.database.status === 'connected' ? '#a78bfa' : '#ef4444'}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px', color: 'var(--text-muted)' }}>
            <span style={{ fontSize: '14px', fontWeight: 600 }}>Database Health</span>
            <Database size={18} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '24px', fontWeight: 'bold', color: '#fff', textTransform: 'capitalize' }}>
            {data?.database.status === 'connected' ? <CheckCircle2 size={24} color="#a78bfa" /> : <XCircle size={24} color="#ef4444" />}
            {data?.database.status}
          </div>
          <div style={{ marginTop: '10px', fontSize: '13px', color: 'var(--text-faint)' }}>
            {data?.database.size_mb} MB Storage | {data?.database.investigations} Logs
          </div>
        </div>

        <div style={{ ...cardStyle, borderTop: `4px solid #fbbf24` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px', color: 'var(--text-muted)' }}>
            <span style={{ fontSize: '14px', fontWeight: 600 }}>Backend Uptime</span>
            <Clock size={18} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fff' }}>
            {data?.resources.uptime}
          </div>
        </div>
      </div>

      {/* SYSTEM RESOURCES (PROGRESS BARS) */}
      <div style={{ ...cardStyle, marginBottom: '30px' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: 0, marginBottom: '25px', color: '#fff', fontSize: '18px' }}>
          <Cpu size={20} style={{ color: 'var(--cyber-blue)' }} /> Hardware Utilization
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '30px' }}>
          
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', color: 'var(--text-main)', fontWeight: 600 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Cpu size={16} color="#38bdf8" /> CPU Usage</span>
              <span>{data?.resources.cpu_percent}%</span>
            </div>
            <div style={{ width: '100%', height: '10px', backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: '5px', overflow: 'hidden' }}>
              <div style={{ width: `${data?.resources.cpu_percent}%`, height: '100%', backgroundColor: data?.resources.cpu_percent > 85 ? '#ef4444' : '#38bdf8', borderRadius: '5px', transition: 'width 0.5s ease-out' }}></div>
            </div>
          </div>
          
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', color: 'var(--text-main)', fontWeight: 600 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Database size={16} color="#a78bfa" /> Memory (RAM)</span>
              <span>{data?.resources.ram_percent}%</span>
            </div>
            <div style={{ width: '100%', height: '10px', backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: '5px', overflow: 'hidden' }}>
              <div style={{ width: `${data?.resources.ram_percent}%`, height: '100%', backgroundColor: data?.resources.ram_percent > 85 ? '#ef4444' : '#a78bfa', borderRadius: '5px', transition: 'width 0.5s ease-out' }}></div>
            </div>
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', color: 'var(--text-main)', fontWeight: 600 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><HardDrive size={16} color="#fbbf24" /> Disk Storage</span>
              <span>{data?.resources.disk_percent}%</span>
            </div>
            <div style={{ width: '100%', height: '10px', backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: '5px', overflow: 'hidden' }}>
              <div style={{ width: `${data?.resources.disk_percent}%`, height: '100%', backgroundColor: data?.resources.disk_percent > 85 ? '#ef4444' : '#fbbf24', borderRadius: '5px', transition: 'width 0.5s ease-out' }}></div>
            </div>
          </div>

        </div>
      </div>

      {/* AGENT HEALTH CARDS */}
      <div>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px', color: '#fff', fontSize: '18px' }}>
          <Activity size={20} style={{ color: '#10b981' }} /> Agent Fleet Health
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '20px' }}>
          {data?.agents && Object.entries(data.agents).map(([key, agent]) => {
            const isHealthy = agent.status === 'healthy';
            const isFallback = agent.model.toLowerCase().includes('fallback');
            
            return (
              <div key={key} style={{ 
                backgroundColor: 'var(--bg-card)',
                border: '1px solid var(--border-subtle)',
                borderLeft: `4px solid ${isHealthy ? (isFallback ? '#fbbf24' : '#10b981') : '#ef4444'}`,
                borderRadius: '8px',
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                gap: '12px',
                transition: 'transform 0.2s ease',
              }}
              onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'none'}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h4 style={{ margin: 0, textTransform: 'capitalize', color: '#fff', fontSize: '16px' }}>{agent.name}</h4>
                  {isHealthy ? <CheckCircle2 size={20} color={isFallback ? "#fbbf24" : "#10b981"} /> : <XCircle size={20} color="#ef4444" />}
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '6px' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Agent Status:</span>
                  <strong style={{ color: isHealthy ? '#10b981' : '#ef4444', fontSize: '13px', textTransform: 'capitalize' }}>
                    {agent.status}
                  </strong>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '6px' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Model State:</span>
                  <strong style={{ color: isFallback ? '#fbbf24' : 'var(--text-main)', fontSize: '13px' }}>
                    {agent.model}
                  </strong>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Feature Dimensions:</span>
                  <span style={{ color: 'var(--text-faint)', fontSize: '13px', fontWeight: 'bold' }}>
                    {agent.features > 0 ? agent.features : <span style={{ color: '#ef4444' }}>0 (Error)</span>}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}
