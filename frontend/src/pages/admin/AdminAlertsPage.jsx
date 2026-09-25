
import React, { useState } from 'react';
import { BellRing, AlertTriangle, AlertOctagon, Info, ShieldAlert, CheckCircle, Search, Filter, RefreshCw, X, Clock, ExternalLink } from 'lucide-react';
import { formatIST } from '../../utils/dateUtils';

export default function AdminAlertsPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterLevel, setFilterLevel] = useState('ALL');
  
  // Real-looking dummy alerts
  const [alerts, setAlerts] = useState([
    { id: 'ALT-9921', time: new Date(Date.now() - 1000 * 60 * 15).toISOString(), component: 'DNS Agent', message: 'DNS Agent response timeout detected', details: 'The DNS enrichment agent took longer than 15,000ms to respond for domain: sp0tify-mod-login.com. Defaulting to fallback features.', level: 'ERROR', status: 'Investigating' },
    { id: 'ALT-9920', time: new Date(Date.now() - 1000 * 60 * 45).toISOString(), component: 'Threat Intelligence', message: 'Rate limit approaching for VirusTotal API', details: 'API quota is at 95% of the daily limit. (475/500 requests used). Recommend switching to secondary API key or caching results more aggressively.', level: 'WARNING', status: 'Investigating' },
    { id: 'ALT-9919', time: new Date(Date.now() - 1000 * 60 * 120).toISOString(), component: 'Authentication', message: 'Multiple failed logins detected', details: 'User admin experienced 5 failed login attempts within 2 minutes from IP 192.168.1.45. Account temporarily locked.', level: 'CRITICAL', status: 'Open' },
    { id: 'ALT-9918', time: new Date(Date.now() - 1000 * 60 * 60 * 4).toISOString(), component: 'Database', message: 'High memory usage on main cluster', details: 'SQLite disk I/O operations are causing memory pressure. Usage: 85%.', level: 'INFO', status: 'Resolved' },
  ]);

  const handleResolve = (id) => {
    setAlerts(alerts.map(a => a.id === id ? { ...a, status: 'Resolved' } : a));
  };

  const getLevelConfig = (level) => {
    switch (level) {
      case 'CRITICAL': return { color: '#f43f5e', bg: 'rgba(244, 63, 94, 0.1)', icon: <ShieldAlert size={20} /> };
      case 'ERROR': return { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', icon: <AlertOctagon size={20} /> };
      case 'WARNING': return { color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)', icon: <AlertTriangle size={20} /> };
      case 'INFO': return { color: '#3b82f6', bg: 'rgba(59, 130, 246, 0.1)', icon: <Info size={20} /> };
      default: return { color: '#94a3b8', bg: 'rgba(148, 163, 184, 0.1)', icon: <Info size={20} /> };
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'Open': return '#f43f5e';
      case 'Investigating': return 'var(--cyber-blue)';
      case 'Resolved': return '#10b981';
      default: return 'var(--text-muted)';
    }
  };

  const filteredAlerts = alerts.filter(a => {
    const matchesSearch = a.message.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          a.component.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          a.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesLevel = filterLevel === 'ALL' || a.level === filterLevel;
    return matchesSearch && matchesLevel;
  });

  return (
    <div className="admin-alerts" style={{ paddingBottom: '40px' }}>
      
      {/* Header section */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px', flexWrap: 'wrap', gap: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <BellRing size={28} style={{ color: 'var(--cyber-blue)' }} />
          <div>
            <h2 style={{ margin: 0, color: '#fff', fontSize: '24px' }}>System Alerts Configuration</h2>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', marginTop: '4px' }}>Monitor and manage real-time system anomalies</div>
          </div>
        </div>
        
        {/* Controls */}
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '8px 15px', width: '250px' }}>
            <Search size={16} style={{ color: 'var(--text-muted)', marginRight: '8px' }} />
            <input 
              type="text" 
              placeholder="Search alerts..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ background: 'transparent', border: 'none', color: '#fff', outline: 'none', width: '100%', fontSize: '14px' }} 
            />
          </div>
          
          <select 
            value={filterLevel} 
            onChange={(e) => setFilterLevel(e.target.value)}
            style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', color: '#fff', padding: '9px 15px', borderRadius: '8px', cursor: 'pointer', outline: 'none' }}
          >
            <option value="ALL">All Severities</option>
            <option value="CRITICAL">Critical</option>
            <option value="ERROR">Error</option>
            <option value="WARNING">Warning</option>
            <option value="INFO">Info</option>
          </select>
          
          <button style={{ display: 'flex', alignItems: 'center', gap: '6px', backgroundColor: 'var(--cyber-blue)', border: 'none', color: '#000', padding: '9px 20px', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>
            <RefreshCw size={16} /> Refresh
          </button>
        </div>
      </div>

      {/* Alerts List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {filteredAlerts.length === 0 ? (
          <div style={{ padding: '60px', textAlign: 'center', backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '12px', color: 'var(--text-faint)' }}>
            <CheckCircle size={48} style={{ color: '#10b981', margin: '0 auto 16px', opacity: 0.8 }} />
            <h3 style={{ color: '#fff', margin: '0 0 8px 0' }}>All Clear</h3>
            <p style={{ margin: 0, fontSize: '14px' }}>No alerts matching your filters. System is operating normally.</p>
          </div>
        ) : (
          filteredAlerts.map((alert) => {
            const config = getLevelConfig(alert.level);
            return (
              <div 
                key={alert.id} 
                className="alert-card"
                style={{ 
                  backgroundColor: 'var(--bg-card)', 
                  border: '1px solid var(--border-subtle)',
                  borderLeft: `4px solid ${config.color}`,
                  borderRadius: '12px', 
                  padding: '24px',
                  display: 'flex',
                  gap: '20px',
                  transition: 'all 0.2s',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                  position: 'relative',
                  overflow: 'hidden'
                }}
              >
                {/* Background Glow */}
                <div style={{
                  position: 'absolute',
                  top: 0, left: 0, width: '100px', height: '100%',
                  background: `linear-gradient(90deg, ${config.bg}, transparent)`,
                  pointerEvents: 'none'
                }} />

                <div style={{ marginTop: '2px', color: config.color, zIndex: 1 }}>
                  {config.icon}
                </div>

                <div style={{ flex: 1, zIndex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                        <span style={{ 
                          backgroundColor: config.bg, 
                          color: config.color, 
                          padding: '3px 8px', 
                          borderRadius: '4px', 
                          fontSize: '10px', 
                          fontWeight: 'bold',
                          letterSpacing: '0.5px'
                        }}>
                          {alert.level}
                        </span>
                        <span style={{ color: 'var(--text-faint)', fontSize: '12px', fontFamily: 'monospace' }}>
                          {alert.id}
                        </span>
                      </div>
                      <h3 style={{ margin: '0 0 10px 0', fontSize: '17px', color: '#fff', fontWeight: 600 }}>{alert.message}</h3>
                    </div>
                    
                    {alert.status !== 'Resolved' && (
                      <button 
                        onClick={() => handleResolve(alert.id)}
                        style={{ 
                          backgroundColor: 'transparent',
                          border: '1px solid var(--border-subtle)',
                          color: 'var(--text-main)',
                          padding: '6px 12px',
                          borderRadius: '6px',
                          fontSize: '12px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#10b981'; e.currentTarget.style.color = '#10b981'; }}
                        onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border-subtle)'; e.currentTarget.style.color = 'var(--text-main)'; }}
                      >
                        <CheckCircle size={14} /> Mark Resolved
                      </button>
                    )}
                  </div>

                  <p style={{ margin: '0 0 16px 0', color: 'var(--text-muted)', fontSize: '14px', lineHeight: '1.5' }}>
                    {alert.details}
                  </p>

                  <div style={{ display: 'flex', gap: '24px', color: 'var(--text-faint)', fontSize: '13px', backgroundColor: 'rgba(0,0,0,0.2)', padding: '10px 15px', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Clock size={14} />
                      <span>{formatIST(alert.time)}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <AlertOctagon size={14} />
                      <span>Component: <strong style={{ color: '#fff', fontWeight: 500 }}>{alert.component}</strong></span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Info size={14} />
                      <span>Status: <strong style={{ color: getStatusColor(alert.status), fontWeight: 500 }}>{alert.status}</strong></span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}


