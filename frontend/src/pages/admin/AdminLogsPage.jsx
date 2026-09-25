import React, { useState, useEffect } from 'react';
import { ListOrdered, Search, Filter, ShieldCheck, Activity, LogIn, LogOut, SearchCheck, Clock, User, HardDrive, Calendar, X, Download, ShieldAlert, Key, UserCheck } from 'lucide-react';
import { getAuditLogs } from '../../services/api';
import { formatIST, formatDuration } from '../../utils/dateUtils';

export default function AdminLogsPage() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('ALL');
  const [selectedLog, setSelectedLog] = useState(null);
  const [showLiveTimer, setShowLiveTimer] = useState(false);
  
  useEffect(() => {
    fetchLogs();
    
    // Auto refresh active session durations
    const interval = setInterval(() => {
      setLogs(currentLogs => {
        const hasActive = currentLogs.some(l => l.isActiveSession);
        if (!hasActive) return currentLogs;
        
        return currentLogs.map(log => {
          if (log.isActiveSession) {
            const loginTime = new Date(log.created_at + (log.created_at.endsWith('Z') ? '' : 'Z')).getTime();
            return { ...log, durationMs: Date.now() - loginTime };
          }
          return log;
        });
      });
      
      setSelectedLog(currentSelection => {
        if (!currentSelection || !currentSelection.isActiveSession) return currentSelection;
        const loginTime = new Date(currentSelection.created_at + (currentSelection.created_at.endsWith('Z') ? '' : 'Z')).getTime();
        return { ...currentSelection, durationMs: Date.now() - loginTime };
      });
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const res = await getAuditLogs(200); // Fetch up to 200 recent logs
      if (res && res.audit_logs) {
        setLogs(processLogs(res.audit_logs));
      }
    } catch (err) {
      console.error("Failed to fetch logs:", err);
    } finally {
      setLoading(false);
    }
  };

  const processLogs = (rawLogs) => {
    const processed = [...rawLogs].map(l => ({...l}));
    const now = Date.now();
    
    for (let i = 0; i < processed.length; i++) {
      if (processed[i].event_type === 'LOGIN_SUCCESS') {
        const loginTimeStr = processed[i].created_at;
        const loginTime = new Date(loginTimeStr + (loginTimeStr.endsWith('Z') ? '' : 'Z')).getTime();
        
        let logoutFound = false;
        // Since logs are DESC (newest at index 0), newer logs are at j < i
        for (let j = i - 1; j >= 0; j--) {
          if (processed[j].event_type === 'LOGOUT' && processed[j].user_id === processed[i].user_id) {
            const logoutTimeStr = processed[j].created_at;
            const logoutTime = new Date(logoutTimeStr + (logoutTimeStr.endsWith('Z') ? '' : 'Z')).getTime();
            
            processed[i].durationMs = logoutTime - loginTime;
            processed[i].isActiveSession = false;
            processed[j].durationMs = logoutTime - loginTime; // tag the logout event too
            logoutFound = true;
            break;
          }
          // If we hit another LOGIN_SUCCESS for the same user, the prev session was abandoned
          if (processed[j].event_type === 'LOGIN_SUCCESS' && processed[j].user_id === processed[i].user_id) {
            break;
          }
        }
        
        if (!logoutFound) {
          processed[i].durationMs = now - loginTime;
          processed[i].isActiveSession = true;
        }
      }
    }
    return processed;
  };

  const exportLogCSV = (log) => {
    let durStr = 'N/A';
    if (log.durationMs !== undefined) {
       durStr = log.isActiveSession ? `Active Session (${formatDuration(log.durationMs)})` : formatDuration(log.durationMs);
    }
    
    const csvContent = "data:text/csv;charset=utf-8," + 
      "Key,Value\n" +
      `Timestamp,${formatIST(log.created_at)}\n` +
      `User ID,${log.user_id || 'System'}\n` +
      `Username,${log.username || 'System'}\n` +
      `Event Type,${log.event_type}\n` +
      `IP/Device,${log.ip_address || 'N/A'}\n` +
      `Duration,${durStr}\n` +
      `Action Details,"${(log.details || '').replace(/"/g, '""')}"`;
      
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `audit_log_${log.id || new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getEventIcon = (type) => {
    if (type === 'LOGIN_FAILURE') return <ShieldAlert size={16} color="#f43f5e" />;
    if (type === 'PASSWORD_RESET') return <Key size={16} color="#eab308" />;
    if (type === 'USER_ENABLED') return <UserCheck size={16} color="#3b82f6" />;
    if (type.includes('LOGIN')) return <LogIn size={16} color="#10b981" />;
    if (type.includes('LOGOUT')) return <LogOut size={16} color="#f59e0b" />;
    if (type.includes('SEARCH')) return <SearchCheck size={16} color="#a78bfa" />;
    if (type.includes('ACTIVITY') || type.includes('CREATE') || type.includes('DELETE') || type.includes('CHANGE') || type.includes('UPDATE')) return <Activity size={16} color="#38bdf8" />;
    return <HardDrive size={16} color="#94a3b8" />;
  };

  const getEventBadge = (type) => {
    let bg = 'rgba(148, 163, 184, 0.1)';
    let color = '#94a3b8';
    
    if (type === 'LOGIN_FAILURE') { bg = 'rgba(244, 63, 94, 0.1)'; color = '#f43f5e'; }
    else if (type === 'PASSWORD_RESET') { bg = 'rgba(234, 179, 8, 0.1)'; color = '#eab308'; }
    else if (type === 'USER_ENABLED') { bg = 'rgba(59, 130, 246, 0.1)'; color = '#3b82f6'; }
    else if (type.includes('LOGIN')) { bg = 'rgba(16, 185, 129, 0.1)'; color = '#10b981'; }
    else if (type.includes('LOGOUT')) { bg = 'rgba(245, 158, 11, 0.1)'; color = '#f59e0b'; }
    else if (type.includes('SEARCH')) { bg = 'rgba(167, 139, 250, 0.1)'; color = '#a78bfa'; }
    else if (type.includes('ACTIVITY') || type.includes('CREATE') || type.includes('DELETE') || type.includes('CHANGE') || type.includes('UPDATE')) { bg = 'rgba(56, 189, 248, 0.1)'; color = '#38bdf8'; }

    return (
      <span style={{ backgroundColor: bg, color: color, padding: '4px 10px', borderRadius: '20px', fontSize: '11px', fontWeight: 'bold', display: 'inline-flex', alignItems: 'center', gap: '6px', border: `1px solid ${bg}` }}>
        {getEventIcon(type)}
        {type.replace('_', ' ')}
      </span>
    );
  };

  const filteredLogs = logs.filter(log => {
    const matchesSearch = 
      (log.username && log.username.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (log.user_id && log.user_id.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (log.details && log.details.toLowerCase().includes(searchTerm.toLowerCase()));
      
    const matchesType = filterType === 'ALL' || 
                        (filterType === 'LOGIN' && (log.event_type.includes('LOGIN') || log.event_type.includes('LOGOUT'))) ||
                        (filterType === 'ACTIVITY' && (log.event_type.includes('ACTIVITY') || log.event_type.includes('CREATE') || log.event_type.includes('DELETE') || log.event_type.includes('CHANGE'))) ||
                        (filterType === 'SEARCH' && log.event_type.includes('SEARCH'));

    return matchesSearch && matchesType;
  });

  return (
    <div className="admin-logs" style={{ paddingBottom: '40px' }}>
      
      {/* Header section */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px', flexWrap: 'wrap', gap: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <ListOrdered size={28} style={{ color: 'var(--cyber-blue)' }} />
          <div>
            <h2 style={{ margin: 0, color: '#fff', fontSize: '24px' }}>System Activity Logs</h2>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', marginTop: '4px' }}>Secure Audit Trail for Super Admins</div>
          </div>
        </div>
        
        {/* Controls */}
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '8px 15px', width: '250px' }}>
            <Search size={16} style={{ color: 'var(--text-muted)', marginRight: '8px' }} />
            <input 
              type="text" 
              placeholder="Search User ID, Name, Action..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ background: 'transparent', border: 'none', color: '#fff', outline: 'none', width: '100%', fontSize: '14px' }} 
            />
          </div>
          
          <select 
            value={filterType} 
            onChange={(e) => setFilterType(e.target.value)}
            style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', color: '#fff', padding: '9px 15px', borderRadius: '8px', cursor: 'pointer', outline: 'none' }}
          >
            <option value="ALL">All Activities</option>
            <option value="LOGIN">Auth & Sessions</option>
            <option value="ACTIVITY">System Changes</option>
            <option value="SEARCH">Queries & Searches</option>
          </select>
          
          <button onClick={fetchLogs} style={{ display: 'flex', alignItems: 'center', gap: '6px', backgroundColor: 'var(--cyber-blue)', border: 'none', color: '#000', padding: '9px 20px', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>
            Refresh
          </button>
        </div>
      </div>

      {/* Main Table Container */}
      <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '12px', overflow: 'hidden', boxShadow: '0 4px 6px rgba(0,0,0,0.2)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ backgroundColor: 'rgba(255,255,255,0.02)', borderBottom: '1px solid var(--border-subtle)' }}>
              <th style={{ padding: '16px 20px', color: 'var(--text-main)', fontWeight: 600, width: '220px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Calendar size={16} /> Timestamp (IST)
              </th>
              <th style={{ padding: '16px 20px', color: 'var(--text-main)', fontWeight: 600, width: '180px' }}>
                <User size={16} style={{ verticalAlign: 'text-bottom', marginRight: '6px' }} /> User
              </th>
              <th style={{ padding: '16px 20px', color: 'var(--text-main)', fontWeight: 600, width: '180px' }}>Activity Type</th>
              <th style={{ padding: '16px 20px', color: 'var(--text-main)', fontWeight: 600 }}>Action Details</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan="4" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  <Activity className="animate-spin" size={24} style={{ margin: '0 auto 10px', color: 'var(--cyber-blue)' }} />
                  Loading audit logs...
                </td>
              </tr>
            ) : filteredLogs.length === 0 ? (
              <tr>
                <td colSpan="4" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  No logs found matching your filters.
                </td>
              </tr>
            ) : (
              filteredLogs.map((log) => {
                const istSplit = formatIST(log.created_at).split(' ');
                const istDate = istSplit[0];
                const istTime = istSplit[1] + ' ' + (istSplit[2] || '');
                return (
                <tr key={log.id} style={{ 
                  borderBottom: '1px solid rgba(255,255,255,0.03)', 
                  transition: 'background-color 0.2s',
                  backgroundColor: 'transparent',
                  cursor: 'pointer'
                }}
                onClick={() => setSelectedLog(log)}
                onMouseEnter={e => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.02)'}
                onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}
                >
                  <td style={{ padding: '16px 20px', color: 'var(--text-muted)', fontSize: '13px' }}>
                    <div style={{ color: '#fff', fontWeight: 500 }}>{istDate}</div>
                    <div style={{ marginTop: '4px' }}>{istTime}</div>
                  </td>
                  
                  <td style={{ padding: '16px 20px' }}>
                    <div style={{ color: '#fff', fontWeight: 600, fontSize: '14px' }}>{log.username || 'System'}</div>
                    {log.user_id && <div style={{ color: 'var(--text-faint)', fontSize: '12px', marginTop: '4px' }}>ID: {log.user_id.substring(0, 8)}...</div>}
                  </td>
                  
                  <td style={{ padding: '16px 20px' }}>
                    {getEventBadge(log.event_type)}
                  </td>
                  
                  <td style={{ padding: '16px 20px', color: 'var(--text-muted)', fontSize: '14px', lineHeight: '1.5' }}>
                    {log.details.length > 80 ? log.details.substring(0, 80) + '...' : log.details}
                    {log.ip_address && (
                       <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-faint)' }}>
                         IP: {log.ip_address}
                       </div>
                    )}
                  </td>
                </tr>
              )})
            )}
          </tbody>
        </table>
      </div>

      {/* Log Details Modal */}
      {selectedLog && (
        <div style={{
          position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
          backgroundColor: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
          display: 'flex', justifyContent: 'center', alignItems: 'center',
          zIndex: 9999
        }} onClick={() => setSelectedLog(null)}>
          <div style={{
            backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
            borderRadius: '12px', width: '600px', maxWidth: '90%', maxHeight: '85vh',
            display: 'flex', flexDirection: 'column', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)'
          }} onClick={e => e.stopPropagation()}>
            
            {/* Modal Header */}
            <div style={{ padding: '20px 24px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                <ShieldCheck size={28} style={{ color: 'var(--cyber-blue)' }} />
                <div>
                  <h3 style={{ margin: 0, color: '#fff', fontSize: '18px' }}>Activity Detail</h3>
                  <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>Reference: {selectedLog.id}</div>
                </div>
              </div>
              <button onClick={() => setSelectedLog(null)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', padding: '5px' }}>
                <X size={24} />
              </button>
            </div>

            {/* Modal Body */}
            <div style={{ padding: '24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
              
              <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
                <div style={{ flex: 1, backgroundColor: 'rgba(255,255,255,0.02)', padding: '15px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ color: 'var(--text-faint)', fontSize: '12px', marginBottom: '5px' }}>Activity Type</div>
                  {getEventBadge(selectedLog.event_type)}
                </div>
                <div style={{ flex: 2, backgroundColor: 'rgba(255,255,255,0.02)', padding: '15px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ color: 'var(--text-faint)', fontSize: '12px', marginBottom: '5px' }}>Timestamp (IST)</div>
                  <div style={{ color: '#fff', fontWeight: 600, fontSize: '14px' }}>{formatIST(selectedLog.created_at)}</div>
                </div>
              </div>

              <div style={{ backgroundColor: 'rgba(255,255,255,0.02)', padding: '15px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ color: 'var(--text-faint)', fontSize: '12px', marginBottom: '10px' }}>User Information</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                  <div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Username</div>
                    <div style={{ color: '#fff', fontWeight: 600, fontSize: '14px', marginTop: '2px' }}>{selectedLog.username || 'System'}</div>
                  </div>
                  <div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>User ID</div>
                    <div style={{ color: 'var(--text-main)', fontSize: '14px', marginTop: '2px' }}>{selectedLog.user_id || 'N/A'}</div>
                  </div>
                </div>
              </div>

              <div style={{ backgroundColor: 'rgba(255,255,255,0.02)', padding: '15px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ color: 'var(--text-faint)', fontSize: '12px', marginBottom: '10px' }}>Action Details (Changes / Searches)</div>
                <div style={{ color: '#fff', fontSize: '14px', lineHeight: '1.6', wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>
                  {selectedLog.details}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                <div style={{ backgroundColor: 'rgba(255,255,255,0.02)', padding: '15px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ color: 'var(--text-faint)', fontSize: '12px', marginBottom: '5px' }}>Network / IP Address</div>
                  <div style={{ color: '#fff', fontWeight: 600, fontSize: '14px' }}>{selectedLog.ip_address || 'N/A'}</div>
                </div>
                <div style={{ backgroundColor: 'rgba(255,255,255,0.02)', padding: '15px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ color: 'var(--text-faint)', fontSize: '12px', marginBottom: '5px' }}>Session Duration</div>
                  <div 
                    onClick={() => { if (selectedLog.durationMs !== undefined) setShowLiveTimer(true) }}
                    style={{ 
                      color: selectedLog.durationMs !== undefined ? (selectedLog.isActiveSession ? '#10b981' : '#f59e0b') : 'var(--text-muted)', 
                      fontWeight: 600, fontSize: '14px', display: 'flex', alignItems: 'center', gap: '6px',
                      cursor: selectedLog.durationMs !== undefined ? 'pointer' : 'default',
                      padding: '4px 8px', borderRadius: '6px',
                      backgroundColor: selectedLog.durationMs !== undefined ? (selectedLog.isActiveSession ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)') : 'transparent',
                      border: selectedLog.durationMs !== undefined ? `1px solid ${selectedLog.isActiveSession ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)'}` : 'none',
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                      if (selectedLog.durationMs !== undefined) {
                        e.currentTarget.style.backgroundColor = selectedLog.isActiveSession ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedLog.durationMs !== undefined) {
                        e.currentTarget.style.backgroundColor = selectedLog.isActiveSession ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)';
                      }
                    }}
                  >
                    {selectedLog.durationMs !== undefined ? <><Clock size={16} /> {selectedLog.isActiveSession ? 'Active Session (' : ''}{formatDuration(selectedLog.durationMs)}{selectedLog.isActiveSession ? ')' : ''}</> : 'N/A (Not a session event)'}
                  </div>
                </div>
              </div>

              {selectedLog.event_type === 'LOGIN_SUCCESS' && (
                <div style={{ backgroundColor: 'rgba(16, 185, 129, 0.05)', padding: '15px', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.2)', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <ShieldCheck size={20} color="#10b981" />
                  <div>
                    <div style={{ color: '#10b981', fontSize: '13px', fontWeight: 'bold' }}>Authentication Verified</div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '12px', marginTop: '2px' }}>Credentials successfully validated. Passwords are stored securely via one-way salted hash.</div>
                  </div>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div style={{ padding: '16px 24px', borderTop: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'flex-end', gap: '15px' }}>
              <button onClick={() => exportLogCSV(selectedLog)} style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'transparent', border: '1px solid var(--border-subtle)', color: 'var(--text-main)', padding: '8px 16px', borderRadius: '6px', cursor: 'pointer' }}>
                <Download size={16} /> Export to CSV
              </button>
              <button onClick={() => setSelectedLog(null)} style={{ backgroundColor: 'var(--cyber-blue)', border: 'none', color: '#000', padding: '8px 20px', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Live Timer Secondary Pop-up */}
      {showLiveTimer && selectedLog && (
        <div style={{
          position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
          backgroundColor: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(8px)',
          display: 'flex', justifyContent: 'center', alignItems: 'center',
          zIndex: 10000
        }} onClick={() => setShowLiveTimer(false)}>
          <div style={{
            backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
            borderRadius: '16px', width: '400px', maxWidth: '90%', padding: '30px',
            display: 'flex', flexDirection: 'column', alignItems: 'center',
            boxShadow: '0 25px 50px -12px rgba(0,0,0,0.8)'
          }} onClick={e => e.stopPropagation()}>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
               <div style={{ color: 'var(--text-muted)' }}>Session Timer</div>
               <X size={20} style={{ cursor: 'pointer', color: 'var(--text-muted)' }} onClick={() => setShowLiveTimer(false)} />
            </div>
            
            <User size={48} color="var(--cyber-blue)" style={{ marginBottom: '15px' }} />
            <h3 style={{ color: '#fff', margin: '0 0 5px 0', fontSize: '20px' }}>{selectedLog.username}</h3>
            <div style={{ color: 'var(--text-faint)', fontSize: '13px', marginBottom: '30px' }}>Ref: {selectedLog.id}</div>
            
            <div style={{
              backgroundColor: selectedLog.isActiveSession ? 'rgba(16, 185, 129, 0.05)' : 'rgba(245, 158, 11, 0.05)',
              border: `1px solid ${selectedLog.isActiveSession ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)'}`,
              borderRadius: '12px',
              padding: '25px',
              width: '100%',
              textAlign: 'center',
              marginBottom: '20px'
            }}>
              <div style={{ color: selectedLog.isActiveSession ? '#10b981' : '#f59e0b', fontSize: '14px', fontWeight: 'bold', marginBottom: '15px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                <Clock size={18} /> {selectedLog.isActiveSession ? 'LIVE ACTIVE SESSION' : 'COMPLETED SESSION'}
              </div>
              <div style={{ color: '#fff', fontSize: '36px', fontWeight: 'bold', fontFamily: 'monospace', letterSpacing: '1px' }}>
                {formatDuration(selectedLog.durationMs)}
              </div>
            </div>
            
            <button onClick={() => setShowLiveTimer(false)} style={{ backgroundColor: 'var(--cyber-blue)', border: 'none', color: '#000', padding: '10px 30px', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', width: '100%' }}>
              Close Timer
            </button>
          </div>
        </div>
      )}

    </div>
  );
}
