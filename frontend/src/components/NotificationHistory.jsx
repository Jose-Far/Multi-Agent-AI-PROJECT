import React, { useState, useEffect } from 'react';
import { ShieldAlert, AlertTriangle, Activity, Info, Search, Check, Filter } from 'lucide-react';
import { getAuthHeaders } from '../services/api';
import { useAuth } from '../services/AuthContext';
import { formatIST } from '../utils/dateUtils';


const SEVERITY_COLORS = {
  critical: '#f43f5e',
  high: '#fb923c',
  medium: '#fbbf24',
  low: '#38bdf8',
  info: '#94a3b8'
};

const SEVERITY_ICONS = {
  critical: ShieldAlert,
  high: AlertTriangle,
  medium: Activity,
  low: Info,
  info: Info
};

export default function NotificationHistory({ role: defaultRole = 'user' }) {
  const { user } = useAuth();
  const role = user?.role || defaultRole;
  const isViewer = role === 'viewer';
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterStatus, setFilterStatus] = useState(''); // 'unread', 'read', ''
  const [search, setSearch] = useState('');

  const fetchNotifications = async () => {
    setLoading(true);
    try {
      let url = `http://127.0.0.1:5050/api/notifications?limit=200`;
      if (filterStatus) url += `&status=${filterStatus}`;
      
      const res = await fetch(url, {
        headers: getAuthHeaders()
      });
      const data = await res.json();
      setNotifications(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNotifications();
  }, [role, filterStatus]);

  const markRead = async (id) => {
    try {
      await fetch(`http://127.0.0.1:5050/api/notifications/${id}/read`, {
        method: 'PATCH',
        headers: getAuthHeaders()
      });
      fetchNotifications();
    } catch (e) {
      console.error(e);
    }
  };

  const markAllRead = async () => {
    try {
      await fetch(`http://127.0.0.1:5050/api/notifications/read-all`, {
        method: 'PATCH',
        headers: getAuthHeaders()
      });
      fetchNotifications();
    } catch (e) {
      console.error(e);
    }
  };

  const filtered = notifications.filter(n => {
    if (!search) return true;
    const lower = search.toLowerCase();
    return n.title.toLowerCase().includes(lower) || 
           n.message.toLowerCase().includes(lower) || 
           n.source.toLowerCase().includes(lower);
  });

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', paddingTop: '20px' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h2 style={{ margin: 0 }}>Notification History</h2>
        <div style={{ display: 'flex', gap: '12px' }}>
          {!isViewer && (
            <button onClick={markAllRead} className="btn-secondary" style={{ padding: '8px 16px', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', color: 'var(--text-main)', borderRadius: '6px', cursor: 'pointer' }}>
              Mark all read
            </button>
          )}
        </div>
      </div>

      <div style={{ 
        display: 'flex', 
        gap: '16px', 
        marginBottom: '24px', 
        backgroundColor: 'var(--bg-card)', 
        padding: '16px', 
        borderRadius: '8px', 
        border: '1px solid var(--border-subtle)' 
      }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <Search size={16} color="var(--text-faint)" style={{ position: 'absolute', left: '12px', top: '10px' }} />
          <input 
            type="text" 
            placeholder="Search alerts by title, message, or source..." 
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ width: '100%', padding: '8px 12px 8px 36px', backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-subtle)', borderRadius: '6px', color: 'var(--text-main)' }}
          />
        </div>
        
        <select 
          value={filterStatus} 
          onChange={e => setFilterStatus(e.target.value)}
          style={{ padding: '8px 16px', backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-subtle)', borderRadius: '6px', color: 'var(--text-main)' }}
        >
          <option value="">All Statuses</option>
          <option value="unread">Unread Only</option>
          <option value="read">Read Only</option>
        </select>
      </div>

      <div style={{ backgroundColor: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--border-subtle)', overflow: 'hidden' }}>
        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-faint)' }}>Loading notifications...</div>
        ) : filtered.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-faint)' }}>No notifications match your filters.</div>
        ) : (
          filtered.map(n => {
            const Icon = SEVERITY_ICONS[n.severity] || Info;
            const color = SEVERITY_COLORS[n.severity] || '#fff';
            const isUnread = n.notif_status === 'unread';

            return (
              <div 
                key={n.notification_id} 
                onClick={() => { if (!isViewer && isUnread) markRead(n.notification_id); }}
                style={{ 
                  padding: '20px', 
                  borderBottom: '1px solid var(--border-subtle)',
                  backgroundColor: isUnread ? 'rgba(255,255,255,0.02)' : 'transparent',
                  display: 'flex',
                  gap: '20px',
                  cursor: (!isViewer && isUnread) ? 'pointer' : 'default',
                  transition: 'background-color 0.2s'
                }}
              >
                <div style={{ marginTop: '2px', backgroundColor: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '50%', height: 'fit-content' }}>
                  <Icon size={20} color={color} />
                </div>
                
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                    <div style={{ fontSize: '15px', fontWeight: isUnread ? 700 : 500, color: 'var(--text-main)' }}>
                      {n.title}
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--text-faint)' }}>
                      {formatIST(n.created_at)}
                    </div>
                  </div>
                  
                  <div style={{ fontSize: '14px', color: 'var(--text-muted)', marginBottom: '12px', lineHeight: '1.5' }}>
                    {n.message}
                  </div>
                  
                  <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: 'var(--text-faint)' }}>
                    <span style={{ textTransform: 'capitalize', backgroundColor: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: '4px' }}>
                      Source: {n.source}
                    </span>
                    <span style={{ backgroundColor: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: '4px' }}>
                      Severity: <span style={{ color: color, textTransform: 'uppercase', fontWeight: 'bold' }}>{n.severity}</span>
                    </span>
                    {n.occurrence_count > 1 && (
                      <span style={{ backgroundColor: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: '4px', color: '#fbbf24' }}>
                        Occurrences: {n.occurrence_count}
                      </span>
                    )}
                  </div>
                </div>
                
                {isUnread && (
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--cyber-blue)' }} title="Unread"></div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

