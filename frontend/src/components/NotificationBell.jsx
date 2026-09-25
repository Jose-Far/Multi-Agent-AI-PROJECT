import React, { useState, useEffect, useRef } from 'react';
import { Bell, ShieldAlert, Activity, AlertTriangle, Info, CheckCircle, Search, Filter, ShieldCheck, X } from 'lucide-react';
import { getAuthHeaders } from '../services/api';
import { useAuth } from '../services/AuthContext';
import { formatIST } from '../utils/dateUtils';


const SEVERITY_ICONS = {
  critical: ShieldAlert,
  high: AlertTriangle,
  medium: Activity,
  low: Info,
  success: CheckCircle
};

const SEVERITY_COLORS = {
  critical: '#e11d48', // Red
  high: '#ea580c',     // Orange
  medium: '#fbbf24',   // Yellow
  low: '#38bdf8',      // Blue
  success: '#10b981'   // Green
};

const CATEGORIES = ['All', 'Unread', 'Security', 'Analysis', 'Agents', 'Fusion', 'System', 'Admin'];

export default function NotificationBell({ onNavigate }) {
  const { role } = useAuth();
  const isViewer = role === 'viewer';
  const [isOpen, setIsOpen] = useState(false);
  const [count, setCount] = useState(0);
  const [notifications, setNotifications] = useState([]);
  const [activeCategory, setActiveCategory] = useState('All');
  const dropdownRef = useRef(null);

  const fetchUnreadCount = async () => {
    try {
      const res = await fetch(`http://127.0.0.1:5050/api/notifications/unread`, { headers: getAuthHeaders() });
      const data = await res.json();
      if (data.count !== undefined) setCount(data.count);
    } catch (e) { console.error(e); }
  };

  const fetchNotifications = async () => {
    try {
      const res = await fetch(`http://127.0.0.1:5050/api/notifications?limit=50`, { headers: getAuthHeaders() });
      const data = await res.json();
      if (Array.isArray(data)) setNotifications(data);
    } catch (e) { console.error(e); }
  };

  useEffect(() => {
    fetchUnreadCount();
    const interval = setInterval(fetchUnreadCount, 30000);
    return () => clearInterval(interval);
  }, [role]);

  useEffect(() => { if (isOpen) fetchNotifications(); }, [isOpen, role]);

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) setIsOpen(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const markAllRead = async () => {
    try {
      await fetch(`http://127.0.0.1:5050/api/notifications/read-all`, { method: 'PATCH', headers: getAuthHeaders() });
      setCount(0); fetchNotifications();
    } catch (e) { console.error(e); }
  };

  const markRead = async (id) => {
    try {
      await fetch(`http://127.0.0.1:5050/api/notifications/${id}/read`, { method: 'PATCH', headers: getAuthHeaders() });
      fetchUnreadCount(); fetchNotifications();
    } catch (e) { console.error(e); }
  };

  const filteredNotifications = notifications.filter(n => {
    if (activeCategory === 'All') return true;
    if (activeCategory === 'Unread') return n.notif_status === 'unread';
    // Naive category matching based on alert type or source
    const t = (n.type || '').toLowerCase();
    const s = (n.source || '').toLowerCase();
    const txt = (n.title + ' ' + n.message).toLowerCase();
    if (activeCategory === 'Security') return t.includes('security') || s.includes('threat') || txt.includes('phishing');
    if (activeCategory === 'Analysis') return t.includes('analysis') || txt.includes('verdict');
    if (activeCategory === 'Agents') return s.includes('agent') || txt.includes('agent');
    if (activeCategory === 'Fusion') return s.includes('fusion') || txt.includes('consensus');
    if (activeCategory === 'System') return t.includes('system') || s.includes('health');
    if (activeCategory === 'Admin') return t.includes('admin') || txt.includes('user');
    return true;
  });

  return (
    <div className="notification-bell-container" ref={dropdownRef} style={{ position: 'relative' }}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        style={{
          background: 'transparent', border: 'none', color: 'var(--text-main)', cursor: 'pointer',
          position: 'relative', padding: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center',
          borderRadius: '50%', backgroundColor: isOpen ? 'rgba(255,255,255,0.1)' : 'transparent', transition: 'all 0.2s'
        }}
      >
        <Bell size={20} />
        {count > 0 && (
          <span style={{
            position: 'absolute', top: '2px', right: '2px', backgroundColor: '#e11d48',
            color: 'white', fontSize: '10px', fontWeight: 'bold', borderRadius: '10px',
            padding: '2px 6px', boxShadow: '0 0 0 2px var(--bg-dark)'
          }}>
            {count}
          </span>
        )}
      </button>

      {isOpen && (
        <div style={{
          position: 'absolute', top: '100%', right: '0', marginTop: '12px',
          width: '450px', backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
          borderRadius: '12px', boxShadow: '0 20px 40px rgba(0,0,0,0.6)', zIndex: 1000, overflow: 'hidden',
          display: 'flex', flexDirection: 'column', animation: 'slideDown 0.2s ease-out'
        }}>
          {/* Header */}
          <div style={{ padding: '16px', borderBottom: '1px solid var(--border-subtle)', backgroundColor: 'var(--bg-card-elevated)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0, fontSize: '16px', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <ShieldCheck size={18} color="var(--cyber-blue)" /> Security Notification Center
            </h3>
            <div style={{ display: 'flex', gap: '12px' }}>
              {!isViewer && count > 0 && (
                <button onClick={markAllRead} style={{ background: 'transparent', border: 'none', color: 'var(--cyber-blue)', fontSize: '12px', cursor: 'pointer', fontWeight: 600 }}>
                  Mark all read
                </button>
              )}
              <button onClick={() => setIsOpen(false)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}><X size={18}/></button>
            </div>
          </div>

          {/* Categories / Filters */}
          <div style={{ padding: '8px 16px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', gap: '6px', overflowX: 'auto', backgroundColor: 'rgba(0,0,0,0.2)' }} className="scrollbar-hide">
            {CATEGORIES.map(cat => (
              <button 
                key={cat} onClick={() => setActiveCategory(cat)}
                style={{
                  padding: '4px 12px', borderRadius: '20px', fontSize: '11px', fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap',
                  backgroundColor: activeCategory === cat ? 'var(--cyber-blue)' : 'transparent',
                  color: activeCategory === cat ? '#000' : 'var(--text-muted)',
                  border: `1px solid ${activeCategory === cat ? 'var(--cyber-blue)' : 'var(--border-subtle)'}`,
                  transition: 'all 0.2s'
                }}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Notifications List */}
          <div style={{ maxHeight: '450px', overflowY: 'auto' }}>
            {filteredNotifications.length === 0 ? (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-faint)' }}>
                <Bell size={32} style={{ opacity: 0.3, marginBottom: '10px' }} />
                <div>No {activeCategory.toLowerCase()} notifications</div>
              </div>
            ) : (
              filteredNotifications.map(n => {
                // Infer Icon & Color mapping dynamically since severity might be mixed
                let severityKey = n.severity || 'info';
                if (n.title.toLowerCase().includes('phish') || n.title.toLowerCase().includes('critical')) severityKey = 'critical';
                else if (n.title.toLowerCase().includes('conflict')) severityKey = 'high';
                else if (n.title.toLowerCase().includes('success') || n.title.toLowerCase().includes('recovered')) severityKey = 'success';
                
                const Icon = SEVERITY_ICONS[severityKey] || Info;
                const color = SEVERITY_COLORS[severityKey] || '#fff';
                const isUnread = n.notif_status === 'unread';

                // Try to extract rich data (like Risk Score / Verdict) from message if it's formatted a certain way, or just show raw
                let displayMsg = n.message;
                let verdict = null;
                let riskScore = null;
                let consensus = null;
                
                // Extremely basic parsing simulation if the message contains these keywords
                if (displayMsg.includes('Verdict:')) {
                  const parts = displayMsg.split('Verdict:');
                  displayMsg = parts[0];
                  verdict = parts[1].split('\\n')[0].trim();
                }

                return (
                  <div 
                    key={n.notification_id} 
                    onClick={() => { if (!isViewer && isUnread) markRead(n.notification_id); }}
                    style={{ 
                      padding: '16px', borderBottom: '1px solid var(--border-subtle)',
                      backgroundColor: isUnread ? 'rgba(56, 189, 248, 0.03)' : 'transparent',
                      cursor: (!isViewer && isUnread) ? 'pointer' : 'default',
                      display: 'flex', gap: '14px', transition: 'background-color 0.2s'
                    }}
                    onMouseEnter={e => { if (!isViewer && isUnread) e.currentTarget.style.backgroundColor = 'rgba(56, 189, 248, 0.08)' }}
                    onMouseLeave={e => { if (!isViewer && isUnread) e.currentTarget.style.backgroundColor = 'rgba(56, 189, 248, 0.03)' }}
                  >
                    <div style={{ 
                      width: '32px', height: '32px', borderRadius: '8px', 
                      backgroundColor: `${color}15`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 
                    }}>
                      <Icon size={16} color={color} />
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px' }}>
                        <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          {isUnread && <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--cyber-blue)', display: 'inline-block' }} />}
                          {n.title}
                        </div>
                        <span style={{ fontSize: '11px', color: 'var(--text-faint)', whiteSpace: 'nowrap' }}>
                          {formatIST(n.created_at).split(' ')[1]}
                        </span>
                      </div>
                      
                      {n.reference_id && (
                        <div style={{ fontSize: '11px', color: 'var(--cyber-blue)', fontFamily: 'monospace', marginBottom: '6px' }}>
                          REF: {n.reference_id}
                        </div>
                      )}
                      
                      <div style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: '1.5', marginBottom: (verdict || n.occurrence_count > 1) ? '10px' : '0' }}>
                        {displayMsg}
                      </div>

                      {/* Mocked Rich Payload display for SOC feel */}
                      {verdict && (
                        <div style={{ display: 'flex', gap: '10px', marginTop: '8px', fontSize: '12px' }}>
                          <span style={{ padding: '2px 6px', backgroundColor: 'rgba(244,63,94,0.1)', color: '#fb7185', borderRadius: '4px', border: '1px solid rgba(244,63,94,0.2)' }}>
                            Verdict: {verdict}
                          </span>
                        </div>
                      )}

                    </div>
                  </div>
                );
              })
            )}
          </div>

          <div style={{ padding: '12px', textAlign: 'center', borderTop: '1px solid var(--border-subtle)', backgroundColor: 'var(--bg-card-elevated)' }}>
            <button 
              onClick={() => { 
                if (onNavigate) { onNavigate('notifications'); setIsOpen(false); } 
                else { window.location.href = role === 'admin' ? '/admin/notifications' : '/notifications'; }
              }}
              style={{ 
                background: 'var(--bg-dark)', border: '1px solid var(--border-subtle)', color: 'var(--text-main)', 
                fontSize: '12px', fontWeight: 600, cursor: 'pointer', padding: '8px 16px', borderRadius: '6px', width: '100%',
                transition: 'all 0.2s'
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--cyber-blue)'; e.currentTarget.style.color = 'var(--cyber-blue)'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border-subtle)'; e.currentTarget.style.color = 'var(--text-main)'; }}
            >
              View Full Security Log &rarr;
            </button>
          </div>
        </div>
      )}
      <style>{`
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
      `}</style>
    </div>
  );
}


