import React, { useState, useEffect } from 'react';
import { RefreshCw, Clock, ShieldAlert, LogOut, ExternalLink } from 'lucide-react';
import NotificationBell from '../NotificationBell';
import { useAuth } from '../../services/AuthContext';

const TAB_TITLES = {
  overview: 'System Overview',
  users: 'User Management',
  agents: 'AI Agent Status',
  models: 'ML Model Registry',
  analytics: 'Operational Analytics',
  monitoring: 'System Monitoring',
  health: 'System Health Checks',
  logs: 'System Logs',
  alerts: 'Alert Configuration',
  settings: 'Global Settings',
  notifications: 'Notifications Center',
};

export default function AdminHeader({ currentTab, apiOnline, onRefresh, onNavigate }) {
  const { user, logout } = useAuth();
  const getISTTime = () => {
    return new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false }) + ' IST';
  };

  const [time, setTime] = useState(getISTTime());
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(getISTTime());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const handleRefresh = async () => {
    if (onRefresh) {
      setRefreshing(true);
      await onRefresh();
      setTimeout(() => setRefreshing(false), 500);
    }
  };

  return (
    <header className="header" style={{
      height: '68px',
      padding: '0 32px',
      borderBottom: '1px solid var(--border-subtle)',
      backgroundColor: 'rgba(10, 8, 18, 0.75)',
      backdropFilter: 'blur(12px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'sticky',
      top: 0,
      zIndex: 20
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
        <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>{TAB_TITLES[currentTab] || currentTab}</h2>
        {apiOnline ? (
          <span style={{ fontSize: '11px', fontWeight: 600, padding: '4px 10px', borderRadius: '12px', backgroundColor: 'rgba(16, 185, 129, 0.1)', color: '#34d399', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
            <span className="pulsing-dot" style={{ backgroundColor: '#10b981', display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', marginRight: '6px', marginBottom: '1px' }}></span>
            SYSTEM OPERATIONAL
          </span>
        ) : (
          <span style={{ fontSize: '11px', fontWeight: 600, padding: '4px 10px', borderRadius: '12px', backgroundColor: 'rgba(244, 63, 94, 0.1)', color: '#fb7185', border: '1px solid rgba(244, 63, 94, 0.2)' }}>
            <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#f43f5e', marginRight: '6px', marginBottom: '1px' }}></span>
            SYSTEM DEGRADED
          </span>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        {/* Switch to Analyst SOC view */}
        <button 
          onClick={() => { window.location.href = '/'; }}
          title="Switch to Analyst SOC Dashboard"
          style={{
            background: 'rgba(56, 189, 248, 0.1)',
            border: '1px solid rgba(56, 189, 248, 0.25)',
            borderRadius: '6px',
            color: '#38bdf8',
            padding: '6px 12px',
            fontSize: '12px',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <span>Analyst SOC</span>
          <ExternalLink size={12} />
        </button>

        <button 
          onClick={handleRefresh}
          style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}
        >
          <RefreshCw size={14} style={{ transition: 'transform 0.5s', transform: refreshing ? 'rotate(360deg)' : 'none' }} />
          Refresh
        </button>

        {/* Phase 8: Notification Bell */}
        <NotificationBell role="admin" onNavigate={onNavigate} />

        <div style={{ width: '1px', height: '24px', backgroundColor: 'var(--border-subtle)' }} />
        
        {/* Identity & Logout */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '32px', height: '32px', borderRadius: '50%', backgroundColor: 'var(--cyber-glow)', border: '1px solid var(--cyber-blue)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ShieldAlert size={16} color="var(--cyber-blue)" />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main)', lineHeight: '1.2' }}>{user?.username || 'Admin'}</span>
            <span style={{ fontSize: '11px', color: 'var(--cyber-blue)', fontWeight: 600, textTransform: 'uppercase' }}>Administrator</span>
          </div>
          
          <button
            onClick={logout}
            title="Log Out"
            style={{
              background: 'transparent',
              border: '1px solid var(--cyber-glow)',
              borderRadius: '6px',
              color: 'var(--cyber-blue)',
              padding: '6px 8px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              cursor: 'pointer',
              fontSize: '11px',
              fontWeight: 600,
              marginLeft: '4px'
            }}
          >
            <LogOut size={13} />
            <span>Logout</span>
          </button>
        </div>
      </div>
    </header>
  );
}
