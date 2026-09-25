import React, { useState, useEffect } from 'react';
import { formatIST } from '../utils/dateUtils';
import { RefreshCw, Clock, User, LogOut, Shield } from 'lucide-react';
import NotificationBell from './NotificationBell';
import { useAuth } from '../services/AuthContext';

const TAB_TITLES = {
  dashboard: 'Security Overview',
  analyze: 'Multi-Agent Website Analysis',
  investigations: 'Investigation History',
  reports: 'Threat Intelligence Reports',
  settings: 'System Diagnostics & Agents',
  users: 'User Management',
  notifications: 'Notifications Center',
};

const ROLE_COLORS = {
  admin: { bg: 'var(--cyber-glow)', border: 'var(--cyber-blue)', text: 'var(--cyber-blue)' },
  analyst: { bg: 'var(--cyber-glow)', border: 'var(--cyber-blue)', text: 'var(--cyber-blue)' },
  viewer: { bg: 'rgba(56, 189, 248, 0.15)', border: '#38bdf8', text: '#38bdf8' }, // Viewer keeps default blue
};

export default function Header({ currentTab, apiOnline, onRefreshHealth, onNavigate }) {
  const { user, logout } = useAuth();
  const getISTTime = () => {
    return formatIST(new Date()).split(' ')[1] + ' IST';
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
    if (onRefreshHealth) {
      setRefreshing(true);
      await onRefreshHealth();
      setTimeout(() => setRefreshing(false), 500);
    }
  };

  const roleStyle = ROLE_COLORS[user?.role] || ROLE_COLORS.analyst;

  return (
    <header style={{
      height: '68px',
      padding: '0 32px',
      borderBottom: '1px solid var(--border-subtle)',
      backgroundColor: 'rgba(6, 9, 15, 0.75)',
      backdropFilter: 'blur(12px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'sticky',
      top: 0,
      zIndex: 20
    }}>
      {/* Title & Breadcrumb */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: '2px' }}>
          <span>PhishDec SOC</span>
          <span>/</span>
          <span style={{ color: 'var(--cyber-blue)' }}>{TAB_TITLES[currentTab] || currentTab}</span>
        </div>
        <h1 style={{ fontSize: '18px', fontWeight: 700, color: '#ffffff', letterSpacing: '-0.3px', margin: 0 }}>
          {TAB_TITLES[currentTab] || currentTab}
        </h1>
      </div>

      {/* Status & Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        
        {/* Phase 8: Notification Bell */}
        <NotificationBell role={user?.role || 'user'} onNavigate={onNavigate} />

        <div style={{ width: '1px', height: '24px', backgroundColor: 'var(--border-subtle)' }} />

        {/* UTC Clock */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '12px',
          color: 'var(--text-muted)',
          backgroundColor: 'rgba(15, 23, 42, 0.6)',
          padding: '6px 12px',
          borderRadius: '6px',
          border: '1px solid var(--border-subtle)',
          fontFamily: 'ui-monospace, monospace'
        }}>
          <Clock size={13} color="var(--text-faint)" />
          <span>{time}</span>
        </div>

        {/* API Health Pill */}
        <div 
          onClick={handleRefresh}
          title="Click to re-ping backend (:5050)"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 14px',
            borderRadius: '20px',
            backgroundColor: apiOnline ? 'rgba(16, 185, 129, 0.12)' : 'rgba(244, 63, 94, 0.12)',
            border: `1px solid ${apiOnline ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          <span className="pulsing-dot" style={{ backgroundColor: apiOnline ? '#10b981' : '#f43f5e' }} />
          <span style={{
            fontSize: '12px',
            fontWeight: 600,
            color: apiOnline ? '#34d399' : '#fb7185',
            letterSpacing: '0.2px'
          }}>
            {apiOnline ? 'API :5050 Online' : 'API :5050 Offline'}
          </span>
          <RefreshCw 
            size={12} 
            color={apiOnline ? '#34d399' : '#fb7185'} 
            style={{ 
              transition: 'transform 0.5s ease',
              transform: refreshing ? 'rotate(360deg)' : 'none'
            }} 
          />
        </div>

        <div style={{ width: '1px', height: '24px', backgroundColor: 'var(--border-subtle)' }} />

        {/* User Identity & Logout (Day 25) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '50%',
              backgroundColor: 'rgba(56, 189, 248, 0.1)',
              border: `1px solid ${roleStyle.border}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <User size={16} color={roleStyle.text} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '13px', fontWeight: 600, color: '#f8fafc', lineHeight: '1.2' }}>
                {user?.username || 'User'}
              </span>
              <span style={{
                fontSize: '10px',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.4px',
                color: roleStyle.text
              }}>
                {user?.role || 'Guest'}
              </span>
            </div>
          </div>

          <button
            onClick={logout}
            title="Log Out of PhishDec"
            style={{
              background: 'transparent',
              border: '1px solid rgba(244, 63, 94, 0.2)',
              borderRadius: '6px',
              color: '#fb7185',
              padding: '6px 8px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              cursor: 'pointer',
              fontSize: '11px',
              fontWeight: 600,
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(244, 63, 94, 0.15)';
              e.currentTarget.style.borderColor = '#f43f5e';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.borderColor = 'rgba(244, 63, 94, 0.2)';
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

