import React from 'react';
import { 
  ShieldAlert, 
  LayoutDashboard, 
  Search, 
  Database, 
  FileText, 
  Settings, 
  Activity,
  Cpu,
  Users
} from 'lucide-react';

import { useAuth } from '../services/AuthContext';

export default function Sidebar({ currentTab, setTab, apiOnline }) {
  const { user } = useAuth();
  
  const navItems = [
    { id: 'dashboard', label: 'Security Overview', icon: LayoutDashboard },
    { id: 'analyze', label: 'Analyze Website', icon: Search, badge: 'Live', hideFor: ['viewer'] },
    { id: 'investigations', label: 'Investigation History', icon: Database },
    { id: 'reports', label: 'Threat Reports', icon: FileText },
    { id: 'settings', label: 'System & Agents', icon: Settings, hideFor: ['viewer', 'analyst'] },
    { id: 'users', label: 'User Management', icon: Users, hideFor: ['viewer', 'analyst'] },
  ].filter(item => !item.hideFor?.includes(user?.role));

  return (
    <aside style={{
      width: '260px',
      backgroundColor: 'var(--bg-darker)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      userSelect: 'none',
      zIndex: 10
    }}>
      {/* Brand Header */}
      <div>
        <div style={{
          padding: '24px 20px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          borderBottom: '1px solid var(--border-subtle)'
        }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '8px',
            background: 'var(--brand-gradient, linear-gradient(135deg, #0284c7 0%, #0369a1 100%))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: 'var(--brand-glow, 0 0 16px rgba(2, 132, 199, 0.4))'
          }}>
            <ShieldAlert size={22} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '16px', letterSpacing: '0.5px', color: '#f8fafc' }}>
              PHISHDEC
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 500, letterSpacing: '0.3px' }}>
              Multi-Agent SOC Platform
            </div>
          </div>
        </div>

        {/* Navigation Items */}
        <nav style={{ padding: '16px 12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  if (item.href) {
                    window.location.href = item.href;
                  } else {
                    setTab(item.id);
                  }
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '10px 14px',
                  borderRadius: '6px',
                  border: 'none',
                  background: active ? 'var(--cyber-glow, rgba(56, 189, 248, 0.12))' : 'transparent',
                  color: active ? 'var(--cyber-blue)' : 'var(--text-muted)',
                  fontSize: '13px',
                  fontWeight: active ? 600 : 500,
                  cursor: 'pointer',
                  textAlign: 'left',
                  width: '100%',
                  transition: 'all 0.15s ease',
                  borderLeft: active ? '3px solid var(--cyber-blue)' : '3px solid transparent'
                }}
                onMouseEnter={(e) => {
                  if (!active) {
                    e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.04)';
                    e.currentTarget.style.color = '#f8fafc';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!active) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.color = 'var(--text-muted)';
                  }
                }}
              >
                <Icon size={18} />
                <span style={{ flex: 1 }}>{item.label}</span>
                {item.badge && (
                  <span style={{
                    fontSize: '10px',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    backgroundColor: 'rgba(56, 189, 248, 0.2)',
                    color: 'var(--cyber-blue)',
                    fontWeight: 700
                  }}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Info Box */}
      <div style={{
        padding: '16px 18px',
        borderTop: '1px solid var(--border-subtle)',
        backgroundColor: 'rgba(15, 23, 42, 0.4)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: 'var(--text-muted)' }}>
            <Cpu size={14} color="var(--cyber-blue)" />
            <span>Active Agents</span>
          </div>
          <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--cyber-blue)' }}>6 / 6</span>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: 'var(--text-muted)' }}>
          <span className="pulsing-dot" style={{ backgroundColor: apiOnline ? '#10b981' : '#f43f5e' }} />
          <span>{apiOnline ? 'Backend API Connected' : 'Connecting to :5050...'}</span>
        </div>
      </div>
    </aside>
  );
}
