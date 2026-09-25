import React from 'react';
import { 
  ShieldAlert, 
  LayoutDashboard, 
  Users, 
  Bot, 
  Brain,
  BarChart3,
  Activity,
  HeartPulse,
  ScrollText,
  Bell,
  Settings,
  ArrowLeft
} from 'lucide-react';

import { useAuth } from '../../services/AuthContext';

export default function AdminSidebar({ currentTab, setTab }) {
  const { user } = useAuth();
  const isSuperAdmin = user?.role === 'super_admin';

  const menuItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'agents', label: 'Agents', icon: Bot, hideForAdmin: true },
    { id: 'models', label: 'Models', icon: Brain, hideForAdmin: true },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'monitoring', label: 'Monitoring', icon: Activity },
    { id: 'health', label: 'Health', icon: HeartPulse },
    { id: 'logs', label: 'Logs', icon: ScrollText, hideForAdmin: true },
    { id: 'alerts', label: 'Alerts', icon: Bell },
    { id: 'settings', label: 'Settings', icon: Settings, hideForAdmin: true }
  ].filter(item => isSuperAdmin || !item.hideForAdmin);

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
            background: 'var(--brand-gradient, linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: 'var(--brand-glow, 0 0 16px rgba(167, 139, 250, 0.4))'
          }}>
            <ShieldAlert size={22} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '16px', letterSpacing: '0.5px', color: '#f8fafc' }}>
              PHISHDEC ADMIN
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 500, letterSpacing: '0.3px' }}>
              System Management
            </div>
          </div>
        </div>

        {/* Navigation Items */}
        <nav style={{ padding: '16px 12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {menuItems.map((item) => {
            const Icon = item.icon;
            const active = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setTab(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '10px 14px',
                  borderRadius: '6px',
                  border: 'none',
                  background: active ? 'var(--cyber-glow, rgba(167, 139, 250, 0.12))' : 'transparent',
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
        <button
          onClick={() => window.location.href = '/'}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            padding: '10px 14px',
            borderRadius: '6px',
            border: '1px solid var(--border-subtle)',
            background: 'transparent',
            color: 'var(--text-muted)',
            fontSize: '12px',
            fontWeight: 600,
            cursor: 'pointer',
            width: '100%',
            transition: 'all 0.15s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
            e.currentTarget.style.color = '#f8fafc';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = 'var(--text-muted)';
          }}
        >
          <ArrowLeft size={16} />
          Return to Analyst View
        </button>
      </div>
    </aside>
  );
}
