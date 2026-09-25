
import React, { useState } from 'react';
import { Settings, Shield, Server, Key, Save, Eye, EyeOff, CheckCircle, RefreshCcw, Bell } from 'lucide-react';

export default function AdminSettingsPage() {
  const [showKey, setShowKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  
  // Simulated configuration state
  const [config, setConfig] = useState({
    virustotal_key: 'a8f5f167f44f4964e6c998dee827110c',
    db_retention: '90',
    log_level: 'INFO',
    fusion_consensus: 2,
    enable_notifications: true
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSave = () => {
    setIsSaving(true);
    setTimeout(() => {
      setIsSaving(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    }, 1000);
  };

  const renderMaskedKey = (key) => {
    return showKey ? key : '�'.repeat(24);
  };

  const cardStyle = {
    backgroundColor: 'var(--bg-card)', 
    border: '1px solid var(--border-subtle)', 
    borderRadius: '12px', 
    padding: '28px',
    boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
    position: 'relative',
    overflow: 'hidden'
  };

  const inputStyle = {
    width: '100%', 
    backgroundColor: 'rgba(6, 9, 15, 0.7)', 
    border: '1px solid var(--border-subtle)', 
    borderRadius: '8px', 
    padding: '12px 16px', 
    color: '#fff', 
    outline: 'none', 
    boxSizing: 'border-box',
    fontSize: '14px',
    transition: 'all 0.2s ease'
  };

  return (
    <div className="admin-settings" style={{ paddingBottom: '60px' }}>
      
      {/* Header section */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{ backgroundColor: 'rgba(56, 189, 248, 0.1)', padding: '10px', borderRadius: '10px' }}>
            <Settings size={28} style={{ color: 'var(--cyber-blue)' }} />
          </div>
          <div>
            <h2 style={{ margin: 0, color: '#fff', fontSize: '24px' }}>Global Settings</h2>
            <div style={{ fontSize: '14px', color: 'var(--text-muted)', marginTop: '4px' }}>Configure system-wide parameters and API integrations</div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '30px', maxWidth: '850px' }}>
        
        {/* Threat Intelligence Settings */}
        <div style={cardStyle}>
          <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '4px', background: 'linear-gradient(90deg, #f43f5e, #fb923c)' }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <Shield size={22} style={{ color: '#f43f5e' }} />
            <h3 style={{ margin: 0, fontSize: '18px', color: '#fff' }}>Threat Intelligence Integrations</h3>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginBottom: '24px', marginTop: 0 }}>Configure third-party API keys required by the specialized agents.</p>
          
          <div>
            <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-main)', fontSize: '14px', fontWeight: 600 }}>VirusTotal API v3 Key</label>
            <div style={{ display: 'flex', gap: '12px' }}>
              <div style={{ 
                flex: 1, backgroundColor: 'rgba(6, 9, 15, 0.7)', border: '1px solid var(--border-subtle)', 
                borderRadius: '8px', padding: '12px 16px', color: '#fff', 
                fontFamily: 'monospace', letterSpacing: showKey ? 'normal' : '2px',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <span style={{ opacity: showKey ? 1 : 0.7 }}>{renderMaskedKey(config.virustotal_key)}</span>
                <button 
                  type="button"
                  onClick={() => setShowKey(!showKey)}
                  style={{ background: 'none', border: 'none', color: 'var(--cyber-blue)', cursor: 'pointer', padding: 0, display: 'flex' }}
                >
                  {showKey ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
              <button 
                type="button"
                style={{ 
                  backgroundColor: 'transparent', 
                  border: '1px solid var(--cyber-blue)', 
                  color: 'var(--cyber-blue)', 
                  padding: '0 24px', 
                  borderRadius: '8px', 
                  cursor: 'pointer',
                  fontWeight: 600,
                  transition: 'all 0.2s',
                  fontSize: '14px'
                }}
                onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(56, 189, 248, 0.1)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; }}
              >
                Update Key
              </button>
            </div>
            <div style={{ color: 'var(--text-faint)', fontSize: '12px', marginTop: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <CheckCircle size={12} color="#10b981" /> Connection verified successfully 2 hours ago
            </div>
          </div>
        </div>

        {/* Database & Logging */}
        <div style={cardStyle}>
          <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '4px', background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)' }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <Server size={22} style={{ color: '#3b82f6' }} />
            <h3 style={{ margin: 0, fontSize: '18px', color: '#fff' }}>Database & Logging</h3>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginBottom: '24px', marginTop: 0 }}>Manage data retention policies and system log verbosity.</p>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-main)', fontSize: '14px', fontWeight: 600 }}>System Log Level</label>
              <select 
                name="log_level"
                value={config.log_level}
                onChange={handleChange}
                style={{ ...inputStyle, cursor: 'pointer', appearance: 'none', backgroundImage: 'url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2394a3b8%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E")', backgroundRepeat: 'no-repeat', backgroundPosition: 'right 16px top 50%', backgroundSize: '12px auto' }}
              >
                <option value="DEBUG">DEBUG (Verbose)</option>
                <option value="INFO">INFO (Standard)</option>
                <option value="WARNING">WARNING (Issues Only)</option>
                <option value="ERROR">ERROR (Critical Only)</option>
              </select>
            </div>
            
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-main)', fontSize: '14px', fontWeight: 600 }}>Data Retention Policy (Days)</label>
              <input 
                type="number" 
                name="db_retention"
                value={config.db_retention} 
                onChange={handleChange}
                style={inputStyle}
                onFocus={(e) => e.target.style.borderColor = 'var(--cyber-blue)'}
                onBlur={(e) => e.target.style.borderColor = 'var(--border-subtle)'}
              />
            </div>
          </div>
        </div>
        
        {/* Fusion Engine Configuration */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
          <div style={cardStyle}>
            <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '4px', background: 'linear-gradient(90deg, #10b981, #34d399)' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <Key size={22} style={{ color: '#10b981' }} />
              <h3 style={{ margin: 0, fontSize: '18px', color: '#fff' }}>Fusion Engine</h3>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginBottom: '24px', marginTop: 0 }}>Consensus mechanism logic.</p>
            
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-main)', fontSize: '14px', fontWeight: 600 }}>Minimum Consensus Threshold</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                <input 
                  type="number" 
                  name="fusion_consensus"
                  min="1" 
                  max="6" 
                  value={config.fusion_consensus} 
                  onChange={handleChange}
                  style={{ ...inputStyle, width: '100px', textAlign: 'center', fontSize: '16px', fontWeight: 'bold' }}
                  onFocus={(e) => e.target.style.borderColor = 'var(--cyber-blue)'}
                  onBlur={(e) => e.target.style.borderColor = 'var(--border-subtle)'}
                />
                <span style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Agents out of 6</span>
              </div>
              <div style={{ color: '#fb923c', fontSize: '12px', marginTop: '12px', display: 'flex', alignItems: 'flex-start', gap: '6px', backgroundColor: 'rgba(251, 146, 60, 0.05)', padding: '10px', borderRadius: '6px' }}>
                <RefreshCcw size={14} style={{ marginTop: '2px', flexShrink: 0 }} />
                <span>Requires restarting the Orchestrator daemon via the CLI for changes to take effect.</span>
              </div>
            </div>
          </div>

          <div style={cardStyle}>
            <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '4px', background: 'linear-gradient(90deg, #a855f7, #d946ef)' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <Bell size={22} style={{ color: '#a855f7' }} />
              <h3 style={{ margin: 0, fontSize: '18px', color: '#fff' }}>System Alerts</h3>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginBottom: '24px', marginTop: 0 }}>Push notifications and automated alerts.</p>
            
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', backgroundColor: 'rgba(6, 9, 15, 0.4)', border: '1px solid var(--border-subtle)', borderRadius: '8px' }}>
              <div>
                <div style={{ color: '#fff', fontSize: '14px', fontWeight: 600 }}>Enable UI Notifications</div>
                <div style={{ color: 'var(--text-faint)', fontSize: '12px', marginTop: '2px' }}>Show toasts for background events</div>
              </div>
              <label style={{ position: 'relative', display: 'inline-block', width: '44px', height: '24px' }}>
                <input 
                  type="checkbox" 
                  name="enable_notifications"
                  checked={config.enable_notifications}
                  onChange={handleChange}
                  style={{ opacity: 0, width: 0, height: 0 }} 
                />
                <span style={{
                  position: 'absolute', cursor: 'pointer', top: 0, left: 0, right: 0, bottom: 0,
                  backgroundColor: config.enable_notifications ? 'var(--cyber-blue)' : 'var(--bg-darker)',
                  transition: '.4s', borderRadius: '24px',
                  border: config.enable_notifications ? 'none' : '1px solid var(--border-subtle)'
                }}>
                  <span style={{
                    position: 'absolute', height: '18px', width: '18px', left: config.enable_notifications ? '24px' : '3px', bottom: config.enable_notifications ? '3px' : '2px',
                    backgroundColor: config.enable_notifications ? '#000' : 'var(--text-muted)',
                    transition: '.4s', borderRadius: '50%'
                  }} />
                </span>
              </label>
            </div>
          </div>
        </div>
        
        {/* Save Button */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '10px' }}>
           <button 
             onClick={handleSave}
             disabled={isSaving}
             style={{ 
               display: 'flex', alignItems: 'center', gap: '8px', 
               backgroundColor: saved ? '#10b981' : 'var(--cyber-blue)', 
               color: '#000', 
               border: 'none', padding: '12px 32px', borderRadius: '8px', 
               fontWeight: 700, fontSize: '15px', cursor: isSaving ? 'not-allowed' : 'pointer',
               transition: 'all 0.3s',
               boxShadow: saved ? '0 0 15px rgba(16, 185, 129, 0.4)' : '0 4px 14px rgba(56, 189, 248, 0.3)'
             }}
           >
             {isSaving ? (
               <><RefreshCcw size={18} className="spin" /> Saving...</>
             ) : saved ? (
               <><CheckCircle size={18} /> Configuration Saved</>
             ) : (
               <><Save size={18} /> Save Global Configuration</>
             )}
           </button>
        </div>

      </div>
      <style>{`
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}


