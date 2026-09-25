import React, { useState, useEffect } from 'react';
import { getAuthHeaders } from '../../services/api';
import { BarChart3, PieChart, Activity, AlertTriangle, ShieldCheck, Cpu, Download, Database, Server, Zap, CheckCircle2, XCircle } from 'lucide-react';

export default function AdminAnalyticsPage() {
  const [period, setPeriod] = useState(30);
  const [filterVerdict, setFilterVerdict] = useState('');
  const [filterRisk, setFilterRisk] = useState('');
  
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAnalytics();
  }, [period, filterVerdict, filterRisk]);

  const fetchAnalytics = async () => {
    setLoading(true);
    setError(null);
    try {
      let url = `http://127.0.0.1:5050/api/admin/analytics?period=${period}`;
      if (filterVerdict) url += `&verdict=${filterVerdict}`;
      if (filterRisk) url += `&risk=${filterRisk}`;
      
      const res = await fetch(url, {
        headers: getAuthHeaders()
      });
      if (res.status === 403) throw new Error("Unauthorized access. Admin privileges required.");
      if (!res.ok) throw new Error("Failed to fetch analytics");
      
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const handleExport = () => {
    window.open(`http://127.0.0.1:5050/api/admin/analytics/export?period=${period}`);
  };

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

  if (error) {
    return (
      <div style={{ ...cardStyle, borderColor: '#f43f5e', textAlign: 'center', marginTop: '40px' }}>
        <XCircle size={48} color="#f43f5e" style={{ margin: '0 auto 15px' }} />
        <h3 style={{ color: '#fff', marginBottom: '10px' }}>Unable to load analytics</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '20px' }}>{error}</p>
        <button onClick={fetchAnalytics} className="btn-primary" style={{ padding: '8px 24px' }}>Retry</button>
      </div>
    );
  }

  const isEmpty = !data || data.overview.total_analyses === 0;

  return (
    <div className="admin-analytics" style={{ paddingBottom: '40px' }}>
      {/* HEADER SECTION */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <BarChart3 size={28} style={{ color: 'var(--cyber-blue)' }} />
          <h2 style={{ margin: 0, color: '#fff', fontSize: '24px' }}>Operational Analytics</h2>
        </div>
        
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <select 
            value={filterVerdict} 
            onChange={e => setFilterVerdict(e.target.value)} 
            style={{ 
              backgroundColor: 'rgba(6, 9, 15, 0.7)', 
              border: '1px solid var(--border-subtle)', 
              borderRadius: '8px', 
              padding: '10px 36px 10px 16px', 
              color: '#fff', 
              outline: 'none', 
              fontSize: '14px',
              transition: 'all 0.2s ease',
              cursor: 'pointer',
              appearance: 'none',
              backgroundImage: 'url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2394a3b8%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E")',
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'right 14px top 50%',
              backgroundSize: '10px auto'
            }}
          >
            <option value="">Verdict: All</option>
            <option value="phishing">Phishing</option>
            <option value="suspicious">Suspicious</option>
            <option value="legitimate">Legitimate</option>
            <option value="unknown">Unknown</option>
          </select>
          <select 
            value={filterRisk} 
            onChange={e => setFilterRisk(e.target.value)} 
            style={{ 
              backgroundColor: 'rgba(6, 9, 15, 0.7)', 
              border: '1px solid var(--border-subtle)', 
              borderRadius: '8px', 
              padding: '10px 36px 10px 16px', 
              color: '#fff', 
              outline: 'none', 
              fontSize: '14px',
              transition: 'all 0.2s ease',
              cursor: 'pointer',
              appearance: 'none',
              backgroundImage: 'url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2394a3b8%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E")',
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'right 14px top 50%',
              backgroundSize: '10px auto'
            }}
          >
            <option value="">Risk: All</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          <select 
            value={period} 
            onChange={e => setPeriod(Number(e.target.value))} 
            style={{ 
              backgroundColor: 'rgba(6, 9, 15, 0.7)', 
              border: '1px solid var(--border-subtle)', 
              borderRadius: '8px', 
              padding: '10px 36px 10px 16px', 
              color: '#fff', 
              outline: 'none', 
              fontSize: '14px',
              transition: 'all 0.2s ease',
              cursor: 'pointer',
              appearance: 'none',
              backgroundImage: 'url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2394a3b8%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E")',
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'right 14px top 50%',
              backgroundSize: '10px auto'
            }}
          >
            <option value={1}>Last 24 Hours</option>
            <option value={7}>Last 7 Days</option>
            <option value={30}>Last 30 Days</option>
            <option value={90}>Last 90 Days</option>
          </select>
          <button 
            onClick={handleExport} 
            style={{ 
              display: 'flex', alignItems: 'center', gap: '8px',
              backgroundColor: 'transparent',
              border: '1px solid var(--cyber-blue)',
              color: 'var(--cyber-blue)',
              padding: '10px 20px',
              borderRadius: '8px',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              boxShadow: 'inset 0 0 0 rgba(56, 189, 248, 0)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(56, 189, 248, 0.1)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(56, 189, 248, 0.2)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.boxShadow = 'inset 0 0 0 rgba(56, 189, 248, 0)';
            }}
          >
            <Download size={18} /> Export CSV
          </button>
        </div>
      </div>

      {isEmpty ? (
        <div style={{ ...cardStyle, textAlign: 'center', marginTop: '40px', padding: '60px 20px' }}>
          <Database size={48} style={{ color: 'var(--text-faint)', margin: '0 auto 15px' }} />
          <h3 style={{ color: '#fff', marginBottom: '10px' }}>No analytics available</h3>
          <p style={{ color: 'var(--text-muted)' }}>There are no investigations recorded for the selected period and filters.</p>
        </div>
      ) : (
        <>
          {/* STATS OVERVIEW GRID */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '30px' }}>
            <div style={{ ...cardStyle, display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                <span style={{ fontSize: '14px', fontWeight: 600 }}>Total Analyses</span>
                <Database size={18} style={{ color: 'var(--cyber-blue)' }} />
              </div>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#fff' }}>{data.overview.total_analyses}</div>
            </div>
            
            <div style={{ ...cardStyle, display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                <span style={{ fontSize: '14px', fontWeight: 600 }}>Analyses Today</span>
                <Activity size={18} style={{ color: '#10b981' }} />
              </div>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#fff' }}>{data.overview.today}</div>
            </div>
            
            <div style={{ ...cardStyle, display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                <span style={{ fontSize: '14px', fontWeight: 600 }}>Average Coverage</span>
                <Server size={18} style={{ color: '#a78bfa' }} />
              </div>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#fff' }}>
                {data.overview.avg_coverage} <span style={{ fontSize: '16px', color: 'var(--text-muted)' }}>/ 6</span>
              </div>
            </div>
            
            <div style={{ ...cardStyle, display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                <span style={{ fontSize: '14px', fontWeight: 600 }}>Fusion Conflicts</span>
                <AlertTriangle size={18} style={{ color: '#f59e0b' }} />
              </div>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#fff' }}>{data.fusion.conflicts}</div>
            </div>
          </div>

          {/* DISTRIBUTIONS GRID */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '30px' }}>
            
            <div style={cardStyle}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: 0, marginBottom: '20px', color: '#fff', fontSize: '18px' }}>
                <PieChart size={20} style={{ color: '#38bdf8' }} /> Verdict Distribution
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {Object.entries(data.verdicts).map(([verdict, count]) => {
                  let color = 'var(--text-muted)';
                  if (verdict === 'phishing') color = '#f43f5e';
                  if (verdict === 'suspicious') color = '#fbbf24';
                  if (verdict === 'legitimate') color = '#10b981';
                  
                  return (
                    <div key={verdict} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: '6px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: color }}></div>
                        <span style={{ textTransform: 'capitalize', color: 'var(--text-main)' }}>{verdict}</span>
                      </div>
                      <span style={{ fontWeight: 600, color: '#fff' }}>{count}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={cardStyle}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: 0, marginBottom: '20px', color: '#fff', fontSize: '18px' }}>
                <ShieldCheck size={20} style={{ color: '#fbbf24' }} /> Risk Distribution
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {Object.entries(data.risks).map(([risk, count]) => {
                  let color = 'var(--text-muted)';
                  if (risk === 'critical') color = '#ef4444';
                  if (risk === 'high') color = '#f97316';
                  if (risk === 'medium') color = '#eab308';
                  if (risk === 'low') color = '#22c55e';
                  
                  return (
                    <div key={risk} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: '6px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: color }}></div>
                        <span style={{ textTransform: 'capitalize', color: 'var(--text-main)' }}>{risk}</span>
                      </div>
                      <span style={{ fontWeight: 600, color: '#fff' }}>{count}</span>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>

          {/* AGENT TABLE */}
          <div style={cardStyle}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: 0, marginBottom: '20px', color: '#fff', fontSize: '18px' }}>
              <Cpu size={20} style={{ color: '#a78bfa' }} /> 6-Agent Analytics
            </h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse', color: 'var(--text-main)' }}>
                <thead>
                  <tr>
                    <th style={{ padding: '15px 10px', color: 'var(--text-muted)', fontWeight: 600, borderBottom: '1px solid var(--border-subtle)' }}>AI Engine</th>
                    <th style={{ padding: '15px 10px', color: 'var(--text-muted)', fontWeight: 600, borderBottom: '1px solid var(--border-subtle)' }}>Total Executions</th>
                    <th style={{ padding: '15px 10px', color: 'var(--text-muted)', fontWeight: 600, borderBottom: '1px solid var(--border-subtle)' }}>Successful</th>
                    <th style={{ padding: '15px 10px', color: 'var(--text-muted)', fontWeight: 600, borderBottom: '1px solid var(--border-subtle)' }}>Errors</th>
                    <th style={{ padding: '15px 10px', color: 'var(--text-muted)', fontWeight: 600, borderBottom: '1px solid var(--border-subtle)' }}>Availability Rate</th>
                    <th style={{ padding: '15px 10px', color: 'var(--text-muted)', fontWeight: 600, borderBottom: '1px solid var(--border-subtle)' }}>Avg Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.agents).map(([key, agent]) => (
                    <tr key={key} style={{ transition: 'background-color 0.2s' }} onMouseEnter={e => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.02)'} onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                      <td style={{ padding: '15px 10px', textTransform: 'capitalize', fontWeight: 600, color: '#fff', borderBottom: '1px solid var(--border-subtle)' }}>{agent.name}</td>
                      <td style={{ padding: '15px 10px', borderBottom: '1px solid var(--border-subtle)' }}>{agent.executions}</td>
                      <td style={{ padding: '15px 10px', color: '#10b981', borderBottom: '1px solid var(--border-subtle)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <CheckCircle2 size={16} /> {agent.success}
                        </div>
                      </td>
                      <td style={{ padding: '15px 10px', color: agent.errors > 0 ? '#f43f5e' : 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          {agent.errors > 0 && <XCircle size={16} />} {agent.errors}
                        </div>
                      </td>
                      <td style={{ padding: '15px 10px', borderBottom: '1px solid var(--border-subtle)' }}>
                        <div style={{ 
                          display: 'inline-block', 
                          padding: '4px 10px', 
                          borderRadius: '20px', 
                          fontSize: '12px', 
                          fontWeight: 600,
                          backgroundColor: agent.availability > 90 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                          color: agent.availability > 90 ? '#10b981' : '#f43f5e'
                        }}>
                          {agent.availability.toFixed(1)}%
                        </div>
                      </td>
                      <td style={{ padding: '15px 10px', borderBottom: '1px solid var(--border-subtle)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: agent.avg_latency > 0 ? 'var(--text-main)' : 'var(--text-faint)' }}>
                          <Zap size={16} style={{ color: agent.avg_latency > 0 ? '#fbbf24' : 'inherit' }} />
                          {agent.avg_latency > 0 ? `${agent.avg_latency.toFixed(1)} ms` : 'N/A'}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
