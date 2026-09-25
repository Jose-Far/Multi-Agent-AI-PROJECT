import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, 
  CheckCircle, 
  AlertTriangle, 
  Activity, 
  Search, 
  ExternalLink, 
  Cpu, 
  ArrowRight,
  Database,
  Globe,
  Layers
} from 'lucide-react';
import StatCard from '../components/StatCard';
import RiskBadge from '../components/RiskBadge';
import { getInvestigations } from '../services/api';

export default function DashboardPage({ onNavigateToAnalyze, onViewInvestigation, onNavigateToInvestigations }) {
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    phishing: 0,
    legitimate: 0,
    avgScore: '0.0'
  });
  const [quickUrl, setQuickUrl] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const data = await getInvestigations(20, 0);
      const analyses = data.analyses || [];
      setHistory(analyses.slice(0, 6));

      // Calculate stats
      const total = data.total || analyses.length;
      let phishingCount = 0;
      let legitimateCount = 0;
      let scoreSum = 0;
      let scoredCount = 0;

      analyses.forEach((item) => {
        if (item.final_verdict === 'phishing') phishingCount++;
        if (item.final_verdict === 'legitimate') legitimateCount++;
        if (item.risk_score !== null && item.risk_score !== undefined) {
          scoreSum += Number(item.risk_score);
          scoredCount++;
        }
      });

      setStats({
        total,
        phishing: phishingCount,
        legitimate: legitimateCount,
        avgScore: scoredCount > 0 ? (scoreSum / scoredCount).toFixed(1) : '0.0'
      });
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickScan = (e) => {
    e.preventDefault();
    if (quickUrl.trim()) {
      onNavigateToAnalyze(quickUrl.trim());
    }
  };

  const agentArchitecture = [
    { name: 'URL AI Agent', role: 'Lexical & Structural ML', status: 'Active', latency: '~12ms', curve: 'Identity / Calibrated' },
    { name: 'HTML AI Agent', role: 'DOM & Form Heuristics', status: 'Active', latency: '~35ms', curve: 'DOM Structure Mapping' },
    { name: 'SSL AI Agent', role: 'X.509 Cryptographic Trust', status: 'Active', latency: '~28ms', curve: 'Cryptographic Mapping' },
    { name: 'DNS AI Agent', role: 'Zone Telemetry & Fast-Flux', status: 'Active', latency: '~42ms', curve: 'Infrastructure Floor' },
    { name: 'Visual AI Agent', role: 'Perceptual Brand & Logo', status: 'Active', latency: '~85ms', curve: 'High-Confidence Direct' },
    { name: 'Threat Intel Agent', role: 'Global Feeds & Telemetry', status: 'Active', latency: '~60ms', curve: 'Indicator Step' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Top Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
        <StatCard
          title="Total Investigations"
          value={stats.total}
          subtitle="Saved to relational SQLite"
          icon={Database}
          color="var(--cyber-blue)"
        />
        <StatCard
          title="Phishing Detections"
          value={stats.phishing}
          subtitle={`${stats.total > 0 ? ((stats.phishing / stats.total) * 100).toFixed(0) : 0}% of scanned targets`}
          icon={ShieldAlert}
          color="#f43f5e"
          badge="High Severity"
        />
        <StatCard
          title="Legitimate Targets"
          value={stats.legitimate}
          subtitle="Confirmed trusted infrastructure"
          icon={CheckCircle}
          color="#34d399"
        />
        <StatCard
          title="Avg Calibrated Risk"
          value={`${stats.avgScore} / 100`}
          subtitle="OSRH-v2 Harmonized scale"
          icon={Activity}
          color="#fbbf24"
        />
      </div>

      {/* Quick URL Scanner Banner */}
      <div style={{
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '8px',
        padding: '24px 28px',
        background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 58, 138, 0.25) 100%)',
        boxShadow: '0 4px 16px rgba(0,0,0,0.25)'
      }}>
        <div style={{ marginBottom: '14px' }}>
          <h2 style={{ fontSize: '16px', fontWeight: 600, color: '#f8fafc', marginBottom: '4px' }}>
            Quick Website Security Scanner
          </h2>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Execute simultaneous evaluation across all 6 specialized cybersecurity agents and calculate calibrated risk score.
          </p>
        </div>

        <form onSubmit={handleQuickScan} style={{ display: 'flex', gap: '12px' }}>
          <div style={{
            position: 'relative',
            flex: 1,
            display: 'flex',
            alignItems: 'center'
          }}>
            <Globe size={16} color="var(--text-faint)" style={{ position: 'absolute', left: '14px' }} />
            <input
              type="text"
              placeholder="Enter target URL (e.g. https://example.com)..."
              value={quickUrl}
              onChange={(e) => setQuickUrl(e.target.value)}
              style={{
                width: '100%',
                padding: '12px 14px 12px 42px',
                borderRadius: '6px',
                border: '1px solid var(--border-subtle)',
                backgroundColor: 'rgba(6, 9, 15, 0.7)',
                color: '#f8fafc',
                fontSize: '13px',
                fontFamily: 'monospace',
                outline: 'none',
                transition: 'border-color 0.2s'
              }}
              onFocus={(e) => e.target.style.borderColor = 'var(--cyber-blue)'}
              onBlur={(e) => e.target.style.borderColor = 'var(--border-subtle)'}
            />
          </div>
          <button
            type="submit"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '0 22px',
              borderRadius: '6px',
              backgroundColor: 'var(--cyber-blue)',
              color: '#06090f',
              fontSize: '13px',
              fontWeight: 700,
              border: 'none',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
              transition: 'opacity 0.2s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.opacity = '0.9'}
            onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
          >
            <span>Scan Target</span>
            <ArrowRight size={16} />
          </button>
        </form>

        {/* Quick Sample Target Chips */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '14px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Presets:
          </span>
          <button
            type="button"
            onClick={() => onNavigateToAnalyze('https://www.youtube.com')}
            style={{
              padding: '4px 10px',
              borderRadius: '4px',
              border: '1px solid rgba(52, 211, 153, 0.3)',
              backgroundColor: 'rgba(52, 211, 153, 0.08)',
              color: '#34d399',
              fontSize: '11px',
              cursor: 'pointer',
              fontWeight: 500
            }}
          >
            YouTube (Legitimate)
          </button>
          <button
            type="button"
            onClick={() => onNavigateToAnalyze('http://sp0tify-mod-login.com')}
            style={{
              padding: '4px 10px',
              borderRadius: '4px',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              backgroundColor: 'rgba(244, 63, 94, 0.08)',
              color: '#fb7185',
              fontSize: '11px',
              cursor: 'pointer',
              fontWeight: 500
            }}
          >
            Spotify Mod (Phishing Sample)
          </button>
          <button
            type="button"
            onClick={() => onNavigateToAnalyze('http://paypa1-security-update.com')}
            style={{
              padding: '4px 10px',
              borderRadius: '4px',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              backgroundColor: 'rgba(244, 63, 94, 0.08)',
              color: '#fb7185',
              fontSize: '11px',
              cursor: 'pointer',
              fontWeight: 500
            }}
          >
            PayPal Verify (Phishing Sample)
          </button>
        </div>
      </div>

      {/* Grid: 6 Agents Status & Recent Investigations */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '24px' }}>
        {/* 6 AI Agents Overview */}
        <div style={{
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '8px',
          padding: '20px 24px',
          display: 'flex',
          flexDirection: 'column',
          gap: '14px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Cpu size={16} color="var(--cyber-blue)" />
              <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc' }}>
                Multi-Agent Sensor Network (6 / 6)
              </h3>
            </div>
            <span style={{ fontSize: '11px', color: '#34d399', fontWeight: 600 }}>
              All Subsystems Operational
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {agentArchitecture.map((agent, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '9px 12px',
                  borderRadius: '6px',
                  backgroundColor: 'rgba(6, 9, 15, 0.4)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '12px'
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#f8fafc' }}>{agent.name}</div>
                  <div style={{ fontSize: '10px', color: 'var(--text-faint)' }}>{agent.role}</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span style={{
                    fontSize: '10px',
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: '4px',
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    color: '#34d399'
                  }}>
                    {agent.status}
                  </span>
                  <div style={{ fontSize: '10px', color: 'var(--text-faint)', marginTop: '2px', fontFamily: 'monospace' }}>
                    {agent.latency}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Investigations Feed */}
        <div style={{
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '8px',
          padding: '20px 24px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          gap: '14px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Database size={16} color="var(--cyber-blue)" />
              <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc' }}>
                Recent Investigations
              </h3>
            </div>
            <button
              onClick={() => onNavigateToInvestigations ? onNavigateToInvestigations() : onNavigateToAnalyze(null, 'investigations')}
              style={{
                fontSize: '11px',
                color: 'var(--cyber-blue)',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontWeight: 600
              }}
            >
              View Full History →
            </button>
          </div>

          {loading ? (
            <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-faint)' }}>
              Loading investigation telemetry...
            </div>
          ) : history.length === 0 ? (
            <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-faint)' }}>
              No previous investigations logged. Run a target scan above.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {history.map((inv) => (
                <div
                  key={inv.analysis_id}
                  onClick={() => onViewInvestigation(inv.analysis_id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '9px 12px',
                    borderRadius: '6px',
                    backgroundColor: 'rgba(6, 9, 15, 0.4)',
                    border: '1px solid var(--border-subtle)',
                    cursor: 'pointer',
                    transition: 'background-color 0.15s ease'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--cyber-glow, rgba(56, 189, 248, 0.05))'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'rgba(6, 9, 15, 0.4)'}
                >
                  <div style={{ maxWidth: '240px', overflow: 'hidden' }}>
                    <div style={{
                      fontWeight: 600,
                      color: '#f8fafc',
                      fontSize: '12px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }}>
                      {inv.url}
                    </div>
                    <div style={{ fontSize: '10px', color: 'var(--text-faint)', fontFamily: 'monospace' }}>
                      {inv.analysis_id}
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{
                      fontSize: '12px',
                      fontFamily: 'monospace',
                      fontWeight: 700,
                      color: inv.risk_score !== null ? '#ffffff' : 'var(--text-faint)'
                    }}>
                      {inv.risk_score !== null ? `${Number(inv.risk_score).toFixed(1)}` : 'N/A'}
                    </span>
                    <RiskBadge type="verdict" value={inv.final_verdict} size="small" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
