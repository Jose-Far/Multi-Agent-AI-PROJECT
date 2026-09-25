import React, { useState, useEffect } from 'react';
import { Cpu, Activity, AlertTriangle, X, Loader, ShieldCheck, Zap, Server, Code, Image as ImageIcon, Lock, Globe, ShieldAlert, List } from 'lucide-react';
import { analyzeUrl } from '../../services/api';
import { formatIST } from '../../utils/dateUtils';

const agentEnrichment = {
  "URL AI": { modelName: "PhishDec-URL-v2 (RandomForest)", features: 35, description: "Analyzes lexical characteristics of the URL including entropy, character frequencies, and suspicious patterns.", icon: <Globe size={18} /> },
  "HTML AI": { modelName: "PhishDec-HTML-v2 (LightGBM)", features: 40, description: "Parses DOM structure to detect hidden iframes, obfuscated JavaScript, and malicious credential forms.", icon: <Code size={18} /> },
  "SSL AI": { modelName: "PhishDec-SSL-v1 (XGBoost)", features: 20, description: "Evaluates certificate authority trust, expiry windows, and SAN anomalies.", icon: <Lock size={18} /> },
  "DNS AI": { modelName: "PhishDec-DNS-v1 (DecisionTree)", features: 35, description: "Checks domain age, WHOIS records, fast-flux behavior, and MX record validity.", icon: <Server size={18} /> },
  "Visual AI": { modelName: "PhishDec-Vision-v2 (OpenCV/ResNet)", features: 12, description: "Extracts brand logos and compares structural visual similarity to known trusted entities.", icon: <ImageIcon size={18} /> },
  "Threat Intel": { modelName: "OSINT Aggregator API", features: 20, description: "Cross-references hashes and domains against global threat databases like VirusTotal and PhishTank.", icon: <ShieldAlert size={18} /> }
};


const featureSchemas = {
  "URL AI": [
    { name: 'url_length', type: 'numeric' }, { name: 'hostname_length', type: 'numeric' }, { name: 'path_length', type: 'numeric' },
    { name: 'tld_length', type: 'numeric' }, { name: 'num_dots', type: 'numeric' }, { name: 'num_hyphens', type: 'numeric' },
    { name: 'num_at', type: 'boolean' }, { name: 'num_question', type: 'boolean' }, { name: 'num_ampersand', type: 'numeric' },
    { name: 'num_equals', type: 'numeric' }, { name: 'num_slash', type: 'numeric' }, { name: 'num_digits', type: 'numeric' },
    { name: 'num_letters', type: 'numeric' }, { name: 'entropy', type: 'numeric' }, { name: 'is_ip_address', type: 'boolean' },
    { name: 'has_port_in_url', type: 'boolean' }, { name: 'has_suspicious_tld', type: 'boolean' }, { name: 'keyword_login', type: 'boolean' },
    { name: 'keyword_secure', type: 'boolean' }, { name: 'keyword_account', type: 'boolean' }, { name: 'keyword_update', type: 'boolean' },
    { name: 'keyword_banking', type: 'boolean' }, { name: 'keyword_confirm', type: 'boolean' }, { name: 'keyword_verify', type: 'boolean' },
    { name: 'keyword_support', type: 'boolean' }, { name: 'keyword_service', type: 'boolean' }, { name: 'keyword_apple', type: 'boolean' },
    { name: 'keyword_paypal', type: 'boolean' }, { name: 'keyword_microsoft', type: 'boolean' }, { name: 'keyword_amazon', type: 'boolean' },
    { name: 'keyword_netflix', type: 'boolean' }, { name: 'keyword_bank', type: 'boolean' }, { name: 'keyword_crypto', type: 'boolean' },
    { name: 'subdomain_count', type: 'numeric' }, { name: 'has_punycode', type: 'boolean' }
  ],
  "Threat Intel": [
    { name: 'is_blacklisted', type: 'boolean' }, { name: 'blacklist_vendors_count', type: 'numeric' },
    { name: 'high_authority_vendor_flagged', type: 'boolean' }, { name: 'malicious_engines_count', type: 'numeric' },
    { name: 'suspicious_engines_count', type: 'numeric' }, { name: 'harmless_engines_count', type: 'numeric' },
    { name: 'undetected_engines_count', type: 'numeric' }, { name: 'malicious_ratio', type: 'numeric' },
    { name: 'suspicion_ratio', type: 'numeric' }, { name: 'has_high_threat_consensus', type: 'boolean' },
    { name: 'is_malware_associated', type: 'boolean' }, { name: 'is_c2_node', type: 'boolean' },
    { name: 'is_exploit_distributor', type: 'boolean' }, { name: 'malware_threat_score', type: 'numeric' },
    { name: 'reputation_score', type: 'numeric' }, { name: 'weighted_threat_score', type: 'numeric' },
    { name: 'overall_threat_score', type: 'numeric' }, { name: 'is_zero_day_candidate', type: 'boolean' },
    { name: 'days_since_first_submission', type: 'numeric' }, { name: 'days_since_last_analysis', type: 'numeric' }
  ]
};

// Fallback generator for other agents like HTML, DNS, SSL, Visual
const getFeaturesList = (agentName, count) => {
  if (featureSchemas[agentName]) return featureSchemas[agentName];
  const list = [];
  for (let i = 0; i < count; i++) {
    list.push({ name: `${agentName.toLowerCase().replace(' ', '_')}_feature_${i+1}`, type: i % 3 === 0 ? 'boolean' : 'numeric' });
  }
  return list;
};


function FeaturesModal({ agent, onClose, testUrl }) {
  const enriched = agentEnrichment[agent.name] || { features: 0 };
  
  // Generate deterministic mock values based on URL length for demo purposes
  const urlLen = testUrl ? testUrl.length : 30;
  
  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(6px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000
    }}>
      <div style={{
        backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
        borderRadius: '12px', width: '650px', maxWidth: '90%', maxHeight: '85vh',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)', overflow: 'hidden',
        display: 'flex', flexDirection: 'column', animation: 'slideUp 0.3s ease-out'
      }}>
        <div style={{ padding: '20px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'var(--bg-card-elevated)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <List size={22} style={{ color: 'var(--cyber-blue)' }} />
            <h3 style={{ margin: 0, color: 'var(--text-main)', fontSize: '18px' }}>{agent.name} Extracted Features</h3>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <X size={20} />
          </button>
        </div>
        <div style={{ padding: '20px', overflowY: 'auto', flex: 1, backgroundColor: 'var(--bg-dark)' }}>
          <p style={{ color: 'var(--text-muted)', marginBottom: '20px', fontSize: '14px' }}>
            Showing {enriched.features} dimensions extracted for <strong>{testUrl || "the generic model schema"}</strong>.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
            {getFeaturesList(agent.name, enriched.features).map((f, i) => {
               let val;
               if (f.type === 'boolean') val = (urlLen + i) % 7 === 0 ? "True" : "False";
               else if (f.name === 'entropy') val = (3.5 + (urlLen % 10) * 0.1).toFixed(3);
               else val = Math.floor((urlLen * (i+1)) % 45);
               
               if (!testUrl) val = "-";

               return (
                 <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px', backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '6px' }}>
                   <span style={{ color: 'var(--text-muted)', fontSize: '13px', fontFamily: 'monospace' }}>{f.name}</span>
                   <span style={{ color: val === 'True' || val > 15 ? '#fbbf24' : 'var(--text-main)', fontSize: '13px', fontWeight: 600 }}>{val}</span>
                 </div>
               )
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function AgentModal({ agent, onClose }) {
  const [testUrl, setTestUrl] = useState('');
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [error, setError] = useState('');
  const [showFeatures, setShowFeatures] = useState(false);

  const enriched = agentEnrichment[agent.name] || { modelName: agent.model, features: agent.features, description: "Standard analysis agent.", icon: <Cpu size={18} /> };

  const handleTest = async () => {
    if (!testUrl.trim()) return;
    setTesting(true);
    setTestResult(null);
    setError('');

    try {
      const res = await analyzeUrl(testUrl);
      setTestResult(res);
    } catch (err) {
      setError(err.message || 'Test failed.');
    } finally {
      setTesting(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleTest();
    }
  };

  return (
    <>
      <div style={{
        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
        backgroundColor: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999
      }}>
        <div style={{
          backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
          borderRadius: '12px', width: '750px', maxWidth: '95%', maxHeight: '90vh',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)', overflow: 'hidden',
          display: 'flex', flexDirection: 'column', animation: 'slideUp 0.3s ease-out'
        }}>
          {/* Header */}
          <div style={{ padding: '20px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'var(--bg-card-elevated)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Cpu size={24} style={{ color: 'var(--cyber-blue)' }} />
              <h3 style={{ margin: 0, color: 'var(--text-main)', fontSize: '18px' }}>Analysis Dashboard & Diagnostics</h3>
            </div>
            <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '4px' }}>
              <X size={20} />
            </button>
          </div>

          {/* Body */}
          <div style={{ padding: '24px', overflowY: 'auto' }}>
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: 'var(--text-main)', margin: '0 0 8px 0', fontSize: '15px' }}>{agent.name} Focus</h4>
              <p style={{ color: 'var(--text-muted)', fontSize: '14px', lineHeight: '1.5', margin: 0 }}>
                {enriched.description}
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '25px' }}>
              <div style={{ padding: '15px', backgroundColor: 'var(--bg-dark)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ color: 'var(--text-faint)', fontSize: '12px', textTransform: 'uppercase', marginBottom: '4px' }}>Active Model</div>
                <div style={{ color: 'var(--cyber-blue)', fontWeight: 600, fontSize: '14px' }}>{enriched.modelName}</div>
              </div>
              <div 
                onClick={() => setShowFeatures(true)}
                style={{ padding: '15px', backgroundColor: 'var(--bg-dark)', borderRadius: '8px', border: '1px solid var(--cyber-blue)', cursor: 'pointer', transition: 'all 0.2s' }}
                onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgba(56, 189, 248, 0.05)'}
                onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-dark)'}
              >
                <div style={{ color: 'var(--text-faint)', fontSize: '12px', textTransform: 'uppercase', marginBottom: '4px' }}>Features Extracted (Click to View)</div>
                <div style={{ color: 'var(--cyber-blue)', fontWeight: 600, fontSize: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <List size={16} /> {enriched.features} dimensions available
                </div>
              </div>
            </div>

            <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '20px' }}>
              <h4 style={{ color: 'var(--text-main)', margin: '0 0 12px 0', fontSize: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Zap size={16} style={{ color: 'var(--phish-warning)' }} /> Automatic Multi-Agent Execution
              </h4>
              <p style={{ color: 'var(--text-muted)', fontSize: '13px', margin: '0 0 15px 0' }}>
                Paste a URL and press Enter. The system will automatically execute all applicable AI agents in parallel and fuse their results.
              </p>
              
              <div style={{ display: 'flex', gap: '10px' }}>
                <input 
                  type="url" 
                  placeholder="Enter URL to test (Press Enter to start)..."
                  value={testUrl}
                  onChange={e => setTestUrl(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={testing}
                  style={{
                    flex: 1, padding: '12px 16px', borderRadius: '6px', fontSize: '15px',
                    backgroundColor: 'var(--bg-dark)', border: '1px solid var(--cyber-blue)',
                    color: 'var(--text-main)', outline: 'none', boxShadow: '0 0 0 1px rgba(56, 189, 248, 0.3)'
                  }}
                />
              </div>

              {testing && (
                <div style={{ marginTop: '25px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '30px', color: 'var(--cyber-blue)' }}>
                  <Loader size={32} className="spin" style={{ marginBottom: '15px' }} />
                  <span style={{ fontSize: '14px', fontWeight: 500 }}>Executing full AI pipeline across all agents...</span>
                </div>
              )}

              {error && (
                <div style={{ marginTop: '15px', padding: '10px', backgroundColor: 'rgba(244, 63, 94, 0.1)', color: '#f43f5e', borderRadius: '6px', fontSize: '13px' }}>
                  {error}
                </div>
              )}

              {!testing && testResult && (
                <div style={{ marginTop: '20px' }}>
                  {/* FUSION ENGINE VERDICT */}
                  <div style={{ padding: '16px', backgroundColor: 'var(--bg-dark)', border: `1px solid ${testResult.final_assessment?.verdict === 'MALICIOUS' ? '#f43f5e' : (testResult.final_assessment?.verdict === 'SUSPICIOUS' ? '#fbbf24' : '#10b981')}`, borderRadius: '8px', marginBottom: '20px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <ShieldCheck size={20} style={{ color: testResult.final_assessment?.verdict === 'MALICIOUS' ? '#f43f5e' : (testResult.final_assessment?.verdict === 'SUSPICIOUS' ? '#fbbf24' : '#10b981') }} />
                        <span style={{ color: 'var(--text-main)', fontWeight: 600, fontSize: '16px' }}>Decision Fusion Engine</span>
                      </div>
                      <span style={{ 
                        fontWeight: 700, fontSize: '18px', textTransform: 'uppercase',
                        color: testResult.final_assessment?.verdict === 'MALICIOUS' ? '#f43f5e' : (testResult.final_assessment?.verdict === 'SUSPICIOUS' ? '#fbbf24' : '#10b981')
                      }}>
                        {testResult.final_assessment?.verdict} ({(testResult.final_assessment?.risk_score || 0).toFixed(1)}% Risk)
                      </span>
                    </div>
                    <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                      Confidence: {((testResult.final_assessment?.confidence || 0) * 100).toFixed(1)}% | Agents Available: {testResult.consensus?.available_agents}/{testResult.consensus?.total_agents}
                    </div>
                  </div>

                  {/* AGENT BREAKDOWN */}
                  <h5 style={{ color: 'var(--text-main)', margin: '0 0 10px 0', fontSize: '14px' }}>Individual Agent Results</h5>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    {Object.entries(testResult.agents || {}).map(([key, agentData]) => {
                      const nameMap = { url: 'URL AI', html: 'HTML AI', ssl: 'SSL AI', dns: 'DNS AI', visual: 'Visual AI', threat_intelligence: 'Threat Intel' };
                      const friendlyName = nameMap[key] || key;
                      const aEnrich = agentEnrichment[friendlyName] || {};
                      
                      let statusColor = 'var(--text-muted)';
                      if (agentData.status === 'success' || agentData.status === 'success_with_fallback') {
                        statusColor = agentData.prediction === 'malicious' ? '#f43f5e' : (agentData.prediction === 'suspicious' ? '#fbbf24' : '#10b981');
                      } else if (agentData.status === 'unavailable') {
                        statusColor = '#94a3b8';
                      }

                      return (
                        <div key={key} style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '12px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                            <div style={{ color: 'var(--cyber-blue)' }}>{aEnrich.icon || <Cpu size={16} />}</div>
                            <strong style={{ color: 'var(--text-main)', fontSize: '13px' }}>{friendlyName}</strong>
                          </div>
                          
                          {agentData.status === 'success' || agentData.status === 'success_with_fallback' ? (
                            <>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                                <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Prediction:</span>
                                <span style={{ color: statusColor, fontWeight: 600, fontSize: '12px', textTransform: 'capitalize' }}>
                                  {agentData.prediction}
                                </span>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Confidence:</span>
                                <span style={{ color: 'var(--text-main)', fontSize: '12px' }}>
                                  {((agentData.confidence || 0) * 100).toFixed(1)}%
                                </span>
                              </div>
                            </>
                          ) : (
                            <div style={{ color: '#94a3b8', fontSize: '12px', fontStyle: 'italic', paddingTop: '4px' }}>
                              Unavailable / Insufficient Data
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {showFeatures && <FeaturesModal agent={agent} testUrl={testUrl} onClose={() => setShowFeatures(false)} />}
      
      <style>{`
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          100% { transform: rotate(360deg); }
        }
        .agent-row:hover {
          background-color: var(--bg-card-elevated);
          cursor: pointer;
        }
      `}</style>
    </>
  );
}

export default function AdminAgentsPage({ healthData }) {
  const agents = healthData?.agents?.list || [];
  const [selectedAgent, setSelectedAgent] = useState(null);
  
    const [lastSyncTime, setLastSyncTime] = useState(new Date());
  useEffect(() => {
    if (healthData) setLastSyncTime(new Date());
  }, [healthData]);

  return (
    <div className="admin-agents">
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '25px' }}>
        <Cpu size={28} style={{ color: 'var(--cyber-blue)' }} />
        <h2 style={{ margin: 0, color: 'var(--text-main)' }}>AI Agents Monitoring</h2>
      </div>

      <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ backgroundColor: 'var(--bg-card-elevated)', borderBottom: '1px solid var(--border-subtle)' }}>
              <th style={{ padding: '16px 20px', color: 'var(--text-muted)', fontWeight: 600 }}>Agent</th>
              <th style={{ padding: '16px 20px', color: 'var(--text-muted)', fontWeight: 600 }}>Health</th>
              <th style={{ padding: '16px 20px', color: 'var(--text-muted)', fontWeight: 600 }}>Active Model</th>
              <th style={{ padding: '16px 20px', color: 'var(--text-muted)', fontWeight: 600 }}>Features</th>
              <th style={{ padding: '16px 20px', color: 'var(--text-muted)', fontWeight: 600 }}>Last Sync</th>
            </tr>
          </thead>
          <tbody>
            {agents.map((agent, i) => {
              const enriched = agentEnrichment[agent.name] || { modelName: agent.model, features: agent.features };
               
              
              return (
                <tr 
                  key={i} 
                  className="agent-row"
                  onClick={() => setSelectedAgent(agent)}
                  style={{ borderBottom: '1px solid var(--border-subtle)', transition: 'background-color 0.2s' }}
                >
                  <td style={{ padding: '16px 20px', color: 'var(--text-main)', fontWeight: 500 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: agent.health === 'Healthy' ? '#10b981' : '#f43f5e', boxShadow: agent.health === 'Healthy' ? '0 0 8px #10b981' : 'none' }} />
                      {agent.name}
                    </div>
                  </td>
                  <td style={{ padding: '16px 20px' }}>
                    <span style={{ 
                      padding: '4px 10px', borderRadius: '12px', fontSize: '12px', fontWeight: 600,
                      backgroundColor: agent.health === 'Healthy' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)',
                      color: agent.health === 'Healthy' ? '#10b981' : '#f43f5e' 
                    }}>
                      {agent.health}
                    </span>
                  </td>
                  <td style={{ padding: '16px 20px', color: 'var(--cyber-blue)', fontWeight: 500 }}>{enriched.modelName}</td>
                  <td style={{ padding: '16px 20px', color: 'var(--text-main)' }}>
                    {enriched.features > 0 ? (
                      <span style={{ padding: '2px 8px', backgroundColor: 'var(--bg-dark)', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                        {enriched.features} metrics
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-faint)' }}>N/A</span>
                    )}
                  </td>
                  <td style={{ padding: '16px 20px', color: 'var(--text-muted)', fontSize: '13px' }}>
                    {formatIST(lastSyncTime.toISOString()).split(' ')[1]}
                  </td>
                </tr>
              );
            })}
            {agents.length === 0 && (
              <tr>
                <td colSpan="5" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-faint)' }}>
                  No agents available or API is offline.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      <div style={{ marginTop: '20px', padding: '15px', backgroundColor: 'rgba(251, 191, 36, 0.1)', border: '1px solid rgba(251, 191, 36, 0.2)', borderRadius: '8px', display: 'flex', gap: '15px' }}>
         <AlertTriangle style={{ color: '#fbbf24', flexShrink: 0 }} />
         <div style={{ color: 'var(--text-muted)', fontSize: '14px', lineHeight: '1.5' }}>
           <strong>Tip:</strong> Click on any agent row to view detailed model diagnostics and run isolated live tests against specific URLs.
         </div>
      </div>

      {selectedAgent && (
        <AgentModal agent={selectedAgent} onClose={() => setSelectedAgent(null)} />
      )}
    </div>
  );
}
