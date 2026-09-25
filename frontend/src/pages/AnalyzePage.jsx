import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  Globe, 
  ShieldAlert, 
  CheckCircle2, 
  AlertTriangle, 
  HelpCircle, 
  Cpu, 
  Database, 
  Clock, 
  Layers, 
  FileText, 
  Sliders, 
  Info,
  ChevronRight,
  ExternalLink
} from 'lucide-react';
import AgentCard from '../components/AgentCard';
import RiskBadge from '../components/RiskBadge';
import LoadingState from '../components/LoadingState';
import AnalysisHeader from '../components/AnalysisHeader';
import ExportModal from '../components/ExportModal';
import RiskOverview from '../components/RiskOverview';
import EvidenceCoverage from '../components/EvidenceCoverage';
import FusionSummary from '../components/FusionSummary';
import RiskFactors from '../components/RiskFactors';
import RiskDistribution from '../components/RiskDistribution';
import EvidenceExplorer from '../components/EvidenceExplorer';
import ExplainabilityPanel from '../components/ExplainabilityPanel';
import InvestigationTimeline from '../components/InvestigationTimeline';
import AnalystNotes from '../components/AnalystNotes';
import InvestigationTags from '../components/InvestigationTags';
import AgentDetailsModal from '../components/AgentDetailsModal';


import { analyzeUrl, getInvestigationDetail, updateInvestigation } from '../services/api';
import { useAuth } from '../services/AuthContext';

export default function AnalyzePage({ initialUrl, targetId, onClearTargetId, onClearInitialUrl, onGenerateReport }) {
  const { user } = useAuth();
  const isViewer = user?.role === 'viewer';
  const [url, setUrl] = useState(initialUrl || 'https://www.youtube.com');
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [urlOnly, setUrlOnly] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const initRef = useRef(false);

  // If a targetId was passed from another page (Dashboard or Investigations), load it directly
  useEffect(() => {
    if (initRef.current) return;
    
    if (targetId) {
      initRef.current = true;
      loadInvestigation(targetId);
      if (onClearTargetId) onClearTargetId();
    } else if (initialUrl && !result) {
      initRef.current = true;
      handleAnalyze(initialUrl);
      if (onClearInitialUrl) onClearInitialUrl();
    }
  }, [targetId, initialUrl]);

  
  const handleSave = async () => {
    if (!result || !result.analysis_id) return;
    try {
      await updateInvestigation(result.analysis_id, { is_saved: true });
      alert(`Investigation ${result.analysis_id} saved successfully! You can view it in the Investigation History tab by selecting "Saved Only".`);
    } catch (e) {
      alert("Error saving investigation: " + e.message);
    }
  };

  const handleAddNote = async () => {
    if (!result || !result.analysis_id) return;
    const note = prompt("Enter your analyst note for this investigation:", result.notes || "");
    if (note === null) return;
    
    try {
      await updateInvestigation(result.analysis_id, { notes: note });
      setResult({...result, notes: note});
      alert("Note saved successfully! You can view it in the Analyst Notes section below or in the Investigation History.");
    } catch (e) {
      alert("Error saving note: " + e.message);
    }
  };

  const handleExport = () => {
    if (!result) return;
    setShowExportModal(true);
  };

  const handleShare = () => {
    if (result && result.analysis_id) {
      const shareText = `[PhishDec Investigation Report]
ID: ${result.analysis_id}
Target: ${result.target?.url}
Verdict: ${result.final_assessment?.verdict?.toUpperCase()}
Risk Score: ${result.final_assessment?.risk_score}/100

Link: http://localhost:5173/analyze?id=${result.analysis_id}`;
      navigator.clipboard.writeText(shareText);
      alert("Full Investigation Summary copied to clipboard! You can now paste it to your team.");
    } else {
      alert("No active investigation to share.");
    }
  };

  const loadInvestigation = async (id) => {
    try {
      setLoading(true);
      setError(null);
      const data = await getInvestigationDetail(id);
      setResult(data);
      if (data?.target?.url) {
        setUrl(data.target.url);
      }
    } catch (err) {
      setError(`Failed to load investigation ${id}: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async (targetToAnalyze = url) => {
    const target = (targetToAnalyze || url).trim();
    if (!target) {
      setError('Please enter a valid URL to analyze.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);

      const data = await analyzeUrl(target, urlOnly);
      setResult(data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message || 'An error occurred during multi-agent analysis.');
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = (e) => {
    e.preventDefault();
    handleAnalyze();
  };

  // Helper to safely render risk factor strings or objects
  const formatFactor = (factor) => {
    if (typeof factor === 'string') return factor;
    if (typeof factor === 'object' && factor !== null) {
      if (factor.feature) {
        return `${factor.feature}: ${factor.value ?? ''} (${factor.direction || 'impact factor'})`;
      }
      return JSON.stringify(factor);
    }
    return String(factor);
  };

  // 6 canonical agent keys
  const AGENT_KEYS = ['url', 'html', 'ssl', 'dns', 'visual', 'threat_intelligence'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      <ExportModal isOpen={showExportModal} onClose={() => setShowExportModal(false)} result={result} />
      {/* Target Analysis Form Card - Hidden for Viewers */}
      {!isViewer && (
        <div style={{
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '8px',
          padding: '24px 28px',
          boxShadow: '0 4px 16px rgba(0,0,0,0.2)'
        }}>
          <form onSubmit={onSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <label style={{
              display: 'block',
              fontSize: '12px',
              fontWeight: 600,
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.6px',
              marginBottom: '8px'
            }}>
              Target URL for Security Inspection
            </label>
            <div style={{ display: 'flex', gap: '12px' }}>
              <div style={{ position: 'relative', flex: 1, display: 'flex', alignItems: 'center' }}>
                <Globe size={18} color="var(--text-faint)" style={{ position: 'absolute', left: '16px' }} />
                <input
                  type="text"
                  placeholder="https://target-domain.com/path"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  disabled={loading}
                  style={{
                    width: '100%',
                    padding: '14px 16px 14px 46px',
                    borderRadius: '6px',
                    border: '1px solid var(--border-subtle)',
                    backgroundColor: 'rgba(6, 9, 15, 0.7)',
                    color: '#ffffff',
                    fontSize: '14px',
                    fontFamily: 'ui-monospace, monospace',
                    outline: 'none',
                    transition: 'border-color 0.2s'
                  }}
                  onFocus={(e) => e.target.style.borderColor = 'var(--cyber-blue)'}
                  onBlur={(e) => e.target.style.borderColor = 'var(--border-subtle)'}
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  padding: '0 28px',
                  borderRadius: '6px',
                  backgroundColor: 'var(--cyber-blue)',
                  color: '#06090f',
                  fontSize: '14px',
                  fontWeight: 700,
                  border: 'none',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  opacity: loading ? 0.7 : 1,
                  boxShadow: '0 0 16px var(--cyber-glow)',
                  whiteSpace: 'nowrap'
                }}
              >
                <Search size={18} />
                <span>{loading ? 'Analyzing...' : 'Execute Analysis'}</span>
              </button>
            </div>
          </div>

          {/* Options & Sample Targets */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '12px',
            borderTop: '1px solid var(--border-subtle)',
            paddingTop: '14px'
          }}>
            {/* Presets */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Test Samples:
              </span>
              <button type="button" onClick={() => { setUrl('https://www.youtube.com'); handleAnalyze('https://www.youtube.com'); }} className="sample-btn sample-btn-legit">
                YouTube (Legitimate)
              </button>
              <button
                type="button"
                onClick={() => { setUrl('http://sp0tify-mod-login.com'); handleAnalyze('http://sp0tify-mod-login.com'); }}
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
                Spotify Mod (Phishing)
              </button>
              <button
                type="button"
                onClick={() => { setUrl('http://paypa1-security-update.com'); handleAnalyze('http://paypa1-security-update.com'); }}
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
                PayPal Verify (Phishing)
              </button>
              <button type="button" onClick={() => { setUrl('https://www.google.com'); handleAnalyze('https://www.google.com'); }} className="sample-btn sample-btn-baseline">
                Google (Baseline)
              </button>
            </div>

            {/* Fast Mode Toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '12px', color: 'var(--text-muted)' }}>
              <input
                type="checkbox"
                checked={urlOnly}
                onChange={(e) => setUrlOnly(e.target.checked)}
                style={{ accentColor: 'var(--cyber-blue)', cursor: 'pointer' }}
              />
              <span>Fast URL-Only Mode</span>
            </label>
          </div>
        </form>
      </div>
      )}

      {/* View-Only Indicator for Viewers without Results */}
      {isViewer && !result && !loading && (
        <div style={{ padding: '60px', textAlign: 'center', backgroundColor: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <ShieldAlert size={32} color="var(--text-faint)" style={{ marginBottom: '16px' }} />
          <h3 style={{ margin: '0 0 8px 0', color: '#f8fafc', fontWeight: 500 }}>No Investigation Loaded</h3>
          <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '14px' }}>Please select an investigation from the History tab.</p>
        </div>
      )}

      {/* Loading Indicator */}
      {loading && <LoadingState targetUrl={url} />}

      {/* Error Banner */}
      {error && !loading && (
        <div style={{
          backgroundColor: 'rgba(244, 63, 94, 0.1)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          borderRadius: '8px',
          padding: '18px 24px',
          display: 'flex',
          alignItems: 'center',
          gap: '14px',
          color: '#fb7185'
        }}>
          <AlertTriangle size={22} color="#f43f5e" />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 600, fontSize: '14px' }}>Analysis Encountered an Error</div>
            <div style={{ fontSize: '12px', marginTop: '2px', opacity: 0.9 }}>{error}</div>
          </div>
          <button
            onClick={() => handleAnalyze()}
            style={{
              backgroundColor: 'rgba(244, 63, 94, 0.2)',
              border: '1px solid rgba(244, 63, 94, 0.4)',
              color: '#ffffff',
              padding: '6px 14px',
              borderRadius: '4px',
              fontSize: '12px',
              cursor: 'pointer'
            }}
          >
            Retry Scan
          </button>
        </div>
      )}

      {/* Results View */}
      {result && !loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <AnalysisHeader 
              target={result.target}
              analysisId={result.analysis_id}
              timestamp={result.timestamp}
              onSave={handleSave}
              onAddNote={handleAddNote}
              onExport={handleExport}
              onShare={handleShare}
            />

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
            <RiskOverview finalAssessment={result.final_assessment} />
            <EvidenceCoverage consensus={result.consensus} analysisId={result.analysis_id} />
          </div>

          {/* 6 AI Agents Grid */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Cpu size={18} color="var(--cyber-blue)" />
                <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
                  Individual Agent Assessments (6-Agent Swarm)
                </h3>
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                Harmonized via Operational Security Risk Calibration (OSRH-v2)
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
              {AGENT_KEYS.map((agentKey) => (
                <AgentCard
                  key={agentKey}
                  agentKey={agentKey}
                  data={result.agents?.[agentKey]}
                    onViewDetails={() => setSelectedAgent(agentKey)}
                  />
              ))}
            </div>
          </div>

          {/* Decision Rationale & Risk Factors */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '20px' }}>
            <FusionSummary consensus={result.consensus} finalAssessment={result.final_assessment} agents={result.agents} />
            <RiskFactors riskFactors={result.explanation?.risk_factors} />
          </div>

          <RiskDistribution score={result.final_assessment?.risk_score} />
          

          
          <ExplainabilityPanel result={result} />
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '20px' }}>
            <EvidenceExplorer agents={result.agents} />
            <InvestigationTimeline timestamp={result.timestamp} />
          </div>


          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '20px' }}>
            <AnalystNotes 
              initialNote={result.notes || ''}
              onSave={async (note) => {
                if (!result.analysis_id) return;
                await updateInvestigation(result.analysis_id, { notes: note });
              }} 
            />
            <InvestigationTags 
              initialTags={result.tags ? JSON.parse(result.tags) : []}
              onTagsChange={async (tags) => {
                if (!result.analysis_id) return;
                await updateInvestigation(result.analysis_id, { tags: tags });
              }} 
            />
          </div>


          <AgentDetailsModal 
            isOpen={!!selectedAgent} 
            agentKey={selectedAgent} 
            agentData={selectedAgent ? result.agents[selectedAgent] : null} 
            onClose={() => setSelectedAgent(null)} 
          />

          {/* Database Persistence Verification Note */}
          <div style={{
            padding: '12px 16px',
            borderRadius: '6px',
            backgroundColor: 'rgba(16, 185, 129, 0.08)',
            border: '1px solid rgba(16, 185, 129, 0.25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '12px',
            color: '#34d399'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Database size={15} />
              <span>Persisted to SQLite database (analyses & agent_results tables): <strong>{result.analysis_id}</strong></span>
            </div>
            <span style={{ fontSize: '11px', color: 'var(--text-faint)', fontFamily: 'monospace' }}>
              Transaction Committed
            </span>
          </div>
        </div>
      )}
    </div>
  );
}


