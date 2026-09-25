import React, { useState, useEffect } from 'react';
import { getReportData, getPdfReportUrl } from '../services/api';
import { ChevronLeft, Download, AlertTriangle, Shield, CheckCircle } from 'lucide-react';
import { formatIST } from '../utils/dateUtils';


export default function ReportPreviewPage({ reportId, onBack }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!reportId) return;
    loadReport();
  }, [reportId]);

  const loadReport = async () => {
    try {
      setLoading(true);
      const data = await getReportData(reportId);
      setReport(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExportPDF = async () => {
    try {
      let response = await fetch(getPdfReportUrl(reportId), {
        headers: {
          'Authorization': `Bearer ${sessionStorage.getItem('phishdec_token')}`
        }
      });
      
      if (!response.ok && response.status === 404) {
        response = await fetch(getPdfReportUrl(reportId).replace('/api/analyses', '/analyses'), {
          headers: {
            'Authorization': `Bearer ${sessionStorage.getItem('phishdec_token')}`
          }
        });
      }

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.message || 'Failed to download PDF report.');
      }
      
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `phishdec_report_${reportId}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(err.message);
    }
  };

  if (loading) {
    return <div style={{ padding: '40px', textAlign: 'center', color: 'var(--cyber-blue)' }}>Building report...</div>;
  }

  if (error || !report) {
    return <div style={{ padding: '40px', textAlign: 'center', color: '#f43f5e' }}>{error || 'Failed to load report'}</div>;
  }

  const { metadata, summary, agents, fusion, evidence, notes, tags, technical } = report;

  const isPhishing = summary.verdict === 'PHISHING';
  const isUnknown = summary.verdict === 'UNKNOWN';

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto' }}>
      
      {/* Top Action Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <button 
          onClick={onBack}
          style={{
            background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', padding: 0
          }}
        >
          <ChevronLeft size={16} /> Back to Investigation
        </button>
        <button 
          onClick={handleExportPDF}
          style={{
            padding: '8px 16px',
            backgroundColor: 'var(--cyber-blue)',
            border: 'none',
            borderRadius: '6px',
            color: '#06090f',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: 600
          }}
        >
          <Download size={16} /> Export PDF
        </button>
      </div>

      {/* REPORT PREVIEW PAPER */}
      <div style={{
        backgroundColor: '#fff',
        borderRadius: '8px',
        padding: '60px',
        color: '#0f172a',
        boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
        marginBottom: '60px'
      }}>
        
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '40px' }}>
          <h1 style={{ margin: 0, fontSize: '32px', color: '#0f172a', fontWeight: 800, letterSpacing: '2px' }}>PHISHDEC</h1>
          <p style={{ margin: '8px 0', color: '#64748b', fontSize: '14px', letterSpacing: '1px' }}>MULTI-AGENT AI CYBERSECURITY ANALYST</p>
          <h2 style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '16px', marginTop: '24px', fontSize: '20px', color: '#0f172a' }}>SECURITY INVESTIGATION REPORT</h2>
        </div>

        {/* Metadata */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '40px', backgroundColor: '#f8fafc', padding: '24px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
          <div>
            <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>Target URL</div>
            <div style={{ fontSize: '15px', color: '#0f172a', wordBreak: 'break-all' }}>{metadata.url || metadata.target}</div>
          </div>
          <div>
            <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>Analysis ID</div>
            <div style={{ fontSize: '15px', color: '#0f172a', fontFamily: 'monospace' }}>{metadata.analysis_id}</div>
            
            <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', marginTop: '12px' }}>Date Generated</div>
            <div style={{ fontSize: '15px', color: '#0f172a' }}>{formatIST(metadata.date)}</div>
          </div>
        </div>

        {/* Assessment */}
        <h3 style={{ fontSize: '18px', color: '#0f172a', borderBottom: '1px solid #e2e8f0', paddingBottom: '8px', marginBottom: '16px' }}>1. Final Assessment</h3>
        <div style={{ 
          borderLeft: `4px solid ${isPhishing ? '#e11d48' : isUnknown ? '#64748b' : '#10b981'}`,
          backgroundColor: isPhishing ? 'rgba(225, 29, 72, 0.05)' : isUnknown ? '#f8fafc' : 'rgba(16, 185, 129, 0.05)',
          padding: '24px',
          marginBottom: '40px',
          borderRadius: '0 8px 8px 0'
        }}>
          <div style={{ fontSize: '24px', fontWeight: 800, color: isPhishing ? '#e11d48' : isUnknown ? '#64748b' : '#10b981', marginBottom: '16px' }}>
            {summary.verdict}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
            <div>
              <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase' }}>Risk Score</div>
              <div style={{ fontSize: '18px', fontWeight: 600, color: '#0f172a' }}>{summary.risk_score !== null ? `${Math.round(summary.risk_score)}/100` : '-'}</div>
            </div>
            <div>
              <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase' }}>Risk Level</div>
              <div style={{ fontSize: '18px', fontWeight: 600, color: '#0f172a' }}>{summary.risk_level}</div>
            </div>
            <div>
              <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase' }}>Confidence</div>
              <div style={{ fontSize: '18px', fontWeight: 600, color: '#0f172a' }}>{summary.confidence ? `${Math.round(summary.confidence * 100)}%` : '-'}</div>
            </div>
            <div>
              <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase' }}>Coverage</div>
              <div style={{ fontSize: '18px', fontWeight: 600, color: '#0f172a' }}>{summary.coverage}</div>
            </div>
          </div>
        </div>

        {/* Six Agents */}
        <h3 style={{ fontSize: '18px', color: '#0f172a', borderBottom: '1px solid #e2e8f0', paddingBottom: '8px', marginBottom: '16px' }}>2. Six-Agent Analysis</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '40px', fontSize: '14px' }}>
          <thead>
            <tr style={{ backgroundColor: '#f1f5f9', borderBottom: '2px solid #cbd5e1' }}>
              <th style={{ padding: '12px', textAlign: 'left', color: '#475569' }}>Agent</th>
              <th style={{ padding: '12px', textAlign: 'left', color: '#475569' }}>Prediction</th>
              <th style={{ padding: '12px', textAlign: 'left', color: '#475569' }}>Risk Score</th>
              <th style={{ padding: '12px', textAlign: 'left', color: '#475569' }}>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(agents || {}).map(([key, a]) => (
              <tr key={key} style={{ borderBottom: '1px solid #e2e8f0' }}>
                <td style={{ padding: '12px', fontWeight: 500, color: '#0f172a' }}>{key.replace('_', ' ').toUpperCase()}</td>
                <td style={{ padding: '12px', color: a.prediction ? '#0f172a' : '#94a3b8' }}>{a.prediction ? a.prediction.toUpperCase() : 'UNKNOWN'}</td>
                <td style={{ padding: '12px', color: '#0f172a' }}>{a.risk_score !== null ? Math.round(a.risk_score) : '-'}</td>
                <td style={{ padding: '12px', color: '#64748b' }}>{a.confidence !== null ? `${Math.round(a.confidence * 100)}%` : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Fusion */}
        <h3 style={{ fontSize: '18px', color: '#0f172a', borderBottom: '1px solid #e2e8f0', paddingBottom: '8px', marginBottom: '16px' }}>3. Decision Fusion</h3>
        <div style={{ backgroundColor: '#f8fafc', padding: '24px', borderRadius: '8px', border: '1px solid #e2e8f0', marginBottom: '40px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div><strong style={{ color: '#475569' }}>Consensus Available:</strong> {fusion.consensus_satisfied ? 'Yes' : 'No'}</div>
            <div><strong style={{ color: '#475569' }}>Evidence State:</strong> {fusion.evidence_state ? fusion.evidence_state.toUpperCase() : 'UNKNOWN'}</div>
            <div><strong style={{ color: '#475569' }}>Conflict Detected:</strong> {fusion.conflict_detected ? 'Yes' : 'No'}</div>
          </div>
          {fusion.conflict_detected && (
            <div style={{ marginTop: '16px', color: '#e11d48', fontWeight: 600 }}>
              ? Agent disagreement was observed. The final assessment was interpreted via consensus thresholds.
            </div>
          )}
        </div>

        {/* Evidence */}
        <h3 style={{ fontSize: '18px', color: '#0f172a', borderBottom: '1px solid #e2e8f0', paddingBottom: '8px', marginBottom: '16px' }}>4. Evidence</h3>
        <div style={{ marginBottom: '40px' }}>
          {Object.entries(evidence || {}).map(([agentName, evList]) => (
            <div key={agentName} style={{ marginBottom: '20px' }}>
              <div style={{ fontWeight: 600, color: '#334155', marginBottom: '8px' }}>{agentName.replace('_', ' ').toUpperCase()}</div>
              {evList && evList.length > 0 ? (
                <ul style={{ margin: 0, paddingLeft: '20px', color: '#475569', fontSize: '14px' }}>
                  {evList.map((item, i) => <li key={i}>{item}</li>)}
                </ul>
              ) : (
                <div style={{ color: '#94a3b8', fontSize: '14px', fontStyle: 'italic' }}>No notable evidence reported.</div>
              )}
            </div>
          ))}
        </div>

        {/* Notes & Tags */}
        <h3 style={{ fontSize: '18px', color: '#0f172a', borderBottom: '1px solid #e2e8f0', paddingBottom: '8px', marginBottom: '16px' }}>5. Analyst Notes & Tags</h3>
        <div style={{ marginBottom: '40px' }}>
          {tags && tags.length > 0 && (
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
              {tags.map(t => (
                <span key={t} style={{ backgroundColor: '#e2e8f0', color: '#334155', padding: '4px 8px', borderRadius: '4px', fontSize: '12px', fontWeight: 500 }}>
                  {t}
                </span>
              ))}
            </div>
          )}
          {notes ? (
            <div style={{ whiteSpace: 'pre-wrap', color: '#475569', fontSize: '14px', backgroundColor: '#f8fafc', padding: '16px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              {notes}
            </div>
          ) : (
            <div style={{ color: '#94a3b8', fontSize: '14px', fontStyle: 'italic' }}>No analyst notes.</div>
          )}
        </div>

      </div>
    </div>
  );
}

