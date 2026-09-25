import React, { useState, useEffect } from 'react';
import { getInvestigations } from '../services/api';
import { FileText, Search, Download, ExternalLink } from 'lucide-react';
import { formatIST } from '../utils/dateUtils';


export default function ReportsPage({ onSelectReport }) {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    try {
      setLoading(true);
      // For now, any successful investigation can be viewed as a report
      const data = await getInvestigations({ status: 'success', limit: 50 });
      setReports(data.analyses || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: 600, color: '#f8fafc', margin: '0 0 8px 0' }}>Reports Library</h1>
        <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '14px' }}>View and export professional PDF security reports.</p>
      </div>

      <div style={{ backgroundColor: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--border-subtle)', overflow: 'hidden' }}>
        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--cyber-blue)' }}>Loading library...</div>
        ) : reports.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-faint)' }}>No reports available.</div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', backgroundColor: 'rgba(6, 9, 15, 0.4)' }}>
                <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Report ID</th>
                <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Target</th>
                <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Verdict</th>
                <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Date</th>
                <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((r) => (
                <tr key={r.analysis_id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '16px', color: '#f8fafc', fontSize: '13px', fontFamily: 'monospace' }}>RPT-{r.analysis_id.substring(0,8)}</td>
                  <td style={{ padding: '16px', color: '#f8fafc', fontSize: '14px' }}>{r.domain}</td>
                  <td style={{ padding: '16px', color: 'var(--text-muted)', fontSize: '13px' }}>{r.final_verdict ? r.final_verdict.toUpperCase() : 'UNKNOWN'}</td>
                  <td style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '13px' }}>{formatIST(r.created_at).split(' ')[0]}</td>
                  <td style={{ padding: '16px' }}>
                    <button 
                      onClick={() => onSelectReport(r.analysis_id)}
                      style={{ background: 'var(--cyber-blue)', border: 'none', color: '#06090f', padding: '6px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}
                    >
                      <FileText size={14} /> View Report
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}


