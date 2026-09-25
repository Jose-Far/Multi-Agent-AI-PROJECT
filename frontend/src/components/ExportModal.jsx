import React from 'react';
import { X, FileJson, FileText, Download, FileCode } from 'lucide-react';
import { API_BASE_URL, getAuthHeaders } from '../services/api';

export default function ExportModal({ isOpen, onClose, result }) {
  if (!isOpen || !result) return null;

  const downloadFile = (content, filename, type) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportJSON = () => {
    downloadFile(JSON.stringify(result, null, 2), `phishdec_${result.analysis_id}.json`, 'application/json');
  };

  const handleExportTXT = () => {
    const txt = `PhishDec Investigation Report\nID: ${result.analysis_id}\nTarget: ${result.target?.url}\nVerdict: ${result.final_assessment?.verdict}\nRisk Score: ${result.final_assessment?.risk_score}/100\n\nAgents Summary:\n${Object.values(result.agents || {}).map(a => `- ${a.agent}: ${a.prediction} (Conf: ${a.confidence})`).join('\n')}`;
    downloadFile(txt, `phishdec_${result.analysis_id}.txt`, 'text/plain');
  };

  const handleExportCSV = () => {
    const csv = `Analysis ID,Target URL,Verdict,Risk Score,Confidence\n${result.analysis_id},${result.target?.url},${result.final_assessment?.verdict},${result.final_assessment?.risk_score},${result.final_assessment?.confidence}`;
    downloadFile(csv, `phishdec_${result.analysis_id}.csv`, 'text/csv');
  };

  const handleExportXML = () => {
    const xml = `<?xml version="1.0" encoding="UTF-8"?>
<investigation>
  <id>${result.analysis_id}</id>
  <target>${result.target?.url}</target>
  <verdict>${result.final_assessment?.verdict}</verdict>
  <risk_score>${result.final_assessment?.risk_score}</risk_score>
</investigation>`;
    downloadFile(xml, `phishdec_${result.analysis_id}.xml`, 'application/xml');
  };

  const handleExportPDF = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyses/${result.analysis_id}/report/pdf`, {
        headers: getAuthHeaders()
      });
      if (!response.ok) {
          const res2 = await fetch(`${API_BASE_URL}/analyses/${result.analysis_id}/report/pdf`, {
              headers: getAuthHeaders()
          });
          if (!res2.ok) throw new Error('PDF generation failed on server');
          const blob = await res2.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `phishdec_${result.analysis_id}.pdf`;
          a.click();
          URL.revokeObjectURL(url);
          return;
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `phishdec_${result.analysis_id}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert("Error generating PDF: " + e.message);
    }
  };

    return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.7)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center'
    }}>
      <div style={{
        backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-focus)',
        borderRadius: '8px', padding: '24px', width: '400px', maxWidth: '90%'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
          <h2 style={{ fontSize: '18px', fontWeight: 'bold' }}>Export Investigation</h2>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}><X size={20} /></button>
        </div>
        
        <style>{`
          .export-btn {
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100%;
            padding: 12px;
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            color: var(--text-main);
            cursor: pointer;
            text-align: left;
            margin-top: 10px;
            transition: all 0.2s ease;
          }
          .export-btn:hover {
            background-color: rgba(56, 189, 248, 0.1);
            border-color: var(--cyber-blue);
            transform: translateX(4px);
          }
        `}</style>
        <p style={{ color: 'var(--text-faint)', marginBottom: '20px', fontSize: '14px' }}>Choose a format to download the complete report for <b>{result.analysis_id}</b>:</p>
        
        <button className="export-btn" onClick={handleExportPDF}><Download size={18} color="#e11d48"/> Download PDF Report</button>
        <button className="export-btn" onClick={handleExportJSON}><FileJson size={18} color="#34d399"/> Download JSON Data</button>
        <button className="export-btn" onClick={handleExportCSV}><FileText size={18} color="#38bdf8"/> Download CSV Summary</button>
        <button className="export-btn" onClick={handleExportTXT}><FileText size={18} color="#94a3b8"/> Download Text Summary</button>
        <button className="export-btn" onClick={handleExportXML}><FileCode size={18} color="#fbbf24"/> Download XML Data</button>
      </div>

    </div>
  );
}
