import React from 'react';
import { formatIST } from '../utils/dateUtils';
import { Shield, Clock, ExternalLink, Globe, Database, FileText, Share2, BookmarkPlus } from 'lucide-react';

export default function AnalysisHeader({ target, timestamp, analysisId, onSave, onAddNote, onExport, onShare }) {
  if (!target) return null;

  const date = timestamp ? new Date(timestamp) : new Date();
  const formattedDate = formatIST(date);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '20px',
      marginBottom: '24px',
      borderBottom: '1px solid var(--border-subtle)',
      paddingBottom: '20px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Shield size={20} color="var(--cyber-blue)" />
            <h2 style={{ fontSize: '18px', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
              PHISHDEC INVESTIGATION
            </h2>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '20px', fontWeight: 700, color: '#fff', marginBottom: '12px' }}>
            <Globe size={18} color="var(--text-muted)" />
            {target.url || target}
            <a href={target.url || target} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--cyber-blue)', marginLeft: '8px' }}>
              <ExternalLink size={16} />
            </a>
          </div>

          <div style={{ display: 'flex', gap: '24px', fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Database size={14} />
              ID: {analysisId || 'PD-UNSAVED'}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Clock size={14} />
              Analyzed: {formattedDate}
            </div>
          </div>
        </div>

        {/* Day 19: Investigation Actions */}
        
    <style>{`
      .action-btn-header {
        display: flex;
        align-items: center;
        gap: 6px;
        background-color: transparent;
        border: 1px solid var(--border-subtle);
        color: var(--text-muted);
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
      }
      .action-btn-header:hover {
        background-color: rgba(255, 255, 255, 0.05);
        color: var(--text-main);
        border-color: var(--text-muted);
      }
      .action-btn-primary {
        border-color: var(--cyber-blue);
        color: var(--cyber-blue);
        font-weight: 600;
      }
      .action-btn-primary:hover {
        background-color: rgba(56, 189, 248, 0.1);
        border-color: #7dd3fc;
        color: #7dd3fc;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
        transform: translateY(-1px);
      }
    `}</style>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={onSave} className="action-btn-header action-btn-primary">
            <BookmarkPlus size={14} /> Save Investigation
          </button>
          <button onClick={onAddNote} className="action-btn-header">
            <FileText size={14} /> Add Note
          </button>
          <button onClick={onExport} className="action-btn-header">
            <FileText size={14} /> Export
          </button>
          <button onClick={onShare} className="action-btn-header">
            <Share2 size={14} /> Share
          </button>
        </div>
      </div>
    </div>
  );
}

