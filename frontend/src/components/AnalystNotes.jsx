import React, { useState } from 'react';
import { FileText, Save, CheckCircle2 } from 'lucide-react';
import { useAuth } from '../services/AuthContext';

export default function AnalystNotes({ initialNote = '', onSave }) {
  const { user } = useAuth();
  const isViewer = user?.role === 'viewer';
  const [note, setNote] = useState(initialNote);
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      if (onSave) await onSave(note);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      console.error("Failed to save note:", err);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div style={{
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: '8px',
      padding: '24px',
      marginTop: '24px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <FileText size={18} color="var(--cyber-blue)" />
        <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
          Analyst Notes
        </h3>
      </div>
      
      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        disabled={isViewer}
        placeholder={isViewer ? "No notes available." : "Add investigation notes... (e.g. 'Potential credential harvesting page. Threat intelligence should be reviewed.')"}
        style={{
          width: '100%',
          minHeight: '100px',
          backgroundColor: 'rgba(6, 9, 15, 0.5)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '6px',
          padding: '12px',
          color: '#f8fafc',
          fontSize: '13px',
          fontFamily: 'inherit',
          resize: 'vertical',
          marginBottom: '16px',
          opacity: isViewer ? 0.7 : 1
        }}
      />
      
      {!isViewer && (
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={handleSave}
            disabled={isSaving}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              backgroundColor: saved ? 'rgba(16, 185, 129, 0.15)' : 'rgba(56, 189, 248, 0.15)',
              border: `1px solid `,
              color: saved ? '#34d399' : 'var(--cyber-blue)',
              padding: '8px 16px',
              borderRadius: '4px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
          >
            {saved ? <CheckCircle2 size={16} /> : <Save size={16} />}
            {isSaving ? 'Saving...' : saved ? 'Saved' : 'Save Note'}
          </button>
        </div>
      )}
    </div>
  );
}
