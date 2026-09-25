import React, { useState } from 'react';
import { Tags, Plus, X } from 'lucide-react';
import { useAuth } from '../services/AuthContext';

export default function InvestigationTags({ initialTags = [], onTagsChange }) {
  const { user } = useAuth();
  const isViewer = user?.role === 'viewer';
  const [tags, setTags] = useState(initialTags);
  const [inputValue, setInputValue] = useState('');

  const addTag = (e) => {
    if (e.key === 'Enter' || e.type === 'click') {
      e.preventDefault();
      const newTag = inputValue.trim();
      if (newTag && !tags.includes(newTag)) {
        const newTags = [...tags, newTag];
        setTags(newTags);
        if (onTagsChange) onTagsChange(newTags);
      }
      setInputValue('');
    }
  };

  const removeTag = (tagToRemove) => {
    const newTags = tags.filter(t => t !== tagToRemove);
    setTags(newTags);
    if (onTagsChange) onTagsChange(newTags);
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
        <Tags size={18} color="var(--cyber-blue)" />
        <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
          Investigation Tags
        </h3>
      </div>
      
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '16px' }}>
        {tags.map((tag, i) => (
          <div key={i} style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            border: '1px solid rgba(56, 189, 248, 0.2)',
            color: 'var(--cyber-blue)',
            padding: '4px 10px',
            borderRadius: '16px',
            fontSize: '12px'
          }}>
            {tag}
            {!isViewer && (
              <button onClick={() => removeTag(tag)} style={{
                background: 'none', border: 'none', color: 'var(--cyber-blue)', 
                cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center'
              }}>
                <X size={12} />
              </button>
            )}
          </div>
        ))}
      </div>
      
      {!isViewer && (
        <div style={{ display: 'flex', gap: '8px' }}>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={addTag}
            placeholder="Add a tag... (e.g. Credential Theft)"
            style={{
              flex: 1,
              backgroundColor: 'rgba(6, 9, 15, 0.5)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '4px',
              padding: '8px 12px',
              color: '#f8fafc',
              fontSize: '13px',
              outline: 'none'
            }}
          />
          <button onClick={addTag} style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            backgroundColor: 'rgba(255,255,255,0.05)',
            border: '1px solid var(--border-subtle)',
            color: 'var(--text-muted)',
            padding: '0 12px',
            borderRadius: '4px',
            cursor: 'pointer'
          }}>
            <Plus size={16} />
          </button>
        </div>
      )}
    </div>
  );
}
