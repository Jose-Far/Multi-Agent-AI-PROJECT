import React from 'react';
import { Search } from 'lucide-react';

export default function InvestigationFilters({ filters, onFilterChange }) {
  const handleChange = (e) => {
    onFilterChange({ ...filters, [e.target.name]: e.target.value });
  };

  const selectStyle = {
    backgroundColor: 'rgba(6, 9, 15, 0.5)',
    border: '1px solid var(--border-subtle)',
    color: '#f8fafc',
    padding: '8px 12px',
    borderRadius: '4px',
    fontSize: '13px',
    outline: 'none',
    cursor: 'pointer'
  };

  return (
    <div style={{
      display: 'flex',
      flexWrap: 'wrap',
      gap: '12px',
      marginBottom: '24px',
      alignItems: 'center',
      backgroundColor: 'var(--bg-card)',
      padding: '16px',
      borderRadius: '8px',
      border: '1px solid var(--border-subtle)'
    }}>
      <div style={{ position: 'relative', flexGrow: 1, minWidth: '250px' }}>
        <Search size={16} color="var(--text-faint)" style={{ position: 'absolute', left: '12px', top: '10px' }} />
        <input 
          type="text" 
          name="search"
          value={filters.search}
          onChange={handleChange}
          placeholder="Search by URL, Domain, Tags, or ID..."
          style={{
            width: '100%',
            backgroundColor: 'rgba(6, 9, 15, 0.5)',
            border: '1px solid var(--border-subtle)',
            color: '#f8fafc',
            padding: '8px 12px 8px 36px',
            borderRadius: '4px',
            fontSize: '13px',
            outline: 'none'
          }}
        />
      </div>

      <select name="verdict" value={filters.verdict} onChange={handleChange} style={selectStyle}>
        <option value="ALL">Verdict: All</option>
        <option value="legitimate">Legitimate</option>
        <option value="suspicious">Suspicious</option>
        <option value="phishing">Phishing</option>
        <option value="unknown">Unknown</option>
      </select>

      <select name="risk_level" value={filters.risk_level} onChange={handleChange} style={selectStyle}>
        <option value="ALL">Risk: All</option>
        <option value="low">Low</option>
        <option value="medium">Medium</option>
        <option value="high">High</option>
        <option value="critical">Critical</option>
        <option value="unknown">Unknown</option>
      </select>

      <select name="status" value={filters.status} onChange={handleChange} style={selectStyle}>
        <option value="ALL">Status: All</option>
        <option value="success">Completed</option>
        <option value="limited_evidence">Partial</option>
        <option value="failed">Failed</option>
      </select>

      <select name="date_filter" value={filters.date_filter} onChange={handleChange} style={selectStyle}>
        <option value="ALL">Date: All Time</option>
        <option value="today">Today</option>
        <option value="yesterday">Yesterday</option>
        <option value="last7days">Last 7 Days</option>
        <option value="last30days">Last 30 Days</option>
      </select>
      
      <select name="saved_only" value={filters.saved_only ? 'true' : 'false'} onChange={(e) => onFilterChange({...filters, saved_only: e.target.value === 'true'})} style={selectStyle}>
        <option value="false">Saved: All</option>
        <option value="true">? Saved Only</option>
      </select>

      <select name="sort" value={filters.sort} onChange={handleChange} style={{...selectStyle, borderLeft: '2px solid var(--cyber-blue)'}}>
        <option value="newest">Sort: Newest</option>
        <option value="oldest">Sort: Oldest</option>
        <option value="risk_high">Sort: Highest Risk</option>
        <option value="risk_low">Sort: Lowest Risk</option>
        <option value="conf_high">Sort: Highest Confidence</option>
        <option value="conf_low">Sort: Lowest Confidence</option>
      </select>
    </div>
  );
}
