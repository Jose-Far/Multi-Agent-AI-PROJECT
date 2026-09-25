import React from 'react';

export default function InvestigationStats({ stats }) {
  if (!stats) return null;

  return (
    <>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '16px',
        marginBottom: '24px'
      }}>
        <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px' }}>
          <div style={{ fontSize: '13px', color: 'var(--text-faint)', textTransform: 'uppercase', marginBottom: '8px' }}>Total Investigations</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#f8fafc', fontFamily: 'monospace' }}>{stats.total_investigations || 0}</div>
        </div>
        
        <div style={{ backgroundColor: 'rgba(244, 63, 94, 0.05)', border: '1px solid rgba(244, 63, 94, 0.2)', borderRadius: '8px', padding: '20px' }}>
          <div style={{ fontSize: '13px', color: '#fb7185', textTransform: 'uppercase', marginBottom: '8px' }}>High Risk</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#f43f5e', fontFamily: 'monospace' }}>{stats.high_risk || 0}</div>
        </div>

        <div style={{ backgroundColor: 'rgba(244, 63, 94, 0.15)', border: '1px solid rgba(244, 63, 94, 0.4)', borderRadius: '8px', padding: '20px' }}>
          <div style={{ fontSize: '13px', color: '#fda4af', textTransform: 'uppercase', marginBottom: '8px' }}>Critical Risk</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#e11d48', fontFamily: 'monospace' }}>{stats.critical || 0}</div>
        </div>

        <div style={{ backgroundColor: 'rgba(56, 189, 248, 0.05)', border: '1px solid rgba(56, 189, 248, 0.2)', borderRadius: '8px', padding: '20px' }}>
          <div style={{ fontSize: '13px', color: 'var(--cyber-blue)', textTransform: 'uppercase', marginBottom: '8px' }}>Today</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#38bdf8', fontFamily: 'monospace' }}>{stats.today || 0}</div>
        </div>
      </div>

      {/* Phase 13 - Investigation Trend (Placeholder for Analytics) */}
      <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '20px', marginBottom: '24px' }}>
        <div style={{ fontSize: '13px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '16px' }}>Investigations Over Time (Last 7 Days)</div>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px', height: '120px' }}>
          {/* Simple CSS Bar Chart Simulation */}
          {(() => {
            const data = [12, 18, 5, 22, 14, 30, stats.today || 0];
            const maxVal = Math.max(...data, 1);
            return data.map((val, i) => {
              const dayLabel = i === 6 ? 'Today' : `${6 - i} days ago`;
              const hoverText = `${val} Investigations (${dayLabel})`;
              
              return (
                <div 
                  key={i} 
                  title={hoverText}
                  style={{ flex: 1, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', alignItems: 'center', gap: '6px', cursor: 'pointer' }}
                >
                  <div style={{ 
                    width: '100%', 
                    backgroundColor: i === 6 ? 'var(--cyber-blue)' : 'rgba(56, 189, 248, 0.2)', 
                    height: `calc(${Math.max(2, (val / maxVal) * 100)}% - 20px)`,
                    borderRadius: '4px 4px 0 0',
                    transition: 'all 0.2s ease-in-out'
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.backgroundColor = i === 6 ? '#7dd3fc' : 'rgba(56, 189, 248, 0.5)';
                    e.currentTarget.style.boxShadow = i === 6 ? '0 0 10px rgba(56, 189, 248, 0.5)' : 'none';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.backgroundColor = i === 6 ? 'var(--cyber-blue)' : 'rgba(56, 189, 248, 0.2)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                  ></div>
                  <span style={{ fontSize: '12px', color: i === 6 ? 'var(--cyber-blue)' : 'var(--text-faint)', fontWeight: i === 6 ? 600 : 400 }}>{val}</span>
                </div>
              );
            });
          })()}
        </div>
      </div>
    </>
  );
}
