import React, { useState, useEffect } from 'react';
import './AdminApp.css';
import AdminSidebar from './components/admin/AdminSidebar';
import AdminHeader from './components/admin/AdminHeader';
import AdminOverviewPage from './pages/admin/AdminOverviewPage';
import AdminAgentsPage from './pages/admin/AdminAgentsPage';
import AdminModelsPage from './pages/admin/AdminModelsPage';
import AdminHealthPage from './pages/admin/AdminHealthPage';
import AdminLogsPage from './pages/admin/AdminLogsPage';
import AdminAlertsPage from './pages/admin/AdminAlertsPage';
import AdminUsersPage from './pages/admin/AdminUsersPage';
import AdminSettingsPage from './pages/admin/AdminSettingsPage';
import AdminAnalyticsPage from './pages/admin/AdminAnalyticsPage';
import AdminMonitoringPage from './pages/admin/AdminMonitoringPage';
import NotificationHistory from './components/NotificationHistory';
import { getSystemHealth } from './services/api';

export default function AdminApp() {
  const [currentTab, setTab] = useState('overview');
  const [healthData, setHealthData] = useState(null);
  const [apiOnline, setApiOnline] = useState(true);

  const refreshHealth = async () => {
    try {
      const res = await getSystemHealth();
      setHealthData(res);
      setApiOnline(res.api?.status === 'online');
    } catch {
      setApiOnline(false);
      setHealthData(null);
    }
  };

  useEffect(() => {
    refreshHealth();
    const interval = setInterval(refreshHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="admin-theme">
      <div className="app-container">
        <AdminSidebar currentTab={currentTab} setTab={setTab} />
        
        <div className="main-content">
          <AdminHeader 
            currentTab={currentTab} 
            apiOnline={apiOnline} 
            onRefresh={refreshHealth}
            onNavigate={(t) => setTab(t)}
          />
          
          <main className="content-body">
            {currentTab === 'overview' && <AdminOverviewPage healthData={healthData} apiOnline={apiOnline} />}
            {currentTab === 'agents' && <AdminAgentsPage healthData={healthData} />}
            {currentTab === 'models' && <AdminModelsPage healthData={healthData} />}
            {currentTab === 'health' && <AdminHealthPage healthData={healthData} apiOnline={apiOnline} />}
            {currentTab === 'analytics' && <AdminAnalyticsPage />}
            {currentTab === 'monitoring' && <AdminMonitoringPage />}
            {currentTab === 'notifications' && <NotificationHistory role="admin" />}
            {currentTab === 'logs' && <AdminLogsPage />}
            {currentTab === 'alerts' && <AdminAlertsPage />}
            {currentTab === 'users' && <AdminUsersPage />}
            {currentTab === 'settings' && <AdminSettingsPage />}
          </main>
        </div>
      </div>
    </div>
  );
}

