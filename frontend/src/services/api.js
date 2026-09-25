export const API_BASE_URL = 'http://127.0.0.1:5050';

export function getAuthHeaders() {
  const token = sessionStorage.getItem('phishdec_token');
  const headers = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

export async function checkHealth() {
  try {
    const res = await fetch(API_BASE_URL + '/health', {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store'
    });
    if (!res.ok) throw new Error('HTTP error ' + res.status);
    return await res.json();
  } catch (err) {
    console.error('Health check failed:', err);
    return { status: 'unhealthy', error: err.message };
  }
}

export async function analyzeUrl(url, urlOnly = false) {
  const payload = { url: url.trim() };
  if (urlOnly) {
    payload.url_only = true;
  }

  const res = await fetch(API_BASE_URL + '/analyze', {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.message || data.error || 'Analysis failed (' + res.status + ')');
  }
  return data;
}

export async function getInvestigations(params = {}) {
  const query = new URLSearchParams();
  if (params.limit) query.append('limit', params.limit);
  if (params.offset) query.append('offset', params.offset);
  if (params.search) query.append('search', params.search);
  if (params.verdict && params.verdict !== 'ALL') query.append('verdict', params.verdict);
  if (params.risk_level && params.risk_level !== 'ALL') query.append('risk_level', params.risk_level);
  if (params.status && params.status !== 'ALL') query.append('status', params.status);
  if (params.saved_only) query.append('saved_only', 'true');
  if (params.date_filter && params.date_filter !== 'ALL') query.append('date_filter', params.date_filter);
  if (params.sort) query.append('sort', params.sort);

  const res = await fetch(`${API_BASE_URL}/analyses?${query.toString()}`, {
    method: 'GET',
    headers: getAuthHeaders(),
  });
  
  const data = await res.json();
  if (!res.ok) throw new Error(data.message || 'Failed to fetch investigations');
  return data;
}

export async function getInvestigationDetail(analysisId) {
  const res = await fetch(API_BASE_URL + '/analyses/' + encodeURIComponent(analysisId), {
    method: 'GET',
    headers: getAuthHeaders(),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.message || 'Investigation ' + analysisId + ' not found');
  }
  return data;
}

export async function deleteInvestigation(analysisId) {
  const res = await fetch(API_BASE_URL + '/analyses/' + encodeURIComponent(analysisId), {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.message || 'Failed to delete ' + analysisId);
  }
  return data;
}

export async function updateInvestigation(analysisId, data) {
  const res = await fetch(API_BASE_URL + '/analyses/' + encodeURIComponent(analysisId), {
    method: 'PATCH',
    headers: getAuthHeaders(),
    body: JSON.stringify(data),
  });

  const responseData = await res.json();
  if (!res.ok) {
    throw new Error(responseData.message || 'Failed to update ' + analysisId);
  }
  return responseData;
}

export async function deleteAnalysis(id) {
  const res = await fetch(`${API_BASE_URL}/analyses/${id}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.message || 'Failed to delete');
  return data;
}

export async function getReportData(id) {
  const res = await fetch(`${API_BASE_URL}/analyses/${id}/report`, {
    headers: getAuthHeaders(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.message || 'Failed to fetch report data');
  return data.report;
}

export function getPdfReportUrl(id) {
  return `${API_BASE_URL}/analyses/${id}/report/pdf`;
}

export const getSystemHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/admin/system`, {
      headers: getAuthHeaders(),
    });
    if (!response.ok) throw new Error('API Error');
    return await response.json();
  } catch (error) {
    console.error("System health error:", error);
    throw error;
  }
};

export const getAdminAnalytics = async (period = 30) => {
  const response = await fetch(`${API_BASE_URL}/api/admin/analytics?period=${period}`, {
    headers: getAuthHeaders(),
  });
  if (!response.ok) throw new Error("Failed to fetch analytics");
  return await response.json();
};

export const getAdminMonitoring = async () => {
  const response = await fetch(`${API_BASE_URL}/api/admin/monitoring`, {
    headers: getAuthHeaders(),
  });
  if (!response.ok) throw new Error("Failed to fetch monitoring");
  return await response.json();
};

export const getAdminUsers = async () => {
  const response = await fetch(`${API_BASE_URL}/api/admin/users`, {
    headers: getAuthHeaders(),
    cache: 'no-store'
  });
  if (response.status === 401) {
    sessionStorage.removeItem('phishdec_token');
    window.location.href = '/';
  }
  if (!response.ok) throw new Error("Failed to fetch users");
  return await response.json();
};

export const createAdminUser = async (userData) => {
  const response = await fetch(`${API_BASE_URL}/api/admin/users`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify(userData),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || "Failed to create user");
  return data;
};

export const updateAdminUser = async (userId, updateData) => {
  const response = await fetch(`${API_BASE_URL}/api/admin/users/${userId}`, {
    method: 'PATCH',
    headers: getAuthHeaders(),
    body: JSON.stringify(updateData),
  });
  if (response.status === 401) {
    sessionStorage.removeItem('phishdec_token');
    window.location.href = '/';
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || "Failed to update user");
  return data;
};

export const resetUserPassword = async (userId, newPassword) => {
  const response = await fetch(`${API_BASE_URL}/api/admin/users/${userId}/reset-password`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ new_password: newPassword }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || "Failed to reset password");
  return data;
};


export const deleteAdminUser = async (userId) => {
  const response = await fetch(`${API_BASE_URL}/api/admin/users/${userId}`, {
    method: 'DELETE',
    headers: getAuthHeaders()
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || 'Failed to delete user');
  return data;
};

export const getAuditLogs = async (limit=100) => {
  const response = await fetch(`${API_BASE_URL}/api/admin/audit-logs?limit=${limit}`, {
    headers: getAuthHeaders()
  });
  if (!response.ok) throw new Error('Failed to fetch audit logs');
  return await response.json();
};
