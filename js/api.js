/* ═══════════════════════════════════════════════════
   SARAL — API Client
   Centralized API communication layer
   ═══════════════════════════════════════════════════ */

const SaralAPI = (() => {
  // Base URL for the FastAPI backend
  const BASE_URL = 'http://localhost:8000';

  async function request(endpoint, options = {}) {
    const url = `${BASE_URL}${endpoint}`;
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      if (err.message === 'Failed to fetch') {
        console.error('[SARAL API] Backend not reachable at', BASE_URL);
        throw new Error('Backend server is not running. Start it with: cd backend && python main.py');
      }
      throw err;
    }
  }

  function formData(obj) {
    const fd = new FormData();
    for (const [key, value] of Object.entries(obj)) {
      if (value !== undefined && value !== null) {
        fd.append(key, value);
      }
    }
    return fd;
  }

  return {
    // ─── Auth ───
    signIn(email, password, role) {
      return request('/api/auth/signin', {
        method: 'POST',
        body: formData({ email, password, role }),
      });
    },

    signUp(name, email, password, role) {
      return request('/api/auth/signup', {
        method: 'POST',
        body: formData({ name, email, password, role }),
      });
    },

    getUser(userId) {
      return request(`/api/auth/user/${userId}`);
    },

    updateProfile(userId, name, email, city, password) {
      const data = { name, email, city };
      if (password) data.password = password;
      return request(`/api/auth/user/${userId}`, {
        method: 'PUT',
        body: formData(data),
      });
    },

    // ─── Analyze (Upload + AI) ───
    analyze(file, userId, violationType, location, description, manualPlate) {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('user_id', userId);
      if (violationType) fd.append('violation_type', violationType);
      if (location) fd.append('location', location);
      if (description) fd.append('description', description);
      if (manualPlate) fd.append('manual_plate', manualPlate);
      return request('/api/analyze', { method: 'POST', body: fd });
    },

    // ─── Reports ───
    getUserReports(userId) {
      return request(`/api/reports/user/${userId}`);
    },

    getUserStats(userId) {
      return request(`/api/reports/stats/${userId}`);
    },

    getPendingReports() {
      return request('/api/reports/pending');
    },

    getAllReports() {
      return request('/api/reports/all');
    },

    getReport(reportId) {
      return request(`/api/reports/${reportId}`);
    },

    updateManualPlate(reportId, manualPlate) {
      const fd = new FormData();
      fd.append('manual_plate', manualPlate);
      return request(`/api/reports/${reportId}/plate`, { method: 'PATCH', body: fd });
    },

    updateReportDetails(reportId, { location, description, violationType, manualPlate } = {}) {
      const fd = new FormData();
      if (location) fd.append('location', location);
      if (description) fd.append('description', description);
      if (violationType) fd.append('violation_type', violationType);
      if (manualPlate) fd.append('manual_plate', manualPlate);
      return request(`/api/reports/${reportId}/details`, { method: 'PATCH', body: fd });
    },

    // ─── Authority Actions ───
    approveReport(reportId) {
      return request(`/api/reports/${reportId}/approve`, { method: 'POST' });
    },

    rejectReport(reportId) {
      return request(`/api/reports/${reportId}/reject`, { method: 'POST' });
    },

    // ─── Authority Stats ───
    getAuthorityStats() {
      return request('/api/authority/stats');
    },

    // ─── Rewards ───
    getRewardsCatalogue() {
      return request('/api/rewards/catalogue');
    },

    redeemReward(userId, rewardId) {
      return request('/api/rewards/redeem', {
        method: 'POST',
        body: formData({ user_id: userId, reward_id: rewardId }),
      });
    },

    getRewardsHistory(userId) {
      return request(`/api/rewards/history/${userId}`);
    },

    // ─── Health ───
    health() {
      return request('/api/health');
    },

    // ─── Utility ───
    mediaUrl(path) {
      if (!path) return '';
      if (path.startsWith('http')) return path;
      return `${BASE_URL}${path}`;
    },
  };
})();
