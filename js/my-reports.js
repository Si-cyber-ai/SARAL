/* ═══════════════════════════════════════════════════
   SARAL — My Reports Page Logic
   API-backed reports, filtering, real status
   ═══════════════════════════════════════════════════ */

(function () {
  'use strict';

  const state = SaralStore.get();
  let activeFilter = 'all';
  let reportsLoaded = false;

  // ─── Sidebar Toggle (mobile) ───
  const toggle = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('sidebar');
  if (toggle && sidebar) {
    toggle.addEventListener('click', () => sidebar.classList.toggle('is-open'));
  }

  // ─── Scroll Reveal ───
  const observer = new IntersectionObserver(
    (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('is-visible'); }),
    { threshold: 0.1 }
  );
  document.querySelectorAll('[data-reveal]').forEach(el => observer.observe(el));

  // ─── Render Reports ───
  const grid = document.getElementById('reportsGrid');
  const emptyEl = document.getElementById('reportsEmpty');

  function formatDate(iso) {
    const d = new Date(iso);
    const now = new Date();
    const diff = now - d;
    const hours = Math.floor(diff / 3600000);
    if (hours < 1) return 'Just now';
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
  }

  function statusMeta(status) {
    const map = {
      pending: { label: 'Under Review', cls: 'pending', icon: '⏳', color: 'amber' },
      verified: { label: 'Approved', cls: 'verified', icon: '✓', color: 'green' },
      resolved: { label: 'Approved', cls: 'resolved', icon: '✓', color: 'green' },
      rejected: { label: 'Rejected', cls: 'rejected', icon: '✗', color: 'red' },
      'Auto-Rejected': { label: 'Auto-Rejected', cls: 'rejected', icon: '✗', color: 'red' },
    };
    return map[status] || map.pending;
  }

  function violationIcon(type) {
    const lcType = (type || '').toLowerCase();
    if (lcType.includes('signal')) return '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><rect x="7" y="1" width="6" height="18" rx="2" stroke="currentColor" stroke-width="1.5"/><circle cx="10" cy="5" r="1.5" fill="currentColor"/><circle cx="10" cy="10" r="1.5" fill="currentColor"/><circle cx="10" cy="15" r="1.5" fill="currentColor"/></svg>';
    if (lcType.includes('speed')) return '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M10 18a8 8 0 100-16 8 8 0 000 16z" stroke="currentColor" stroke-width="1.5"/><path d="M10 6v4l3 2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
    if (lcType.includes('parking')) return '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><rect x="3" y="3" width="14" height="14" rx="3" stroke="currentColor" stroke-width="1.5"/><path d="M8 14V6h3a3 3 0 010 6H8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    if (lcType.includes('wrong')) return '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M10 2v16M6 6l4-4 4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M4 14h12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
    if (lcType.includes('helmet')) return '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M4 12a6 6 0 1112 0" stroke="currentColor" stroke-width="1.5"/><path d="M3 12h14v2a2 2 0 01-2 2H5a2 2 0 01-2-2v-2z" stroke="currentColor" stroke-width="1.5"/></svg>';
    if (lcType.includes('lane')) return '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M10 2v16M4 2v16M16 2v16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-dasharray="3 3"/></svg>';
    return '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M10 3v5M10 12h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="10" cy="10" r="8" stroke="currentColor" stroke-width="1.5"/></svg>';
  }

  function buildReportCard(report) {
    const meta = statusMeta(report.status);
    const card = document.createElement('div');
    card.className = 'report-card';
    card.dataset.id = report.id;
    card.dataset.status = report.status;

    const isParking = report.source === 'parking';

    // Plate badge
    const plateStr = report.plate
      ? `<span class="report-card__plate" style="font-family:monospace;font-size:12px;font-weight:600;color:#1e40af;background:rgba(79,140,255,.08);padding:3px 8px;border-radius:6px;margin-top:4px;display:inline-block;">${report.plate}</span>`
      : '';

    // Parking-specific detail pills (vehicle type + Parking AI badge)
    const parkingDetails = isParking ? `
      <div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:6px;">
        ${report.vehicle_type ? `<span style="display:inline-flex;align-items:center;gap:3px;background:rgba(139,92,246,.1);color:#5b21b6;font-size:10px;font-weight:700;padding:2px 7px;border-radius:5px;border:1px solid rgba(139,92,246,.2);">🚙 ${report.vehicle_type}</span>` : ''}
        <span style="display:inline-flex;align-items:center;gap:3px;background:rgba(249,115,22,.1);color:#c2410c;font-size:9px;font-weight:800;padding:1px 6px;border-radius:4px;border:1px solid rgba(249,115,22,.22);text-transform:uppercase;letter-spacing:.5px;">🅿 Parking AI</span>
      </div>
    ` : '';

    const thumbnailHtml = report.thumbnail
      ? `<img src="${SaralAPI.mediaUrl(report.thumbnail)}" style="width:100%;height:120px;object-fit:cover;border-radius:12px;margin-bottom:12px;" alt="Evidence" />`
      : '';

    // Karma badge shown once approved
    const karmaBadge = (report.status === 'verified' && report.points > 0)
      ? `<span class="report-card__points">+${report.points} pts</span>`
      : '';

    card.innerHTML = `
      ${thumbnailHtml}
      <div class="report-card__top">
        <div class="report-card__icon report-card__icon--${meta.color}">
          ${violationIcon(report.type)}
        </div>
        <div class="report-card__meta">
          <span class="report-card__type">${report.type}</span>
          <span class="report-card__id">RPT-${String(report.id).padStart(4, '0')}</span>
          ${plateStr}
          ${parkingDetails}
        </div>
        <span class="report-card__status report-card__status--${meta.cls}">
          ${meta.label}
        </span>
      </div>
      <div class="report-card__body">
        <div class="report-card__detail">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 1.75a4.375 4.375 0 00-4.375 4.375C2.625 9.5 7 12.25 7 12.25s4.375-2.75 4.375-6.125A4.375 4.375 0 007 1.75z" stroke="currentColor" stroke-width="1.2"/><circle cx="7" cy="6.125" r="1.5" stroke="currentColor" stroke-width="1.2"/></svg>
          <span>${report.location || 'Location not specified'}</span>
        </div>
        <div class="report-card__detail">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1.75" y="2.625" width="10.5" height="9.625" rx="1.5" stroke="currentColor" stroke-width="1.2"/><path d="M1.75 5.25h10.5M4.375 1.75v1.75M9.625 1.75v1.75" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>
          <span>${formatDate(report.date)}</span>
        </div>
      </div>
      <div class="report-card__footer">
        <div class="report-card__confidence">
          <div class="report-card__conf-bar">
            <div class="report-card__conf-fill" style="width:${report.confidence}%"></div>
          </div>
          <span class="report-card__conf-label">AI Confidence: ${report.confidence}%</span>
        </div>
        ${karmaBadge}
      </div>
    `;

    return card;
  }

  function renderReports() {
    const reports = state.reports;
    const filtered = activeFilter === 'all'
      ? reports
      : reports.filter(r => r.status === activeFilter);

    grid.innerHTML = '';

    if (filtered.length === 0) {
      emptyEl.style.display = 'flex';
    } else {
      emptyEl.style.display = 'none';
      filtered.forEach((r, i) => {
        const card = buildReportCard(r);
        card.style.animationDelay = `${i * 0.06}s`;
        card.classList.add('report-card--animate');
        grid.appendChild(card);
      });
    }

    updateCounts();
  }

  function updateCounts() {
    const reports = state.reports;
    const totalEl = document.getElementById('totalCount');
    const pendingEl = document.getElementById('pendingCount');
    const verifiedEl = document.getElementById('verifiedCount');
    const resolvedEl = document.getElementById('resolvedCount');
    if (totalEl) totalEl.textContent = reports.length;
    if (pendingEl) pendingEl.textContent = reports.filter(r => r.status === 'pending').length;
    if (verifiedEl) verifiedEl.textContent = reports.filter(r => r.status === 'verified').length;
    if (resolvedEl) resolvedEl.textContent = reports.filter(r => r.status === 'verified' || r.status === 'resolved').length;
  }

  // ─── Filter Buttons ───
  document.querySelectorAll('.reports-filter__btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.reports-filter__btn').forEach(b => b.classList.remove('reports-filter__btn--active'));
      btn.classList.add('reports-filter__btn--active');
      activeFilter = btn.dataset.filter;
      renderReports();
    });
  });

  // ─── Load Reports from API ───
  async function loadReports() {
    const userId = SaralAuth.getUserId();
    if (!userId) return;

    try {
      const data = await SaralAPI.getUserReports(userId);
      if (data.reports) {
        SaralStore.syncReports(data.reports);
        renderReports();
      }
    } catch (err) {
      console.error('[MyReports] Failed to load:', err);
      SaralToast.show('Failed to load reports: ' + err.message, 'error');
      // Render whatever is in state
      renderReports();
    }
  }

  // ─── Update sidebar profile ───
  async function updateSidebar() {
    await SaralAuth.refreshUser();
    const user = SaralAuth.getUser();
    const s = SaralStore.get();
    const profileName = document.querySelector('.sidebar__profile-name');
    const profileAvatar = document.querySelector('.sidebar__profile-avatar span');
    const profileBadge = document.querySelector('.sidebar__profile-badge');
    if (profileName) profileName.textContent = user.name;
    if (profileAvatar) profileAvatar.textContent = s.user.initials;
    // Compute tier from live karma
    try {
      const userId = SaralAuth.getUserId();
      const stats = await SaralAPI.getUserStats(userId);
      if (profileBadge) {
        const { current } = SaralStore.getTierInfo(stats.karma_points);
        profileBadge.innerHTML = `<svg width="10" height="10" viewBox="0 0 10 10" fill="none"><circle cx="5" cy="5" r="4" fill="${current.color}"/></svg> ${current.icon} ${current.name} Champion`;
      }
    } catch (_) { /* keep static badge */ }
  }

  // ─── Init ───
  updateSidebar();
  loadReports();

})();
