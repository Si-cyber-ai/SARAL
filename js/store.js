/* ═══════════════════════════════════════════════════
   SARAL — Global State Store & Toast System
   API-backed state — no dummy data
   ═══════════════════════════════════════════════════ */

const SaralStore = (() => {
  const STORAGE_KEY = 'saral_state';

  const defaultState = {
    user: {
      id: null,
      name: '',
      initials: '',
      email: '',
      city: '',
      tier: 'Bronze',
      points: 0,
      role: 'user',
    },
    reports: [],
    rewards: {
      redeemed: [],
      catalogue: [],
    },
    settings: {
      blurPlates: true,
      hideIdentity: false,
      emailNotifs: true,
      rewardAlerts: true,
    },
  };

  function load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        return deepMerge(defaultState, saved);
      }
    } catch (e) { /* ignore */ }
    return JSON.parse(JSON.stringify(defaultState));
  }

  function save(state) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) { /* ignore */ }
  }

  function deepMerge(target, source) {
    const out = { ...target };
    for (const key of Object.keys(source)) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        out[key] = deepMerge(target[key] || {}, source[key]);
      } else {
        out[key] = source[key];
      }
    }
    return out;
  }

  const TIERS = [
    { name: 'Bronze', min: 0, max: 500, color: '#cd7f32', icon: '🥉' },
    { name: 'Silver', min: 500, max: 1000, color: '#94a3b8', icon: '🥈' },
    { name: 'Gold', min: 1000, max: 1500, color: '#FFD700', icon: '🥇' },
    { name: 'Platinum', min: 1500, max: 2500, color: '#a78bfa', icon: '💎' },
    { name: 'Diamond', min: 2500, max: Infinity, color: '#38bdf8', icon: '💠' },
  ];

  function computeTier(points) {
    for (let i = TIERS.length - 1; i >= 0; i--) {
      if (points >= TIERS[i].min) return TIERS[i].name;
    }
    return 'Bronze';
  }

  function getTierInfo(points) {
    let current = TIERS[0], next = TIERS[1];
    for (let i = 0; i < TIERS.length; i++) {
      if (points >= TIERS[i].min) { current = TIERS[i]; next = TIERS[i + 1] || null; }
    }
    return { current, next, TIERS };
  }

  function computeInitials(name) {
    if (!name) return '??';
    const parts = name.split(' ').filter(Boolean);
    if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    return name.substring(0, 2).toUpperCase();
  }

  let state = load();

  return {
    get: () => state,
    set: (updater) => {
      if (typeof updater === 'function') updater(state);
      else Object.assign(state, updater);
      if (state.user.points !== undefined) {
        state.user.tier = computeTier(state.user.points);
      }
      if (state.user.name) {
        state.user.initials = computeInitials(state.user.name);
      }
      save(state);
    },
    reset: () => {
      state = JSON.parse(JSON.stringify(defaultState));
      save(state);
    },
    syncUser: (apiUser) => {
      state.user.id = apiUser.id;
      state.user.name = apiUser.name;
      state.user.email = apiUser.email;
      state.user.role = apiUser.role;
      state.user.city = apiUser.city || '';
      state.user.points = apiUser.karma_points || 0;
      state.user.tier = computeTier(state.user.points);
      state.user.initials = computeInitials(apiUser.name);
      save(state);
    },
    syncReports: (reports) => {
      state.reports = reports.map(r => {
        // Parse distance_px from description for parking reports
        // (stored as "distance:342" by the backend)
        let distance_px = null;
        if (r.source === 'parking' && r.description) {
          const m = r.description.match(/^distance:(\d+)$/);
          if (m) distance_px = parseInt(m[1], 10);
        }
        return {
          id: r.id,
          type: r.violation_type,
          plate: r.plate_number || '',
          location: r.location || '',
          date: r.created_at,
          confidence: r.confidence || 0,
          status: r.status === 'Under Review' ? 'pending'
            : r.status === 'Approved' ? 'verified'
              : r.status === 'Rejected' ? 'rejected'
                : r.status === 'Auto-Rejected' ? 'Auto-Rejected' : 'pending',
          points: r.status === 'Approved' ? 150 : 0,
          thumbnail: r.media_url || null,
          description: r.description || '',
          helmet_detected: r.helmet_detected || '',
          // ── New fields ──
          fine_amount: r.fine_amount || '',
          source: r.source || 'helmet_plate',
          vehicle_type: r.vehicle_type || '',
          distance_px: distance_px,
        };
      });
      save(state);
    },
    getTierInfo,
  };
})();


/* ─── Toast Notification System ─── */
const SaralToast = (() => {
  let container;

  function ensureContainer() {
    if (!container || !document.body.contains(container)) {
      container = document.createElement('div');
      container.className = 'saral-toast-container';
      document.body.appendChild(container);
    }
  }

  function show(message, type = 'info', duration = 3500) {
    ensureContainer();

    const toast = document.createElement('div');
    toast.className = `saral-toast saral-toast--${type}`;

    const icons = {
      success: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="7.5" stroke="currentColor" stroke-width="1.5"/><path d="M6 9l2 2 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
      error: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="7.5" stroke="currentColor" stroke-width="1.5"/><path d="M6.5 6.5l5 5M11.5 6.5l-5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
      info: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="7.5" stroke="currentColor" stroke-width="1.5"/><path d="M9 8v4M9 6h.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
      reward: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M9 1l2.09 4.26L16 5.97l-3.5 3.42.83 4.84L9 12.14l-4.33 2.09.83-4.84L2 5.97l4.91-.71L9 1z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>',
    };

    toast.innerHTML = `
      <span class="saral-toast__icon">${icons[type] || icons.info}</span>
      <span class="saral-toast__msg">${message}</span>
    `;

    container.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
      toast.classList.add('saral-toast--show');
    });

    // Auto-remove
    setTimeout(() => {
      toast.classList.remove('saral-toast--show');
      toast.classList.add('saral-toast--hide');
      toast.addEventListener('transitionend', () => toast.remove());
    }, duration);
  }

  return { show };
})();
