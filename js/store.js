/* ═══════════════════════════════════════════════════
   SARAL — Global State Store & Toast System
   Shared across all dashboard pages
   ═══════════════════════════════════════════════════ */

const SaralStore = (() => {
  const STORAGE_KEY = 'saral_state';

  const defaultState = {
    user: {
      name: 'Aarav Kumar',
      initials: 'AK',
      email: 'aarav.kumar@email.com',
      city: 'Bengaluru',
      tier: 'Gold',
      points: 1250,
    },
    reports: [
      {
        id: 'RPT-2024-001',
        type: 'Signal Violation',
        location: 'MG Road, Bengaluru',
        date: '2025-01-15T10:30:00',
        confidence: 94,
        status: 'verified',
        points: 150,
        thumbnail: null,
      },
      {
        id: 'RPT-2024-002',
        type: 'Wrong-Way Driving',
        location: 'NH48, Gurugram',
        date: '2025-01-15T07:15:00',
        confidence: 88,
        status: 'pending',
        points: 0,
        thumbnail: null,
      },
      {
        id: 'RPT-2024-003',
        type: 'Parking Violation',
        location: 'Sector 17, Chandigarh',
        date: '2025-01-14T14:45:00',
        confidence: 91,
        status: 'resolved',
        points: 150,
        thumbnail: null,
      },
      {
        id: 'RPT-2024-004',
        type: 'Over Speeding',
        location: 'Ring Road, Delhi',
        date: '2025-01-13T18:20:00',
        confidence: 96,
        status: 'resolved',
        points: 150,
        thumbnail: null,
      },
      {
        id: 'RPT-2024-005',
        type: 'No Helmet',
        location: 'Anna Salai, Chennai',
        date: '2025-01-13T09:10:00',
        confidence: 92,
        status: 'pending',
        points: 0,
        thumbnail: null,
      },
      {
        id: 'RPT-2024-006',
        type: 'Lane Violation',
        location: 'FC Road, Pune',
        date: '2025-01-12T16:00:00',
        confidence: 85,
        status: 'pending',
        points: 0,
        thumbnail: null,
      },
    ],
    rewards: {
      redeemed: [],
      catalogue: [
        { id: 'RW-001', title: 'FASTag Recharge', desc: 'Get ₹200 FASTag credit for highway tolls', cost: 500, icon: 'fastag' },
        { id: 'RW-002', title: 'Metro Pass', desc: '5-day unlimited metro rides in your city', cost: 400, icon: 'metro' },
        { id: 'RW-003', title: 'Fuel Voucher', desc: '₹300 fuel discount at partner stations', cost: 600, icon: 'fuel' },
        { id: 'RW-004', title: 'Gift Card', desc: '₹500 Amazon/Flipkart gift card', cost: 800, icon: 'gift' },
      ],
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
        // Merge with defaults to ensure new keys exist
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

  let state = load();

  return {
    get: () => state,
    set: (updater) => {
      if (typeof updater === 'function') updater(state);
      else Object.assign(state, updater);
      save(state);
    },
    reset: () => {
      state = JSON.parse(JSON.stringify(defaultState));
      save(state);
    },
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
