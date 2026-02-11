/* ═══════════════════════════════════════════════════
   SARAL — Rewards Page Logic
   Point balance, redemption, history
   ═══════════════════════════════════════════════════ */

(function () {
  'use strict';

  const state = SaralStore.get();

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

  // ─── Reward Icons ───
  const rewardIcons = {
    fastag: '<svg width="28" height="28" viewBox="0 0 28 28" fill="none"><rect x="3" y="8" width="22" height="12" rx="3" stroke="currentColor" stroke-width="1.5"/><path d="M8 12h4M8 16h2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="20" cy="14" r="2" stroke="currentColor" stroke-width="1.5"/></svg>',
    metro: '<svg width="28" height="28" viewBox="0 0 28 28" fill="none"><rect x="6" y="4" width="16" height="20" rx="4" stroke="currentColor" stroke-width="1.5"/><path d="M6 16h16" stroke="currentColor" stroke-width="1.5"/><circle cx="10" cy="20" r="1.5" fill="currentColor"/><circle cx="18" cy="20" r="1.5" fill="currentColor"/><path d="M10 8h8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M10 12h8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    fuel: '<svg width="28" height="28" viewBox="0 0 28 28" fill="none"><rect x="4" y="6" width="12" height="18" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M16 10l4-2v12l-4 2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><rect x="7" y="10" width="6" height="5" rx="1" stroke="currentColor" stroke-width="1.5"/></svg>',
    gift: '<svg width="28" height="28" viewBox="0 0 28 28" fill="none"><rect x="3" y="12" width="22" height="12" rx="2" stroke="currentColor" stroke-width="1.5"/><rect x="5" y="8" width="18" height="4" rx="1.5" stroke="currentColor" stroke-width="1.5"/><path d="M14 8v16" stroke="currentColor" stroke-width="1.5"/><path d="M14 8c-2-3-6-3-6 0s4 0 6 0c2-3 6-3 6 0s-4 0-6 0z" stroke="currentColor" stroke-width="1.5"/></svg>',
  };

  // ─── Render Balance ───
  function renderBalance() {
    const pts = state.user.points;
    animateNumber(document.getElementById('pointsBalance'), pts);

    // Tier logic
    const tiers = [
      { name: 'Bronze',   min: 0,    max: 500 },
      { name: 'Silver',   min: 500,  max: 1000 },
      { name: 'Gold',     min: 1000, max: 1500 },
      { name: 'Platinum', min: 1500, max: 2500 },
      { name: 'Diamond',  min: 2500, max: Infinity },
    ];

    let current = tiers[0];
    let next = tiers[1];
    for (let i = 0; i < tiers.length; i++) {
      if (pts >= tiers[i].min) {
        current = tiers[i];
        next = tiers[i + 1] || null;
      }
    }

    document.getElementById('tierName').textContent = current.name;
    const tierBadge = document.getElementById('tierBadge');
    tierBadge.className = 'rewards-balance__tier rewards-balance__tier--' + current.name.toLowerCase();

    if (next) {
      const ptsToGo = next.min - pts;
      const progress = ((pts - current.min) / (next.min - current.min)) * 100;
      document.getElementById('nextTier').textContent = next.name;
      document.getElementById('tierProgress').style.width = Math.min(progress, 100) + '%';
      document.getElementById('tierPtsLabel').textContent = `${ptsToGo} pts to go`;
    } else {
      document.getElementById('nextTier').textContent = 'Max';
      document.getElementById('tierProgress').style.width = '100%';
      document.getElementById('tierPtsLabel').textContent = 'Max tier reached!';
    }
  }

  function animateNumber(el, target) {
    let current = 0;
    const duration = 800;
    const start = performance.now();
    function tick(now) {
      const t = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      current = Math.round(ease * target);
      el.textContent = current.toLocaleString('en-IN');
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  // ─── Render Catalogue ───
  function renderCatalogue() {
    const grid = document.getElementById('rewardsGrid');
    grid.innerHTML = '';

    state.rewards.catalogue.forEach((reward, i) => {
      const canAfford = state.user.points >= reward.cost;
      const card = document.createElement('div');
      card.className = 'reward-card';
      card.style.animationDelay = `${i * 0.08}s`;

      card.innerHTML = `
        <div class="reward-card__icon">
          ${rewardIcons[reward.icon] || rewardIcons.gift}
        </div>
        <h3 class="reward-card__title">${reward.title}</h3>
        <p class="reward-card__desc">${reward.desc}</p>
        <div class="reward-card__footer">
          <span class="reward-card__cost">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 1l1.46 2.97L12 4.38l-2.44 2.39.58 3.37L7 8.58 3.86 10.14l.58-3.37L2 4.38l3.54-.41L7 1z" fill="currentColor"/></svg>
            ${reward.cost} pts
          </span>
          <button class="reward-card__btn ${canAfford ? '' : 'reward-card__btn--disabled'}"
                  data-reward-id="${reward.id}" ${canAfford ? '' : 'disabled'}>
            ${canAfford ? 'Redeem' : 'Not enough'}
          </button>
        </div>
      `;

      grid.appendChild(card);
    });

    // Attach redeem handlers
    grid.querySelectorAll('.reward-card__btn:not([disabled])').forEach(btn => {
      btn.addEventListener('click', () => handleRedeem(btn.dataset.rewardId));
    });
  }

  // ─── Redeem Handler ───
  function handleRedeem(rewardId) {
    const reward = state.rewards.catalogue.find(r => r.id === rewardId);
    if (!reward || state.user.points < reward.cost) {
      SaralToast.show('Insufficient Karma Points', 'error');
      return;
    }

    // Deduct points
    SaralStore.set(s => {
      s.user.points -= reward.cost;
      s.rewards.redeemed.push({
        id: 'RDM-' + Date.now(),
        rewardId: reward.id,
        title: reward.title,
        cost: reward.cost,
        date: new Date().toISOString(),
      });
    });

    SaralToast.show(`Redeemed "${reward.title}" for ${reward.cost} pts!`, 'success');

    // Re-render
    renderBalance();
    renderCatalogue();
    renderHistory();
  }

  // ─── Render History ───
  function renderHistory() {
    const container = document.getElementById('rewardsHistory');
    const emptyEl = document.getElementById('historyEmpty');
    const redeemed = state.rewards.redeemed;

    // Clear old items but keep empty
    container.querySelectorAll('.rewards-history__item').forEach(el => el.remove());

    if (redeemed.length === 0) {
      emptyEl.style.display = 'flex';
      return;
    }
    emptyEl.style.display = 'none';

    // Show newest first
    [...redeemed].reverse().forEach(item => {
      const el = document.createElement('div');
      el.className = 'rewards-history__item';
      const catItem = state.rewards.catalogue.find(c => c.id === item.rewardId);
      const icon = catItem ? (rewardIcons[catItem.icon] || rewardIcons.gift) : rewardIcons.gift;

      const d = new Date(item.date);
      const dateStr = d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });

      el.innerHTML = `
        <div class="rewards-history__icon">${icon}</div>
        <div class="rewards-history__info">
          <span class="rewards-history__title">${item.title}</span>
          <span class="rewards-history__date">${dateStr}</span>
        </div>
        <span class="rewards-history__cost">-${item.cost} pts</span>
      `;
      container.appendChild(el);
    });
  }

  // ─── Init ───
  renderBalance();
  renderCatalogue();
  renderHistory();

})();
