/* ═══════════════════════════════════════════════════
   SARAL — Settings Page Logic
   API-backed profile, Privacy, Notifications tabs
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

  // ─── Tab Switching ───
  const tabBtns = document.querySelectorAll('.settings-tabs__btn');
  const panels = document.querySelectorAll('.settings-panel:not(.settings-panel--always)');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('settings-tabs__btn--active'));
      btn.classList.add('settings-tabs__btn--active');

      const tab = btn.dataset.tab;
      panels.forEach(p => {
        p.classList.toggle('settings-panel--active', p.id === `panel-${tab}`);
      });
    });
  });

  // ─── Populate Profile Fields ───
  const nameInput = document.getElementById('settingsName');
  const emailInput = document.getElementById('settingsEmail');
  const cityInput = document.getElementById('settingsCity');

  // Load fresh user data from API
  async function loadProfile() {
    try {
      await SaralAuth.refreshUser();
    } catch (e) { /* ignore */ }
    const s = SaralStore.get();
    if (nameInput) nameInput.value = s.user.name || '';
    if (emailInput) emailInput.value = s.user.email || '';
    if (cityInput) cityInput.value = s.user.city || '';

    const displayName = document.getElementById('profileDisplayName');
    const profileInitials = document.getElementById('profileInitials');
    const profileTier = document.getElementById('profileTier');
    const profileBadge = document.querySelector('.sidebar__profile-badge');
    if (displayName) displayName.textContent = s.user.name;
    if (profileInitials) profileInitials.textContent = s.user.initials;

    // Fetch live karma and compute correct tier
    try {
      const userId = SaralAuth.getUserId();
      const stats = await SaralAPI.getUserStats(userId);
      const { current } = SaralStore.getTierInfo(stats.karma_points);
      if (profileTier) profileTier.textContent = current.name + ' Champion';
      if (profileBadge) profileBadge.innerHTML =
        `<svg width="10" height="10" viewBox="0 0 10 10" fill="none"><circle cx="5" cy="5" r="4" fill="${current.color}"/></svg> ${current.icon} ${current.name} Champion`;
    } catch (_) {
      // Fallback to cached tier
      if (profileTier) profileTier.textContent = (s.user.tier || 'Bronze') + ' Champion';
    }
  }

  loadProfile();

  // ─── Save Profile (API-backed) ───
  document.getElementById('saveProfile').addEventListener('click', async () => {
    const name = nameInput.value.trim();
    const email = emailInput.value.trim();
    const city = cityInput.value.trim();

    if (!name) {
      SaralToast.show('Name cannot be empty', 'error');
      return;
    }

    const userId = SaralAuth.getUserId();
    if (!userId) {
      SaralToast.show('Please sign in again', 'error');
      return;
    }

    try {
      const result = await SaralAPI.updateProfile(userId, name, email, city);

      // Update local store
      SaralStore.set(s => {
        s.user.name = name;
        s.user.email = email;
        s.user.city = city;
        const parts = name.split(' ').filter(Boolean);
        s.user.initials = parts.length >= 2
          ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
          : name.substring(0, 2).toUpperCase();
      });

      // Update UI
      const displayName = document.getElementById('profileDisplayName');
      const profileInitials = document.getElementById('profileInitials');
      if (displayName) displayName.textContent = name;
      if (profileInitials) profileInitials.textContent = SaralStore.get().user.initials;

      SaralToast.show('Profile saved successfully', 'success');
    } catch (err) {
      SaralToast.show('Failed to save: ' + err.message, 'error');
    }
  });

  // ─── Privacy Toggles (local only — no backend for these) ───
  const blurToggle = document.getElementById('toggleBlurPlates');
  const hideToggle = document.getElementById('toggleHideIdentity');

  if (blurToggle) {
    blurToggle.checked = state.settings.blurPlates;
    blurToggle.addEventListener('change', () => {
      SaralStore.set(s => { s.settings.blurPlates = blurToggle.checked; });
      SaralToast.show(
        blurToggle.checked ? 'License plate blurring enabled' : 'License plate blurring disabled',
        'info'
      );
    });
  }

  if (hideToggle) {
    hideToggle.checked = state.settings.hideIdentity;
    hideToggle.addEventListener('change', () => {
      SaralStore.set(s => { s.settings.hideIdentity = hideToggle.checked; });
      SaralToast.show(
        hideToggle.checked ? 'Identity hidden on leaderboard' : 'Identity visible on leaderboard',
        'info'
      );
    });
  }

  // ─── Notification Toggles (local only) ───
  const emailNotifToggle = document.getElementById('toggleEmailNotifs');
  const rewardAlertToggle = document.getElementById('toggleRewardAlerts');

  if (emailNotifToggle) {
    emailNotifToggle.checked = state.settings.emailNotifs;
    emailNotifToggle.addEventListener('change', () => {
      SaralStore.set(s => { s.settings.emailNotifs = emailNotifToggle.checked; });
      SaralToast.show(
        emailNotifToggle.checked ? 'Email notifications enabled' : 'Email notifications disabled',
        'info'
      );
    });
  }

  if (rewardAlertToggle) {
    rewardAlertToggle.checked = state.settings.rewardAlerts;
    rewardAlertToggle.addEventListener('change', () => {
      SaralStore.set(s => { s.settings.rewardAlerts = rewardAlertToggle.checked; });
      SaralToast.show(
        rewardAlertToggle.checked ? 'Reward alerts enabled' : 'Reward alerts disabled',
        'info'
      );
    });
  }

  // ─── Reset All Data ───
  document.getElementById('resetData').addEventListener('click', () => {
    if (confirm('Are you sure you want to reset all local data? This cannot be undone.')) {
      SaralStore.reset();
      SaralToast.show('All local data has been reset', 'info');
      setTimeout(() => window.location.reload(), 800);
    }
  });

})();
