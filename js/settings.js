/* ═══════════════════════════════════════════════════
   SARAL — Settings Page Logic
   Profile, Privacy, Notifications tabs
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

  nameInput.value = state.user.name;
  emailInput.value = state.user.email;
  cityInput.value = state.user.city;

  document.getElementById('profileDisplayName').textContent = state.user.name;
  document.getElementById('profileInitials').textContent = state.user.initials;
  document.getElementById('profileTier').textContent = state.user.tier + ' Champion';

  // ─── Save Profile ───
  document.getElementById('saveProfile').addEventListener('click', () => {
    const name = nameInput.value.trim();
    const email = emailInput.value.trim();
    const city = cityInput.value.trim();

    if (!name) {
      SaralToast.show('Name cannot be empty', 'error');
      return;
    }

    SaralStore.set(s => {
      s.user.name = name;
      s.user.email = email;
      s.user.city = city;
      // Compute initials from name
      const parts = name.split(' ').filter(Boolean);
      s.user.initials = parts.length >= 2
        ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
        : name.substring(0, 2).toUpperCase();
    });

    // Update UI
    document.getElementById('profileDisplayName').textContent = name;
    document.getElementById('profileInitials').textContent = state.user.initials;

    SaralToast.show('Profile saved successfully', 'success');
  });

  // ─── Privacy Toggles ───
  const blurToggle = document.getElementById('toggleBlurPlates');
  const hideToggle = document.getElementById('toggleHideIdentity');

  blurToggle.checked = state.settings.blurPlates;
  hideToggle.checked = state.settings.hideIdentity;

  blurToggle.addEventListener('change', () => {
    SaralStore.set(s => { s.settings.blurPlates = blurToggle.checked; });
    SaralToast.show(
      blurToggle.checked ? 'License plate blurring enabled' : 'License plate blurring disabled',
      'info'
    );
  });

  hideToggle.addEventListener('change', () => {
    SaralStore.set(s => { s.settings.hideIdentity = hideToggle.checked; });
    SaralToast.show(
      hideToggle.checked ? 'Identity hidden on leaderboard' : 'Identity visible on leaderboard',
      'info'
    );
  });

  // ─── Notification Toggles ───
  const emailNotifToggle = document.getElementById('toggleEmailNotifs');
  const rewardAlertToggle = document.getElementById('toggleRewardAlerts');

  emailNotifToggle.checked = state.settings.emailNotifs;
  rewardAlertToggle.checked = state.settings.rewardAlerts;

  emailNotifToggle.addEventListener('change', () => {
    SaralStore.set(s => { s.settings.emailNotifs = emailNotifToggle.checked; });
    SaralToast.show(
      emailNotifToggle.checked ? 'Email notifications enabled' : 'Email notifications disabled',
      'info'
    );
  });

  rewardAlertToggle.addEventListener('change', () => {
    SaralStore.set(s => { s.settings.rewardAlerts = rewardAlertToggle.checked; });
    SaralToast.show(
      rewardAlertToggle.checked ? 'Reward alerts enabled' : 'Reward alerts disabled',
      'info'
    );
  });

  // ─── Reset All Data ───
  document.getElementById('resetData').addEventListener('click', () => {
    if (confirm('Are you sure you want to reset all data? This cannot be undone.')) {
      SaralStore.reset();
      SaralToast.show('All data has been reset', 'info');
      setTimeout(() => window.location.reload(), 800);
    }
  });

})();
