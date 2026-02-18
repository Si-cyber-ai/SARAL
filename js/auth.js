/* ═══════════════════════════════════════════════════
   SARAL — Authentication & Role-Based Access Control
   API-backed auth with session persistence
   ═══════════════════════════════════════════════════ */

const SaralAuth = (() => {
  const AUTH_KEY = 'saral_auth';

  function getSession() {
    try {
      const raw = localStorage.getItem(AUTH_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }

  function saveSession(session) {
    if (session) {
      localStorage.setItem(AUTH_KEY, JSON.stringify(session));
      // Also sync to SaralStore if available
      if (typeof SaralStore !== 'undefined' && SaralStore.syncUser) {
        SaralStore.syncUser(session);
      }
    } else {
      localStorage.removeItem(AUTH_KEY);
    }
  }

  function isAuthenticated() {
    return !!getSession();
  }

  function getUser() {
    return getSession();
  }

  function getRole() {
    const s = getSession();
    return s ? s.role : null;
  }

  function getUserId() {
    const s = getSession();
    return s ? s.id : null;
  }

  async function signIn(email, password, role) {
    try {
      const data = await SaralAPI.signIn(email, password, role);
      if (data.success) {
        saveSession(data.user);
        return { success: true, user: data.user };
      }
      return { success: false, error: 'Invalid credentials' };
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  async function signUp(name, email, password, role) {
    try {
      const data = await SaralAPI.signUp(name, email, password, role);
      if (data.success) {
        saveSession(data.user);
        return { success: true, user: data.user };
      }
      return { success: false, error: 'Sign up failed' };
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  function signOut() {
    saveSession(null);
    if (typeof SaralStore !== 'undefined') {
      SaralStore.reset();
    }
    window.location.replace('signin.html');
  }

  function redirectToDashboard() {
    const role = getRole();
    if (role === 'authority') {
      window.location.href = 'authority.html';
    } else if (role === 'user') {
      window.location.href = 'dashboard.html';
    } else {
      window.location.href = 'index.html';
    }
  }

  function requireAuth(requiredRole) {
    if (!isAuthenticated()) {
      window.location.href = 'signin.html';
      return false;
    }
    if (requiredRole && getRole() !== requiredRole) {
      window.location.href = 'signin.html';
      return false;
    }
    // Sync user to store on page load
    const session = getSession();
    if (session && typeof SaralStore !== 'undefined' && SaralStore.syncUser) {
      SaralStore.syncUser(session);
    }
    return true;
  }

  /** Refresh user data from API */
  async function refreshUser() {
    const session = getSession();
    if (!session || !session.id) return;
    try {
      const fresh = await SaralAPI.getUser(session.id);
      saveSession({ ...session, ...fresh });
    } catch (e) {
      console.warn('[Auth] Failed to refresh user:', e.message);
    }
  }

  return {
    isAuthenticated,
    getUser,
    getRole,
    getUserId,
    signIn,
    signUp,
    signOut,
    redirectToDashboard,
    requireAuth,
    refreshUser,
  };
})();


/* ─── Dynamic Navbar Renderer (for landing page) ─── */
const SaralNav = (() => {

  function render() {
    const actionsEl = document.querySelector('.nav__actions');
    if (!actionsEl) return;

    if (SaralAuth.isAuthenticated()) {
      const user = SaralAuth.getUser();
      const role = user.role;

      if (role === 'user') {
        actionsEl.innerHTML = `
          <a href="dashboard.html" class="nav__btn nav__btn--ghost">Dashboard</a>
          <button class="nav__btn nav__btn--ghost" onclick="SaralAuth.signOut(); window.location.href='index.html';">Logout</button>
          <a href="report.html" class="nav__btn nav__btn--primary">
            <span>Report Now</span>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </a>
        `;
      } else if (role === 'authority') {
        actionsEl.innerHTML = `
          <a href="authority.html" class="nav__btn nav__btn--ghost">Authority Panel</a>
          <button class="nav__btn nav__btn--ghost" onclick="SaralAuth.signOut(); window.location.href='index.html';">Logout</button>
        `;
      }
    } else {
      actionsEl.innerHTML = `
        <a href="signin.html" class="nav__btn nav__btn--ghost">Sign In</a>
        <a href="signup.html" class="nav__btn nav__btn--primary">
          <span>Get Started</span>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </a>
      `;
    }
  }

  return { render };
})();
