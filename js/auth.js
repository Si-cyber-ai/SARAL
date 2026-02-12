/* ═══════════════════════════════════════════════════
   SARAL — Authentication & Role-Based Access Control
   Vanilla JS auth context with role routing
   ═══════════════════════════════════════════════════ */

const SaralAuth = (() => {
  const AUTH_KEY = 'saral_auth';
  const USERS_KEY = 'saral_users';

  // ─── Default mock users ───
  const defaultUsers = [
    { name: 'Aarav Kumar',      email: 'aarav@saral.in',      password: 'citizen123', role: 'user' },
    { name: 'Insp. T. Prasad',  email: 'prasad@authority.in',  password: 'authority123', role: 'authority' },
  ];

  function getUsers() {
    try {
      const raw = localStorage.getItem(USERS_KEY);
      return raw ? JSON.parse(raw) : [...defaultUsers];
    } catch { return [...defaultUsers]; }
  }

  function saveUsers(users) {
    localStorage.setItem(USERS_KEY, JSON.stringify(users));
  }

  function getSession() {
    try {
      const raw = localStorage.getItem(AUTH_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }

  function saveSession(session) {
    if (session) {
      localStorage.setItem(AUTH_KEY, JSON.stringify(session));
    } else {
      localStorage.removeItem(AUTH_KEY);
    }
  }

  // ─── Public API ───

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

  function signIn(email, password, role) {
    const users = getUsers();
    const user = users.find(u =>
      u.email.toLowerCase() === email.toLowerCase() &&
      u.password === password &&
      u.role === role
    );
    if (!user) return { success: false, error: 'Invalid credentials or role mismatch' };

    const session = { name: user.name, email: user.email, role: user.role };
    saveSession(session);
    return { success: true, user: session };
  }

  function signUp(name, email, password, role) {
    const users = getUsers();
    if (users.find(u => u.email.toLowerCase() === email.toLowerCase())) {
      return { success: false, error: 'An account with this email already exists' };
    }
    users.push({ name, email, password, role });
    saveUsers(users);

    const session = { name, email, role };
    saveSession(session);
    return { success: true, user: session };
  }

  function signOut() {
    saveSession(null);
    // Replace history to prevent back-button access to dashboard
    window.location.replace('signin.html');
  }

  /**
   * Redirect based on role after login
   */
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

  /**
   * Guard: call on protected pages.
   * @param {string|null} requiredRole - 'user', 'authority', or null (any authenticated)
   */
  function requireAuth(requiredRole) {
    if (!isAuthenticated()) {
      window.location.href = 'signin.html';
      return false;
    }
    if (requiredRole && getRole() !== requiredRole) {
      window.location.href = 'signin.html';
      return false;
    }
    return true;
  }

  // Initialize default users into localStorage if not present
  if (!localStorage.getItem(USERS_KEY)) {
    saveUsers(defaultUsers);
  }

  return {
    isAuthenticated,
    getUser,
    getRole,
    signIn,
    signUp,
    signOut,
    redirectToDashboard,
    requireAuth,
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
