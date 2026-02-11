/* ═══════════════════════════════════════════════════
   SARAL — Dashboard Interactions
   ═══════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ─── Sidebar toggle (mobile) ───
  const sidebar = document.getElementById('sidebar');
  const sidebarToggle = document.getElementById('sidebarToggle');
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('is-open');
    });
    // Close when clicking outside
    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('is-open') &&
          !sidebar.contains(e.target) &&
          !sidebarToggle.contains(e.target)) {
        sidebar.classList.remove('is-open');
      }
    });
  }

  // ─── Scroll Reveal ───
  const revealElements = document.querySelectorAll('[data-reveal]');
  if (revealElements.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1 }
    );
    revealElements.forEach(el => observer.observe(el));
  }

  // ─── Animated Number Counters ───
  const counters = document.querySelectorAll('[data-count]');
  if (counters.length) {
    const counterObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const el = entry.target;
            const target = parseInt(el.dataset.count, 10);
            const duration = 1500;
            const start = performance.now();
            const easeOut = t => 1 - Math.pow(1 - t, 3);

            const tick = (now) => {
              const elapsed = now - start;
              const progress = Math.min(elapsed / duration, 1);
              const current = Math.round(target * easeOut(progress));
              el.textContent = current.toLocaleString();
              if (progress < 1) requestAnimationFrame(tick);
            };
            requestAnimationFrame(tick);
            counterObserver.unobserve(el);
          }
        });
      },
      { threshold: 0.3 }
    );
    counters.forEach(c => counterObserver.observe(c));
  }

  // ─── Generate Contribution Heatmap ───
  const heatmap = document.getElementById('heatmap');
  if (heatmap) {
    const levels = ['', 'l1', 'l2', 'l3', 'l4'];
    const data = [];
    // Generate 84 cells (12 weeks × 7 days)
    for (let i = 0; i < 84; i++) {
      const rand = Math.random();
      let level = 0;
      if (rand > 0.7) level = 1;
      if (rand > 0.82) level = 2;
      if (rand > 0.9) level = 3;
      if (rand > 0.96) level = 4;
      data.push(level);
    }

    data.forEach((lvl) => {
      const cell = document.createElement('div');
      cell.className = 'dash-heatmap__cell' + (lvl > 0 ? ' dash-heatmap__cell--l' + lvl : '');
      heatmap.appendChild(cell);
    });
  }

  // ─── Tab switching ───
  document.querySelectorAll('.dash-card__tabs').forEach(tabGroup => {
    tabGroup.querySelectorAll('.dash-card__tab').forEach(tab => {
      tab.addEventListener('click', () => {
        tabGroup.querySelectorAll('.dash-card__tab').forEach(t => t.classList.remove('dash-card__tab--active'));
        tab.classList.add('dash-card__tab--active');
      });
    });
  });

  // ─── Card hover animations ───
  document.querySelectorAll('.dash-stat').forEach(card => {
    card.addEventListener('mouseenter', function () {
      this.style.transform = 'translateY(-3px) scale(1.02)';
    });
    card.addEventListener('mouseleave', function () {
      this.style.transform = '';
    });
  });

})();
