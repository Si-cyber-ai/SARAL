/* ═══════════════════════════════════════════════════
   SARAL — Landing Page Interactions
   Premium micro-interactions & scroll animations
   ═══════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ─── Navbar scroll behavior ───
  const nav = document.getElementById('nav');
  const handleNavScroll = () => {
    if (!nav) return;
    if (window.scrollY > 60) {
      nav.classList.add('nav--scrolled');
    } else {
      nav.classList.remove('nav--scrolled');
    }
  };
  window.addEventListener('scroll', handleNavScroll, { passive: true });
  handleNavScroll();

  // ─── Mobile menu toggle ───
  const mobileToggle = document.getElementById('mobileToggle');
  const navLinks = document.getElementById('navLinks');
  if (mobileToggle && navLinks) {
    mobileToggle.addEventListener('click', () => {
      navLinks.classList.toggle('is-open');
      mobileToggle.classList.toggle('is-active');
    });
    // Close on link click
    navLinks.querySelectorAll('.nav__link').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('is-open');
        mobileToggle.classList.remove('is-active');
      });
    });
  }

  // ─── Smooth scroll for anchor links ───
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', e => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        const offset = 80;
        const top = target.getBoundingClientRect().top + window.pageYOffset - offset;
        window.scrollTo({ top, behavior: 'smooth' });
      }
    });
  });

  // ─── Scroll Reveal (Intersection Observer) ───
  const revealElements = document.querySelectorAll('[data-reveal]');
  if (revealElements.length) {
    const revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            revealObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
    );
    revealElements.forEach(el => revealObserver.observe(el));
  }

  // ─── Animated Number Counter ───
  const animateCounters = () => {
    const counters = document.querySelectorAll('[data-count]');
    const counterObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const el = entry.target;
            const target = parseInt(el.dataset.count, 10);
            const duration = 2000;
            const startTime = performance.now();
            const startVal = 0;

            const easeOutExpo = t => t === 1 ? 1 : 1 - Math.pow(2, -10 * t);

            const tick = (now) => {
              const elapsed = now - startTime;
              const progress = Math.min(elapsed / duration, 1);
              const easedProgress = easeOutExpo(progress);
              const current = Math.round(startVal + (target - startVal) * easedProgress);

              el.textContent = current.toLocaleString();

              if (progress < 1) {
                requestAnimationFrame(tick);
              }
            };

            requestAnimationFrame(tick);
            counterObserver.unobserve(el);
          }
        });
      },
      { threshold: 0.3 }
    );
    counters.forEach(c => counterObserver.observe(c));
  };
  animateCounters();

  // ─── Workflow Timeline — Node Activation on Scroll ───
  const workflowNodes = document.querySelectorAll('.workflow__node');
  if (workflowNodes.length) {
    const nodeObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const node = entry.target;
            const idx = parseInt(node.dataset.step, 10);

            // Activate all nodes up to this one with stagger
            workflowNodes.forEach(n => {
              const nIdx = parseInt(n.dataset.step, 10);
              if (nIdx <= idx) {
                setTimeout(() => {
                  n.classList.add('is-active');
                }, (nIdx - 1) * 200);
              }
            });
          }
        });
      },
      { threshold: 0.4 }
    );
    workflowNodes.forEach(n => nodeObserver.observe(n));
  }

  // ─── Parallax for Hero chips ───
  const chips = document.querySelectorAll('.hero__chip');
  if (chips.length && window.matchMedia('(min-width: 769px)').matches) {
    window.addEventListener('mousemove', (e) => {
      const x = (e.clientX / window.innerWidth - 0.5) * 2;
      const y = (e.clientY / window.innerHeight - 0.5) * 2;

      chips.forEach((chip, i) => {
        const factor = (i + 1) * 6;
        chip.style.transform = `translate(${x * factor}px, ${y * factor}px)`;
      });
    }, { passive: true });
  }

  // ─── Button ripple effect ───
  document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('mouseenter', function (e) {
      const rect = this.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      this.style.setProperty('--ripple-x', x + 'px');
      this.style.setProperty('--ripple-y', y + 'px');
    });
  });

  // ─── Subtle card tilt on hover ───
  document.querySelectorAll('.cap-panel, .impact__card').forEach(card => {
    card.addEventListener('mousemove', function (e) {
      const rect = this.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width - 0.5) * 6;
      const y = ((e.clientY - rect.top) / rect.height - 0.5) * 6;
      this.style.transform = `translateY(-4px) perspective(600px) rotateY(${x}deg) rotateX(${-y}deg)`;
    });
    card.addEventListener('mouseleave', function () {
      this.style.transform = '';
    });
  });

  // ─── Hero background parallax ───
  const heroRadial = document.querySelector('.hero__bg-radial');
  if (heroRadial) {
    window.addEventListener('scroll', () => {
      const scrolled = window.pageYOffset;
      if (scrolled < window.innerHeight) {
        heroRadial.style.transform = `translate(-50%, calc(-50% + ${scrolled * 0.15}px))`;
      }
    }, { passive: true });
  }

})();
