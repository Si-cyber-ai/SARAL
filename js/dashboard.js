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

  // ─── Generate Violation Hotspot Map (Leaflet) ───
  const mapEl = document.getElementById('leafletMap');
  const violationList = document.getElementById('violationList');

  if (mapEl && typeof L !== 'undefined') {
    // Real lat/lng coordinates for Indian cities
    const violationData = [
      { id: 1,  name: 'MG Road Junction',      area: 'Central Delhi',  lat: 28.6328, lng: 77.2197, count: 87, types: ['Signal', 'Speeding', 'Parking'] },
      { id: 2,  name: 'NH-48 Toll Plaza',       area: 'Gurugram',       lat: 28.4595, lng: 77.0266, count: 74, types: ['Wrong-Way', 'Speeding'] },
      { id: 3,  name: 'AIIMS Flyover',          area: 'South Delhi',    lat: 28.5672, lng: 77.2100, count: 63, types: ['Speeding', 'Lane Violation'] },
      { id: 4,  name: 'Sector 17 Crossing',     area: 'Chandigarh',     lat: 30.7415, lng: 76.7797, count: 56, types: ['Parking', 'Signal'] },
      { id: 5,  name: 'Outer Ring Road',         area: 'Bengaluru',      lat: 12.9352, lng: 77.6245, count: 49, types: ['Speeding', 'No Helmet'] },
      { id: 6,  name: 'Marine Drive',            area: 'Mumbai',         lat: 18.9438, lng: 72.8235, count: 42, types: ['Parking', 'Signal'] },
      { id: 7,  name: 'Anna Salai',              area: 'Chennai',        lat: 13.0524, lng: 80.2508, count: 38, types: ['Signal', 'Wrong-Way'] },
      { id: 8,  name: 'JM Road',                 area: 'Pune',           lat: 18.5286, lng: 73.8446, count: 31, types: ['Parking', 'No Helmet'] },
      { id: 9,  name: 'Ashram Chowk',            area: 'East Delhi',     lat: 28.5797, lng: 77.2580, count: 27, types: ['Signal', 'Speeding'] },
      { id: 10, name: 'Rajiv Chowk',             area: 'New Delhi',      lat: 28.6328, lng: 77.2195, count: 22, types: ['Parking', 'Signal'] },
      { id: 11, name: 'Silk Board Junction',     area: 'Bengaluru',      lat: 12.9170, lng: 77.6230, count: 68, types: ['Speeding', 'Signal'] },
      { id: 12, name: 'Hinjewadi IT Park',       area: 'Pune',           lat: 18.5912, lng: 73.7389, count: 14, types: ['Parking'] },
      { id: 13, name: 'Bandra-Worli Sea Link',   area: 'Mumbai',         lat: 19.0380, lng: 72.8195, count: 53, types: ['Speeding', 'Lane Violation'] },
      { id: 14, name: 'Cyber City',              area: 'Gurugram',       lat: 28.4940, lng: 77.0887, count: 45, types: ['Parking', 'Signal', 'Speeding'] },
      { id: 15, name: 'Howrah Bridge',            area: 'Kolkata',        lat: 22.5851, lng: 88.3468, count: 41, types: ['Wrong-Way', 'No Helmet'] },
      { id: 16, name: 'MI Road',                  area: 'Jaipur',         lat: 26.9157, lng: 75.8024, count: 35, types: ['Signal', 'Parking'] },
      { id: 17, name: 'Hazratganj',               area: 'Lucknow',        lat: 26.8510, lng: 80.9462, count: 29, types: ['Parking', 'Wrong-Way'] },
      { id: 18, name: 'Tank Bund Road',           area: 'Hyderabad',      lat: 17.4239, lng: 78.4738, count: 58, types: ['Speeding', 'Signal', 'No Helmet'] },
    ];

    const maxCount = Math.max(...violationData.map(d => d.count));

    // Initialise Leaflet map centred on India
    const map = L.map('leafletMap', {
      center: [22.5, 78.5],
      zoom: 5,
      scrollWheelZoom: true,
      zoomControl: true,
      attributionControl: false,
    });

    // Use CartoDB Positron tiles (clean, light styling)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 18,
      subdomains: 'abcd',
    }).addTo(map);

    // Attribution (smaller)
    L.control.attribution({ prefix: false, position: 'bottomright' })
      .addAttribution('&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://osm.org/copyright">OSM</a>')
      .addTo(map);

    // Store circle markers for hover interaction
    const circleMarkers = {};

    // Helper: get colour from count
    function getColor(count) {
      const ratio = count / maxCount;
      if (ratio > 0.75) return '#ef4444';
      if (ratio > 0.5)  return '#f97316';
      if (ratio > 0.3)  return '#fbbf24';
      return '#60a5fa';
    }

    // Add circle markers
    violationData.forEach(v => {
      const ratio = v.count / maxCount;
      const radius = 8 + ratio * 22;
      const color = getColor(v.count);

      const circle = L.circleMarker([v.lat, v.lng], {
        radius: radius,
        fillColor: color,
        color: color,
        weight: 2,
        opacity: 0.9,
        fillOpacity: 0.35 + ratio * 0.3,
      }).addTo(map);

      // Popup
      circle.bindPopup(`
        <div class="vmap-popup">
          <strong>${v.name}</strong>
          <span class="vmap-popup__area">${v.area}</span>
          <span class="vmap-popup__count">${v.count} violations</span>
          <span class="vmap-popup__types">${v.types.join(' &middot; ')}</span>
        </div>
      `, { className: 'vmap-popup-container', closeButton: false, offset: [0, -4] });

      circle.on('mouseover', function () {
        this.setStyle({ fillOpacity: 0.8, weight: 3 });
        this.openPopup();
      });
      circle.on('mouseout', function () {
        this.setStyle({ fillOpacity: 0.35 + ratio * 0.3, weight: 2 });
        this.closePopup();
      });

      circleMarkers[v.id] = circle;
    });

    // Fit bounds to all markers
    const bounds = L.latLngBounds(violationData.map(v => [v.lat, v.lng]));
    map.fitBounds(bounds.pad(0.15));

    // Render ranked list
    if (violationList) {
      const sorted = [...violationData].sort((a, b) => b.count - a.count);
      sorted.forEach((v, i) => {
        const pct = Math.round((v.count / maxCount) * 100);
        const row = document.createElement('div');
        row.className = 'violation-map__row';
        row.dataset.id = v.id;
        row.innerHTML = `
          <span class="violation-map__rank">${i + 1}</span>
          <div class="violation-map__row-info">
            <span class="violation-map__row-name">${v.name}</span>
            <span class="violation-map__row-area">${v.area}</span>
            <div class="violation-map__row-bar">
              <div class="violation-map__row-fill" style="width:${pct}%"></div>
            </div>
          </div>
          <span class="violation-map__row-count">${v.count}</span>
        `;

        // Hover: highlight marker on map
        row.addEventListener('mouseenter', () => {
          const c = circleMarkers[v.id];
          if (c) { c.setStyle({ fillOpacity: 0.85, weight: 4, radius: 18 }); c.openPopup(); }
        });
        row.addEventListener('mouseleave', () => {
          const c = circleMarkers[v.id];
          const ratio = v.count / maxCount;
          if (c) { c.setStyle({ fillOpacity: 0.35 + ratio * 0.3, weight: 2, radius: 8 + ratio * 22 }); c.closePopup(); }
        });
        // Click: fly to location
        row.addEventListener('click', () => {
          map.flyTo([v.lat, v.lng], 12, { duration: 1.2 });
          const c = circleMarkers[v.id];
          if (c) setTimeout(() => c.openPopup(), 1300);
        });

        violationList.appendChild(row);
      });
    }
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
