function updateTypeCount(){
  if (!els.typesCount) return;
  const total = els.typeChips.querySelectorAll('input[type="checkbox"]').length;
  const selected = els.typeChips.querySelectorAll('input[type="checkbox"]:checked').length;
  els.typesCount.textContent = total ? ` (${selected}/${total})` : '';
}

function updatePackCount(){
  if (!els.packsCount) return;
  const total = els.packChips.querySelectorAll('input[type="checkbox"]').length;
  const selected = els.packChips.querySelectorAll('input[type="checkbox"]:checked').length;
  els.packsCount.textContent = total ? ` (${selected}/${total})` : '';
}
// Simple frontend to drive the API and visualize results on a Leaflet map

const apiBase = window.location.origin;

const els = {
  centerText: document.getElementById('centerText'),
  radius: document.getElementById('radius'),
  excludeSAB: document.getElementById('excludeSAB'),
  categories: document.getElementById('categories'),
  catSelectAll: document.getElementById('catSelectAll'),
  catSelectNone: document.getElementById('catSelectNone'),
  searchBtn: document.getElementById('searchBtn'),
  loadMoreBtn: document.getElementById('loadMoreBtn'),
  exportCsvBtn: document.getElementById('exportCsvBtn'),
  resultsList: document.getElementById('resultsList'),
  resultsCount: document.getElementById('resultsCount'),
  minRating: document.getElementById('minRating'),
  minReviews: document.getElementById('minReviews'),
  applyFiltersBtn: document.getElementById('applyFiltersBtn'),
  drawRadiusBtn: document.getElementById('drawRadiusBtn'),
  drawStatus: document.getElementById('drawStatus'),
  typeChips: document.getElementById('typeChips'),
  typesSelectAll: document.getElementById('typesSelectAll'),
  typesSelectNone: document.getElementById('typesSelectNone'),
  typesCount: document.getElementById('typesCount'),
  packChips: document.getElementById('packChips'),
  packsSelectAll: document.getElementById('packsSelectAll'),
  packsSelectNone: document.getElementById('packsSelectNone'),
  packsCount: document.getElementById('packsCount'),
  toggleSidebarBtn: document.getElementById('toggleSidebarBtn'),
  toggleDensityBtn: document.getElementById('toggleDensityBtn'),
  tableHeader: document.getElementById('tableHeader'),
  clearRadiusBtn: document.getElementById('clearRadiusBtn'),
  resetFiltersBtn: document.getElementById('resetFiltersBtn'),
  densityToggleBtn: document.getElementById('densityToggleBtn'),
  highRecallToggle: document.getElementById('highRecallToggle'),
};

// Default high-recall to off to minimize API pagination costs
if (els.highRecallToggle) els.highRecallToggle.checked = false;

let map;
let markersLayer;
let lastResponse = null; // cache of last SearchResponse
let selectedCategories = new Set();
let nextPageToken = null; // pagination token
let allPacks = []; // taxonomy from /categories

// Draw radius state
let drawMode = false;
let drawStartLatLng = null;
let drawCircle = null; // Leaflet circle overlay
let centerOverride = null; // {lat,lng} when drawn
let radiusOverride = null; // meters when drawn
let drawByDrag = false; // keep drag disabled; we use click-to-center

// Type filters state (primaryType)
let selectedTypes = new Set();
// Category pack filters state (labels from r.categories)
let selectedPacks = new Set();

// Sorting state
let sortKey = 'name'; // 'name' | 'category' | 'address' | 'city' | 'state' | 'zip' | 'phone' | 'website' | 'distance'
let sortDir = 'asc'; // 'asc' | 'desc'

function initMap() {
  map = L.map('map');
  const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  });
  osm.addTo(map);
  markersLayer = L.layerGroup().addTo(map);
  map.setView([42.3314, -83.0458], 12); // default Detroit

  // Click-to-center radius only
  map.on('click', (e) => {
    if (!drawMode) return;
    const milesVal = parseFloat(els.radius.value);
    const radiusMiles = !isNaN(milesVal) ? milesVal : 5;
    const meters = Math.max(1, Math.round(radiusMiles * 1609.34));
    centerOverride = { lat: e.latlng.lat, lng: e.latlng.lng };
    radiusOverride = meters;
    els.centerText.value = `${centerOverride.lat.toFixed(6)}, ${centerOverride.lng.toFixed(6)}`;
    if (!drawCircle) {
      drawCircle = L.circle(e.latlng, { radius: meters, color: '#2f81f7' }).addTo(map);
    } else {
      drawCircle.setLatLng(e.latlng);
      drawCircle.setRadius(meters);
    }
    try { map.fitBounds(drawCircle.getBounds(), { padding: [20, 20] }); } catch (_) { map.setView(e.latlng); }
  });
}

function haversineMiles(lat1, lon1, lat2, lon2) {
  const R = 3958.7613; // Earth radius in miles
  const toRad = (d) => d * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat/2)**2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2)**2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function computeDistanceMiles(r){
  const cLat = lastResponse?.centerLat;
  const cLng = lastResponse?.centerLng;
  if (typeof r.lat === 'number' && typeof r.lng === 'number' && typeof cLat === 'number' && typeof cLng === 'number'){
    return haversineMiles(cLat, cLng, r.lat, r.lng);
  }
  return null;
}

// Parse formatted address into street, city, state, zip (US-centric heuristic)
function parseAddressParts(addr){
  const out = { street: '', city: '', state: '', zip: '' };
  if (!addr || typeof addr !== 'string') return out;
  let parts = addr.split(',').map(s => s.trim()).filter(Boolean);
  if (parts.length < 2){ out.street = addr; return out; }
  // If last part is a country (e.g., 'USA' or 'United States'), drop it for parsing city/state/zip
  const lastPart = parts[parts.length - 1];
  const isCountry = /^(USA|United States|US)$/i.test(lastPart) || (/^[A-Za-z\s]+$/.test(lastPart) && !/\d/.test(lastPart) && lastPart.length > 2);
  if (isCountry) parts = parts.slice(0, -1);
  if (parts.length >= 3){
    const stateZipPart = parts[parts.length - 1] || '';
    out.city = parts[parts.length - 2] || '';
    out.street = parts.slice(0, parts.length - 2).join(', ');
    const m = stateZipPart.match(/([A-Z]{2})\s*,?\s*(\d{5})(?:-\d{4})?/i);
    if (m){
      out.state = (m[1] || '').toUpperCase();
      out.zip = m[2] || '';
    } else {
      const segs = stateZipPart.split(/\s+/);
      if (segs.length >= 2){
        out.state = (segs[0] || '').toUpperCase();
        out.zip = segs[1] || '';
      } else {
        out.state = stateZipPart.toUpperCase();
      }
    }
  } else if (parts.length === 2){
    // e.g., 'Detroit, MI 48235'
    out.street = '';
    out.city = parts[0];
    const m = parts[1].match(/([A-Z]{2})\s*,?\s*(\d{5})(?:-\d{4})?/i);
    if (m){
      out.state = (m[1] || '').toUpperCase();
      out.zip = m[2] || '';
    } else {
      out.state = parts[1].toUpperCase();
    }
  } else {
    out.street = parts[0] || addr;
  }
  return out;
}

function valueForSort(r, idx){
  switch (sortKey){
    case 'name': return (r.name || '').toLowerCase();
    case 'category': {
      const packs = Array.isArray(r.categories) && r.categories.length ? r.categories : [(getCategoryLabel(r).label)];
      return packs.join('|').toLowerCase();
    }
    case 'address': {
      const ap = parseAddressParts(r.formattedAddress);
      return (ap.street || '').toLowerCase();
    }
    case 'city': return (parseAddressParts(r.formattedAddress).city || '').toLowerCase();
    case 'state': return (parseAddressParts(r.formattedAddress).state || '').toLowerCase();
    case 'zip': return (parseAddressParts(r.formattedAddress).zip || '').toLowerCase();
    case 'phone': return (r.phone || '').toLowerCase();
    case 'website': return (r.website || '').toLowerCase();
    case 'rating': return typeof r.rating === 'number' ? r.rating : -1;
    case 'distance': {
      const d = computeDistanceMiles(r);
      return d == null ? Number.POSITIVE_INFINITY : d;
    }
    default: return idx;
  }
}

function sortList(list){
  if (!sortKey) return list;
  const withIndex = list.map((r, i) => ({ r, i }));
  withIndex.sort((a,b) => {
    const va = valueForSort(a.r, a.i);
    const vb = valueForSort(b.r, b.i);
    if (va < vb) return sortDir === 'asc' ? -1 : 1;
    if (va > vb) return sortDir === 'asc' ? 1 : -1;
    return 0;
  });
  return withIndex.map(x => x.r);
}

function wireSorting(){
  if (!els.tableHeader) return;
  const cells = els.tableHeader.querySelectorAll('.cell[data-sort]');
  const updateSortIndicators = () => {
    cells.forEach(c => {
      const key = c.getAttribute('data-sort');
      c.classList.remove('sort-asc','sort-desc','sorted');
      if (key === sortKey){
        c.classList.add('sorted');
        c.classList.add(sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
      }
    });
  };
  updateSortIndicators();
  cells.forEach(cell => {
    cell.style.cursor = 'pointer';
    cell.addEventListener('click', () => {
      const key = cell.getAttribute('data-sort');
      if (sortKey === key){
        sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        sortKey = key;
        sortDir = 'asc';
      }
      if (!lastResponse) return;
      const filtered = clientFilter(lastResponse.results || []);
      const sorted = sortList(filtered);
      renderResults(sorted);
      updateSortIndicators();
    });
  });
}

function wireToggles(){
  // Sidebar collapse
  if (els.toggleSidebarBtn){
  }
}

// Persist group collapsed state in localStorage
function getGroupCollapsed(name){
  try{
    const raw = localStorage.getItem('groupCollapsed');
    if (!raw) return null;
    const obj = JSON.parse(raw);
    if (obj && Object.prototype.hasOwnProperty.call(obj, name)) return !!obj[name];
  }catch(_){}
  return null;
}
function setGroupCollapsed(name, val){
  try{
    const raw = localStorage.getItem('groupCollapsed');
    const obj = raw ? JSON.parse(raw) : {};
    obj[name] = !!val;
    localStorage.setItem('groupCollapsed', JSON.stringify(obj));
  }catch(_){}
}

function getCategoryLabel(r) {
  const pt = (r.primaryType || '').toLowerCase();
  // Map common primary types to friendly labels and badge classes
  const map = {
    'car_repair': { label: 'TRADITIONAL AUTO', css: 'badge-red' },
    'car_wash': { label: 'CAR WASH', css: 'badge-blue' },
    'car_dealer': { label: 'DEALERS', css: 'badge-indigo' },
    'plumber': { label: 'PLUMBERS', css: 'badge-teal' },
    'electrician': { label: 'ELECTRICIANS', css: 'badge-amber' },
    'roofing_contractor': { label: 'ROOFING', css: 'badge-amber' },
    'locksmith': { label: 'LOCKSMITHS', css: 'badge-emerald' },
    'moving_company': { label: 'MOVING', css: 'badge-violet' },
    'car_rental': { label: 'RENTAL FLEETS', css: 'badge-indigo' },
    'hardware_store': { label: 'HARDWARE / SUPPLY', css: 'badge-slate' },
    'painter': { label: 'PAINTING', css: 'badge-purple' },
  };
  if (map[pt]) return map[pt];
  if (pt) return { label: pt.replace(/_/g, ' ').toUpperCase(), css: 'badge-slate' };
  return { label: 'OTHER', css: 'badge-slate' };
}

function mapPackLabelToClass(lbl){
  const L = (lbl || '').toUpperCase();
  if (L.includes('AUTO')) return 'badge-red';
  if (L.includes('LOGISTICS') || L.includes('COURIER') || L.includes('MOVING')) return 'badge-emerald';
  if (L.includes('INDUSTRIAL') || L.includes('MANUFACTURING')) return 'badge-indigo';
  if (L.includes('HOME') || L.includes('HVAC') || L.includes('PLUMB') || L.includes('ELECTRIC') || L.includes('ROOF')) return 'badge-amber';
  return 'badge-slate';
}

function buildTypeChips(results) {
  // Collect unique primary types from results
  const hasOther = (results || []).some(r => !r.primaryType);
  const types = Array.from(new Set((results || []).map(r => r.primaryType).filter(Boolean))).sort();
  els.typeChips.innerHTML = '';
  // If nothing selected yet, default to all selected
  const initializeAll = selectedTypes.size === 0;
  const allTypes = hasOther ? [...types, '__other__'] : types;
  for (const t of allTypes) {
    const id = `type_${t}`;
    const label = document.createElement('label');
    label.className = 'chip';
    label.innerHTML = `
      <input type="checkbox" id="${id}" value="${t}">
      <span>${t === '__other__' ? 'other' : t}</span>
    `;
    const cb = label.querySelector('input');
    if (initializeAll) {
      cb.checked = true;
      selectedTypes.add(t);
    } else {
      cb.checked = selectedTypes.has(t);
    }
    cb.addEventListener('change', () => {
      if (cb.checked) selectedTypes.add(t); else selectedTypes.delete(t);
      if (lastResponse) {
        const filtered = clientFilter(lastResponse.results || []);
        const sorted = sortList(filtered);
        renderResults(sorted);
      }
      updateTypeCount();
    });
    els.typeChips.appendChild(label);
  }
  // Fallback: if none are checked (e.g., after certain flows), select all
  const anyChecked = els.typeChips.querySelectorAll('input[type="checkbox"]:checked').length > 0;
  if (!anyChecked) {
    els.typeChips.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.checked = true; selectedTypes.add(cb.value);
    });
  }
  updateTypeCount();
}

function buildPackChips(results) {
  // Build from full taxonomy so user can always see and toggle all packs
  const packs = (allPacks || []).map(p => p.label).sort();
  els.packChips.innerHTML = '';
  const initializeAll = selectedPacks.size === 0;
  for (const p of packs) {
    const id = `pack_${p.replace(/\s+/g,'_')}`;
    const label = document.createElement('label');
    label.className = 'chip';
    label.innerHTML = `
      <input type="checkbox" id="${id}" value="${p}">
      <span>${p}</span>
    `;
    const cb = label.querySelector('input');
    if (initializeAll) {
      cb.checked = true;
      selectedPacks.add(p);
    } else {
      cb.checked = selectedPacks.has(p);
    }
    cb.addEventListener('change', () => {
      if (cb.checked) selectedPacks.add(p); else selectedPacks.delete(p);
      if (lastResponse) {
        const filtered = clientFilter(lastResponse.results || []);
        const sorted = sortList(filtered);
        renderResults(sorted);
      }
      updatePackCount();
    });
    els.packChips.appendChild(label);
  }
  // Fallback: if none are checked, select all
  const anyChecked = els.packChips.querySelectorAll('input[type="checkbox"]:checked').length > 0;
  if (!anyChecked) {
    els.packChips.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.checked = true; selectedPacks.add(cb.value);
    });
  }
  updatePackCount();
}

async function fetchCategories() {
  const res = await fetch(`${apiBase}/categories`);
  if (!res.ok) throw new Error('Failed to load categories');
  const data = await res.json();
  allPacks = Array.isArray(data) ? data : [];
  renderCategoryGroupsHorizontal(data);
  // Also populate the Packs chips from full taxonomy immediately
  buildPackChips(lastResponse?.results || []);
}

function renderCategoryGroupsHorizontal(packs) {
  els.categories.innerHTML = '';
  // Build index by key for safety
  const byKey = new Map(packs.map(p => [p.key, p]));
  const groupDefs = [
    {
      name: 'Automotive & Fleet Core',
      keys: [
        'auto_traditional','quick_lube','tire_shops','auto_glass','body_collision','car_wash','towing','dealers',
        // new automotive/fleet
        'diesel_truck_repair','mobile_mechanics','fleet_washing','vehicle_wraps','upfitters','auto_parts_b2b','tint_ppf','onsite_fueling'
      ]
    },
    {
      name: 'Home / Field Services',
      keys: [
        'plumbers','electricians','roofing','hvac','pest_control','locksmiths','landscaping','tree_service','painting',
        // new fleet-heavy residential/commercial services
        'garage_doors','gutters','fencing','pool_service','septic','irrigation','solar_install'
      ]
    },
    {
      name: 'Logistics / Mobility',
      keys: [
        'moving','courier','rental_fleets',
        // new logistics
        'trucking_companies','charter_bus','shuttle_transport'
      ]
    },
    {
      name: 'Industrial / Construction Ops',
      keys: [
        'hardware_supply','waste_dumpster','excavate_pave_concrete',
        // new industrial
        'equipment_rental','forklift_service','portable_toilets','storage_containers','line_strip_marking','snow_removal','sand_gravel_delivery','crane_services','mobile_welding'
      ]
    },
    {
      name: 'Recreation',
      keys: [
        'golf_courses',
        'campgrounds',
        { label: 'Marinas & Boat Repair', combine: ['marinas','marina_services','boat_repair'] },
        { label: 'Powersports (Dealers + Repair)', combine: ['powersports_dealers','powersports_repair'] }
      ]
    },
  ];

  for (const g of groupDefs) {
    const groupWrap = document.createElement('div');
    groupWrap.className = 'group';

    const header = document.createElement('div');
    header.className = 'group-header';
    header.innerHTML = `
      <div class="left">
        <button class="collapse-btn" title="Collapse/Expand" aria-label="Collapse/Expand">▸</button>
        <div class="group-title">${g.name}</div>
      </div>
      <div class="group-controls">
        <button class="group-all">All</button>
        <button class="group-none">None</button>
      </div>
    `;
    groupWrap.appendChild(header);

    const chips = document.createElement('div');
    chips.className = 'group-chips';
    // Support plain pack keys and composite entries {label, combine: [...keys]}
    const toEntries = (g.keys || []).map(entry => {
      if (typeof entry === 'string') {
        if (!byKey.has(entry)) return null;
        const pack = byKey.get(entry);
        return { type: 'single', key: pack.key, label: pack.label };
      }
      if (entry && Array.isArray(entry.combine)) {
        const keys = entry.combine.filter(k => byKey.has(k));
        if (keys.length === 0) return null;
        return { type: 'combo', keys, label: entry.label || keys.join(', ') };
      }
      return null;
    }).filter(Boolean);

    toEntries.forEach(ent => {
      const pill = document.createElement('label');
      pill.className = 'category-pill';
      const id = ent.type === 'single' ? `cat_${ent.key}` : `combo_${ent.label.replace(/[^a-z0-9]+/gi,'_')}`;
      pill.title = ent.label;
      pill.innerHTML = `
        <input type="checkbox" id="${id}" ${ent.type==='single' ? `value="${ent.key}"` : ''} ${ent.type==='combo' ? `data-combine="${ent.keys.join(',')}"` : ''}>
        <span>${ent.label}</span>
      `;
      const input = pill.querySelector('input');
      // Initialize checked state: if none selected yet, do not auto-check; we let global Select All handle defaults
      if (ent.type === 'single') {
        input.checked = selectedCategories.has(ent.key);
        input.addEventListener('change', () => {
          if (input.checked) selectedCategories.add(ent.key); else selectedCategories.delete(ent.key);
        });
      } else {
        // combo: checked if all underlying keys are selected
        const allSelected = ent.keys.every(k => selectedCategories.has(k));
        input.checked = allSelected;
        input.addEventListener('change', () => {
          if (input.checked) {
            ent.keys.forEach(k => selectedCategories.add(k));
          } else {
            ent.keys.forEach(k => selectedCategories.delete(k));
          }
          // Also update sibling single checkboxes to reflect changes
          ent.keys.forEach(k => {
            const cb = chips.querySelector(`input[value="${k}"]`);
            if (cb) cb.checked = selectedCategories.has(k);
          });
        });
      }
      chips.appendChild(pill);
    });
    groupWrap.appendChild(chips);

    // wire group toggles
    const btnAll = header.querySelector('.group-all');
    const btnNone = header.querySelector('.group-none');
    btnAll.addEventListener('click', () => {
      // Select all singles and all combos (expands combos to their keys)
      chips.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.checked = true; });
      toEntries.forEach(ent => {
        if (ent.type === 'single') selectedCategories.add(ent.key);
        else ent.keys.forEach(k => selectedCategories.add(k));
      });
    });
    btnNone.addEventListener('click', () => {
      chips.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.checked = false; });
      toEntries.forEach(ent => {
        if (ent.type === 'single') selectedCategories.delete(ent.key);
        else ent.keys.forEach(k => selectedCategories.delete(k));
      });
    });

    // Accordion collapse behavior
    const collapseBtn = header.querySelector('.collapse-btn');
    const toggleCollapsed = () => {
      groupWrap.classList.toggle('collapsed');
      setGroupCollapsed(g.name, groupWrap.classList.contains('collapsed'));
    };
    collapseBtn.addEventListener('click', toggleCollapsed);
    // Default collapsed on all screens, but respect saved preference
    const saved = getGroupCollapsed(g.name);
    if (saved === null) {
      groupWrap.classList.add('collapsed');
      setGroupCollapsed(g.name, true);
    } else if (saved) {
      groupWrap.classList.add('collapsed');
    }

    els.categories.appendChild(groupWrap);
  }
}

function getRequestBody() {
  const centerText = els.centerText.value.trim();
  const milesVal = parseFloat(els.radius.value);
  const radiusMiles = !isNaN(milesVal) ? milesVal : 5;
  const metersFromMiles = Math.max(1, Math.round(radiusMiles * 1609.34));
  const exclude = !!els.excludeSAB.checked;
  const categories = Array.from(selectedCategories);
  // Prefer drawn center/radius if available
  let center;
  if (centerOverride && typeof centerOverride.lat === 'number' && typeof centerOverride.lng === 'number') {
    center = { lat: centerOverride.lat, lng: centerOverride.lng };
  } else {
    center = centerText ? { text: centerText } : { text: 'Detroit, MI' };
  }
  const body = {
    center,
    // If a circle was drawn, radiusOverride is in meters; otherwise use converted miles input
    radiusMeters: radiusOverride || metersFromMiles,
    categories,
    excludeServiceAreaOnly: exclude,
    maxResults: 500,
    highRecall: !!(els.highRecallToggle && els.highRecallToggle.checked),
  };
  return body;
}

function clientFilter(results) {
  const minRating = els.minRating ? parseFloat(els.minRating.value) : NaN;
  const minReviews = els.minReviews ? parseInt(els.minReviews.value, 10) : NaN;
  return results.filter(r => {
    // Filter by selected primary types if any selected (but if all types are selected, skip filter)
    if (selectedTypes.size > 0) {
      if (lastResponse && Array.isArray(lastResponse.results)) {
        const allTypesSet = new Set(lastResponse.results.map(x => x.primaryType).filter(Boolean));
        if (lastResponse.results.some(x => !x.primaryType)) allTypesSet.add('__other__');
        // If the selection covers all known types, skip type filtering
        let coversAll = true;
        for (const t of allTypesSet) { if (!selectedTypes.has(t)) { coversAll = false; break; } }
        if (coversAll) {
          // no-op: skip type filter
        } else {
          const pt = r.primaryType || '__other__';
          if (!selectedTypes.has(pt)) return false;
        }
      } else {
        const pt = r.primaryType || '__other__';
        if (!selectedTypes.has(pt)) return false;
      }
    }
    // Filter by selected packs if any selected
    if (selectedPacks.size > 0) {
      const packs = Array.isArray(r.categories) ? r.categories : [];
      const hasAny = packs.some(p => selectedPacks.has(p));
      if (!hasAny) return false;
    }
    if (!isNaN(minRating) && (r.rating ?? 0) < minRating) return false;
    if (!isNaN(minReviews) && (r.userRatingCount ?? 0) < minReviews) return false;
    return true;
  });
}

function renderResults(list) {
  els.resultsList.innerHTML = '';
  els.resultsCount.textContent = String(list.length);

  markersLayer.clearLayers();

  const bounds = [];
  const cLat = lastResponse?.centerLat;
  const cLng = lastResponse?.centerLng;
  if (!list.length){
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.innerHTML = '<div class="icon">🔍</div><div>No businesses found matching your criteria</div>';
    els.resultsList.appendChild(empty);
    return;
  }
  list.forEach((r, idx) => {
    const row = document.createElement('div');
    row.className = 'result-row';
    const rawPhone = (r.phone || '').trim();
    const phone = rawPhone ? `<a href="tel:${rawPhone.replace(/[^+\d]/g,'')}">${rawPhone}</a>` : '-';
    const rawWebsite = (r.website || '').trim();
    const website = rawWebsite ? `<a href="${rawWebsite}" target="_blank" rel="noopener">Visit Site</a>` : '-';
    const cat = getCategoryLabel(r);
    // Build category badges: prefer pack labels from r.categories; fallback to primaryType mapping
    let badgesHTML = '';
    if (Array.isArray(r.categories) && r.categories.length > 0) {
      const labels = r.categories.slice(0, 4); // cap to 4 badges for space
      badgesHTML = labels.map(lbl => `<span class="badge ${mapPackLabelToClass(lbl)}">${lbl}</span>`).join(' ');
    } else {
      badgesHTML = `<span class="badge ${cat.css}">${cat.label}</span>`;
    }
    const distNum = (typeof r.lat === 'number' && typeof r.lng === 'number' && typeof cLat === 'number' && typeof cLng === 'number')
      ? (haversineMiles(cLat, cLng, r.lat, r.lng))
      : null;
    const distMiles = distNum == null ? '' : (distNum.toFixed(1) + ' mi');
    const ap = parseAddressParts(r.formattedAddress || '');
    row.innerHTML = `
      <div class="cell name">${r.name || ''}</div>
      <div class="cell category"><span class="badges">${badgesHTML}</span></div>
      <div class="cell addr">${ap.street || ''}</div>
      <div class="cell city">${ap.city || ''}</div>
      <div class="cell state">${ap.state || ''}</div>
      <div class="cell zip">${ap.zip || ''}</div>
      <div class="cell phone">${phone}</div>
      <div class="cell website">${website}</div>
      <div class="cell distance">${distMiles}</div>
    `;
    els.resultsList.appendChild(row);

    if (typeof r.lat === 'number' && typeof r.lng === 'number') {
      const m = L.marker([r.lat, r.lng]).bindPopup(`<strong>${r.name || ''}</strong><br>${r.formattedAddress || ''}`);
      markersLayer.addLayer(m);
      bounds.push([r.lat, r.lng]);
    }
  });

  if (bounds.length) {
    map.fitBounds(bounds, { padding: [20, 20] });
  }
}

async function doSearch() {
  els.searchBtn.disabled = true;
  els.exportCsvBtn.disabled = true;
  els.loadMoreBtn.disabled = true;
  try {
    const body = getRequestBody();
    const res = await fetch(`${apiBase}/search/places`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`Search failed: ${res.status}`);
    const data = await res.json();
    lastResponse = data;
    nextPageToken = data.nextPageToken || null;
    buildTypeChips(lastResponse.results || []);
    buildPackChips(lastResponse.results || []);
    const filtered = clientFilter(lastResponse.results || []);
    const sorted = sortList(filtered);
    renderResults(sorted);
    els.exportCsvBtn.disabled = false;
    els.loadMoreBtn.disabled = !nextPageToken;
  } catch (err) {
    console.error(err);
    alert('Search failed. See console for details.');
  } finally {
    els.searchBtn.disabled = false;
  }
}

async function loadMore() {
  if (!nextPageToken) return;
  els.loadMoreBtn.disabled = true;
  try {
    const res = await fetch(`${apiBase}/search/places/next?token=${encodeURIComponent(nextPageToken)}`);
    if (!res.ok) throw new Error(`Load more failed: ${res.status}`);
    const data = await res.json();
    nextPageToken = data.nextPageToken || null;
    // Append results to lastResponse
    const existing = lastResponse?.results || [];
    const mergedById = new Map();
    for (const r of existing) mergedById.set(r.placeId, r);
    for (const r of (data.results || [])) mergedById.set(r.placeId, r);
    const merged = Array.from(mergedById.values());
    lastResponse.results = merged;
    // Rebuild type chips including any new primary types
    buildTypeChips(lastResponse.results || []);
    buildPackChips(lastResponse.results || []);
    const filtered = clientFilter(merged);
    const sorted = sortList(filtered);
    renderResults(sorted);
  } catch (err) {
    console.error(err);
    alert('Load more failed. See console for details.');
  } finally {
    els.loadMoreBtn.disabled = !nextPageToken;
  }
}

async function enrichDetails(items) {
  const ids = (items || []).map(r => r.placeId).filter(Boolean);
  if (!ids.length) return;
  try {
    const res = await fetch(`${apiBase}/places/details`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ placeIds: ids.slice(0, 50) }),
    });
    if (!res.ok) return;
    const data = await res.json();
    const details = data.details || {};
    for (const item of items) {
      const d = details[item.placeId];
      if (d) {
        item.phone = d.phone || item.phone;
        item.website = d.website || item.website;
      }
    }
  } catch (_) {
    // ignore details errors
  }
}

async function exportCsv() {
  const body = getRequestBody();
  try {
    // Build query for primary type filters if any selected
    let requestUrl = `${apiBase}/search/places/csv`;
    if (selectedTypes.size > 0) {
      const params = Array.from(selectedTypes).map(t => `filterPrimaryTypes=${encodeURIComponent(t)}`).join('&');
      requestUrl += `?${params}`;
    }
    const res = await fetch(requestUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`CSV export failed: ${res.status}`);
    const blob = await res.blob();
    const blobUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = blobUrl;
    a.download = 'places_export.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(blobUrl);
  } catch (err) {
    console.error(err);
    alert('CSV export failed. See console for details.');
  }
}

function setDrawStatus(text) {
  els.drawStatus.textContent = text;
}

function toggleDrawMode() {
  drawMode = !drawMode;
  if (drawMode) {
    setDrawStatus('Click on the map to place/move the radius');
  } else {
    setDrawStatus('');
  }
}

function wireEvents() {
  if (els.searchBtn) els.searchBtn.addEventListener('click', doSearch);
  if (els.applyFiltersBtn) {
    els.applyFiltersBtn.addEventListener('click', () => {
      if (!lastResponse) return;
      const filtered = clientFilter(lastResponse.results || []);
      const sorted = sortList(filtered);
      renderResults(sorted);
    });
  }
  if (els.loadMoreBtn) els.loadMoreBtn.addEventListener('click', loadMore);
  if (els.exportCsvBtn) els.exportCsvBtn.addEventListener('click', exportCsv);
  if (els.drawRadiusBtn) els.drawRadiusBtn.addEventListener('click', toggleDrawMode);
  if (els.toggleSidebarBtn) {
    const setLabel = () => {
      els.toggleSidebarBtn.textContent = document.body.classList.contains('sidebar-collapsed') ? 'Show Sidebar' : 'Hide Sidebar';
    };
    setLabel();
    els.toggleSidebarBtn.addEventListener('click', () => {
      document.body.classList.toggle('sidebar-collapsed');
      setLabel();
    });
  }
  if (els.clearRadiusBtn) els.clearRadiusBtn.addEventListener('click', () => {
    if (drawCircle) { try { map.removeLayer(drawCircle); } catch(_) {} drawCircle = null; }
    centerOverride = null; radiusOverride = null; setDrawStatus('');
  });
  if (els.resetFiltersBtn) els.resetFiltersBtn.addEventListener('click', () => {
    // Clear packs/types selections
    selectedCategories.clear();
    selectedTypes.clear();
    selectedPacks.clear();
    // Uncheck all inputs
    document.querySelectorAll('#categories input[type="checkbox"]').forEach(cb => cb.checked = false);
    if (els.typeChips) els.typeChips.innerHTML = '';
    if (els.packChips) els.packChips.innerHTML = '';
    // Rebuild chips from current data if any
    if (lastResponse) {
      buildTypeChips(lastResponse.results || []);
      buildPackChips(lastResponse.results || []);
      const filtered = clientFilter(lastResponse.results || []);
      const sorted = sortList(filtered);
      renderResults(sorted);
    }
  });
  if (els.densityToggleBtn) els.densityToggleBtn.addEventListener('click', () => {
    document.body.classList.toggle('compact-density');
  });
  // Live radius updates: if a circle exists, update its radius when input changes
  els.radius.addEventListener('input', () => {
    if (!drawCircle || !centerOverride) return;
    const milesVal = parseFloat(els.radius.value);
    const radiusMiles = !isNaN(milesVal) ? milesVal : 5;
    const meters = Math.max(1, Math.round(radiusMiles * 1609.34));
    radiusOverride = meters;
    drawCircle.setRadius(meters);
    // Keep the circle in view
    try {
      map.fitBounds(drawCircle.getBounds(), { padding: [20, 20] });
    } catch (_) {}
  });
  // Global select all/none (packs/groups)
  els.catSelectAll.addEventListener('click', () => {
    document.querySelectorAll('#categories input[type="checkbox"]').forEach(cb => {
      cb.checked = true;
      const val = cb.getAttribute('value');
      const combine = cb.getAttribute('data-combine');
      if (val) {
        selectedCategories.add(val);
      } else if (combine) {
        combine.split(',').forEach(k => { if (k) selectedCategories.add(k); });
      }
    });
  });
  els.catSelectNone.addEventListener('click', () => {
    document.querySelectorAll('#categories input[type="checkbox"]').forEach(cb => {
      cb.checked = false;
      const val = cb.getAttribute('value');
      const combine = cb.getAttribute('data-combine');
      if (val) {
        selectedCategories.delete(val);
      } else if (combine) {
        combine.split(',').forEach(k => { if (k) selectedCategories.delete(k); });
      }
    });
  });
  // Primary type filter actions
  els.typesSelectAll.addEventListener('click', () => {
    const chips = els.typeChips.querySelectorAll('.chip input[type="checkbox"]');
    chips.forEach(cb => { cb.checked = true; selectedTypes.add(cb.value); });
    if (lastResponse) {
      const filtered = clientFilter(lastResponse.results || []);
      const sorted = sortList(filtered);
      renderResults(sorted);
    }
    updateTypeCount();
  });
  els.typesSelectNone.addEventListener('click', () => {
    const chips = els.typeChips.querySelectorAll('.chip input[type="checkbox"]');
    chips.forEach(cb => { cb.checked = false; selectedTypes.delete(cb.value); });
    if (lastResponse) {
      const filtered = clientFilter(lastResponse.results || []);
      const sorted = sortList(filtered);
      renderResults(sorted);
    }
    updateTypeCount();
  });

  wireSorting();
  wireToggles();
}

(async function init() {
  initMap();
  wireEvents();
  try {
    await fetchCategories();
  } catch (e) {
    console.error(e);
  }
})();
