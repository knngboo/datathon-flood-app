/* Texas Flood Check on GitHub Pages: what server.py does on the Dell, done here in the browser.
   Same ranking, hotspots and data sources (FEMA, NOAA, USGS, OpenStreetMap, the organizers' photos).
   Requests and feedback are saved in this browser only. The AI model ran on the Dell, so where the AI
   would write, this copy shows the app's own backup text (what the app shows when the AI is down). */
(() => {
  const realFetch = window.fetch.bind(window);
  const HF = 'https://huggingface.co/datasets/jennifer-fowler/UTSA-flood-images/resolve/main/';
  const REQ = 'tfc.requests', FEED = 'tfc.feedback', HIST = 'tfc.history';
  const store = {
    get: (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set: (k, v) => localStorage.setItem(k, JSON.stringify(v)),
  };

  let PLACES = [], PATHS = {}, THUMBS = new Set();
  const CACHE = { history: {} };
  const ready = Promise.all([
    realFetch('places.json').then(r => r.json()),
    realFetch('demo.json').then(r => r.json()).catch(() => ({})),
  ]).then(([places, demo]) => {
    PLACES = places;
    for (const p of places) { PATHS[p.id] = p.path; if (p.before) PATHS[p.before] = p.before_path; }
    Object.assign(CACHE.history, demo.history || {}, store.get(HIST, {}));
    THUMBS = new Set(demo.thumbs || []);
  });
  window.imgUrl = id => THUMBS.has(id) ? `photos/${id}.jpg` : HF + (PATHS[id] || `images/${id}.png`);

  // ---------- helpers ----------
  const rnd = n => x => Math.round(x * 10 ** n) / 10 ** n;
  const r1 = rnd(1), r2 = rnd(2), r3 = rnd(3), r4 = rnd(4), r5 = rnd(5);
  const sleep = ms => new Promise(ok => setTimeout(ok, ms));
  const qs = o => new URLSearchParams(o).toString();
  const km = (lat1, lon1, lat2, lon2) => {
    const t = Math.PI / 180, a = Math.sin((lat2 - lat1) * t / 2) ** 2 + Math.cos(lat1 * t) * Math.cos(lat2 * t) * Math.sin((lon2 - lon1) * t / 2) ** 2;
    return 6371 * 2 * Math.asin(Math.sqrt(a));
  };
  const nearest = (lat, lon) => PLACES.reduce((b, p) => km(lat, lon, p.lat, p.lon) < km(lat, lon, b.lat, b.lon) ? p : b);
  const title = s => s.toLowerCase().replace(/(^|[^a-z])([a-z])/g, (m, a, b) => a + b.toUpperCase());  // like Python's str.title()
  async function jget(url, ms = 15000) {
    const r = await realFetch(url, { signal: AbortSignal.timeout(ms) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }
  const memo = new Map();  // like lru_cache: answers are kept, errors are not
  const once = (key, fn) => {
    if (!memo.has(key)) memo.set(key, fn().catch(e => { memo.delete(key); throw e; }));
    return memo.get(key);
  };
  const txZip = z => { z = (z || '').slice(0, 5); return /^\d{5}$/.test(z) && (z.startsWith('7') || z.startsWith('885')) ? z : ''; };

  // ---------- addresses (OpenStreetMap: Nominatim + Photon) ----------
  const SUFFIX = '\\b(street|st|drive|dr|road|rd|lane|ln|avenue|ave|boulevard|blvd|court|ct|circle|cir|way|trail|trl|parkway|pkwy)\\b\\.?';
  const noSuffix = s => s.replace(new RegExp(SUFFIX, 'gi'), '');
  const nominatim = (q, limit = 5) => once(`n|${q}`, async () =>
    (await jget('https://nominatim.openstreetmap.org/search?' + qs({ q, format: 'json', limit, countrycodes: 'us', addressdetails: 0 }), 6000))
      .map(h => [+h.lat, +h.lon, h.display_name]));

  async function suggest(q) {
    q = q.trim().replace(/\s+/g, ' ');
    const num = q.match(/^(\d+)\s/);
    const words = noSuffix(q.replace(/^\d+\s+/, '')).replace(/,/g, ' ').split(/\s+/).filter(w => w.length > 2 && !['tx', 'texas'].includes(w.toLowerCase()));
    const street = words.length ? words[0].toLowerCase() : '';
    const tx = /\b(tx|texas)\b/i.test(q) ? q : q + ', Texas';
    const bare = noSuffix(tx).replace(/\s+/g, ' ').trim().replace(/^[ ,]+|[ ,]+$/g, '');
    const tries = [...new Set([tx, bare, bare.replace(/^\d+\s+/, '')])];
    const found = new Map();
    for (const [i, t] of tries.entries()) {
      if (i) await sleep(1000);  // Nominatim allows one request a second
      for (const [lat, lon, label] of await nominatim(t)) {
        if (!label.includes('Texas') || (street && !label.toLowerCase().includes(street))) continue;
        const short = label.split(',').map(x => x.trim()).filter(x => x !== 'United States').slice(0, 4).join(', ');
        const rank = (num && label.startsWith(num[1]) ? 2 : 0) + (street ? 1 : 0);
        if (!found.has(short)) found.set(short, { label: num ? short.replaceAll(`${num[1]}, `, `${num[1]} `) : short, lat, lon, rank });
      }
      if ([...found.values()].some(v => v.rank >= 2)) break;
    }
    return [...found.values()].sort((a, b) => b.rank - a.rank).slice(0, 5).map(({ rank, ...o }) => o);
  }

  const photon = q => once(`p|${q}`, async () => {
    const feats = (await jget('https://photon.komoot.io/api/?' + qs({ q, limit: 6, lang: 'en', bbox: '-106.65,25.84,-93.51,36.5' }), 8000)).features;
    const out = new Map();
    for (const f of feats) {
      const p = f.properties;
      if (p.state !== 'Texas') continue;
      const line1 = [p.housenumber, p.street || p.name].filter(Boolean).join(' ').trim();
      const line2 = [p.city || p.county, 'TX', txZip(p.postcode)].filter(Boolean).join(', ');
      const [lon, lat] = f.geometry.coordinates, label = line1 ? `${line1}, ${line2}` : line2;
      out.set(label, { label, lat, lon });
    }
    return [...out.values()].slice(0, 5);
  });

  const photonReverse = (lat, lon) => once(`pr|${lat},${lon}`, async () => {
    const feats = (await jget('https://photon.komoot.io/reverse?' + qs({ lat, lon, lang: 'en' }), 8000)).features || [];
    const p = feats.length ? feats[0].properties : {};
    return { road: p.street || p.name || '', house_number: p.housenumber || '', city: p.city || '', county: p.county || '', postcode: p.postcode || '', state: p.state || '' };
  });
  const nearLabel = (line1, line2) => line1 ? `Near ${line1}, ${line2}` : `Near ${line2}`;
  async function reverse(lat, lon) {
    try {
      return await once(`nr|${lat},${lon}`, async () => {
        const a = (await jget('https://nominatim.openstreetmap.org/reverse?' + qs({ lat, lon, format: 'json', zoom: 17, addressdetails: 1 }), 10000)).address || {};
        return nearLabel([a.house_number, a.road].filter(Boolean).join(' ').trim(),
          [a.city || a.town || a.village || a.county, 'TX', txZip(a.postcode)].filter(Boolean).join(', '));
      });
    } catch {
      const a = await photonReverse(lat, lon);
      return nearLabel([a.house_number, a.road].filter(Boolean).join(' ').trim(), [a.city || a.county, 'TX', txZip(a.postcode)].filter(Boolean).join(', '));
    }
  }
  async function whereIs(lat, lon) {
    try {
      return await once(`w|${lat},${lon}`, async () => {
        const a = (await jget('https://nominatim.openstreetmap.org/reverse?' + qs({ lat, lon, format: 'json', zoom: 16, addressdetails: 1 }), 8000)).address || {};
        return [(a.county || '').replace(' County', ''), txZip(a.postcode), a.state || ''];
      });
    } catch {
      const a = await photonReverse(lat, lon);
      return [a.county.replace(' County', ''), txZip(a.postcode), a.state];
    }
  }

  // ---------- flood history (OpenFEMA), flood zone (NFHL), river gauges (NOAA, USGS) ----------
  const FLOOD_TYPES = ['Flood', 'Hurricane', 'Severe Storm', 'Severe Storm(s)', 'Coastal Storm', 'Tropical Storm'];
  const countyDeclarations = county => once(`c|${county}`, async () => {
    const rows = (await jget('https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries?' + qs({
      $filter: `state eq 'TX' and designatedArea eq '${county} (County)'`,
      $select: 'disasterNumber,declarationTitle,incidentType,declarationDate,declarationType', $orderby: 'declarationDate desc', $top: 500 }), 20000)).DisasterDeclarationsSummaries;
    const seen = new Map();
    for (const x of rows) if (FLOOD_TYPES.includes(x.incidentType) && !seen.has(x.disasterNumber)) seen.set(x.disasterNumber, x);
    return [...seen.values()].map(x => ({ number: x.disasterNumber, year: +x.declarationDate.slice(0, 4), title: title(x.declarationTitle), type: x.incidentType, major: x.declarationType === 'DR' }));
  });
  const fema = (zip, disaster) => once(`f|${zip}|${disaster}`, async () => {
    const out = {};
    await Promise.all(['Renters', 'Owners'].map(async kind => {
      const rows = (await jget(`https://www.fema.gov/api/open/v2/HousingAssistance${kind}?` + qs({ $filter: `disasterNumber eq ${disaster} and zipCode eq '${zip}'`, $top: 100 }), 20000))[`HousingAssistance${kind}`] || [];
      out[kind.toLowerCase()] = { applied: rows.reduce((t, x) => t + (x.validRegistrations || 0), 0) };
    }));
    return out;
  });

  async function computeHistory(lat, lon) {
    const [county, zip, state] = await whereIs(r3(lat), r3(lon));
    if (state !== 'Texas' || !county) return null;
    const events = await countyDeclarations(county);
    if (!events.length) return { county, zip, total: 0, events: [] };
    const thisYear = new Date().getFullYear(), decades = {};
    for (const e of events) { const d = `${Math.floor(e.year / 10) * 10}s`; decades[d] = (decades[d] || 0) + 1; }
    // Households that applied to FEMA, for up to 12 major floods since 2004 (4 at a time; the Dell asked one by one)
    const want = events.filter(e => zip && e.major && e.year >= 2004), got = new Map();
    for (let i = 0, answered = 0; i < want.length && answered < 12; i += 4) {
      const batch = want.slice(i, i + 4);
      const ns = await Promise.all(batch.map(e => fema(zip, e.number).then(f => f.renters.applied + f.owners.applied).catch(() => null)));
      batch.forEach((e, j) => { got.set(e.number, ns[j]); if (ns[j] != null) answered++; });
    }
    const graded = [], severity = [];
    let have = 0;
    for (const e of events) {
      const n = zip && e.major && e.year >= 2004 && have < 12 ? got.get(e.number) ?? null : null;
      if (n != null) have++;
      const cls = !e.major ? 'Minor' : (n || 0) >= 5000 ? 'Severe' : (n || 0) >= 100 ? 'Major' : 'Moderate';
      graded.push({ ...e, households: n, class: cls });
      if (n) severity.push({ year: e.year, title: e.title, households: n });
    }
    const classes = Object.fromEntries(['Minor', 'Moderate', 'Major', 'Severe'].map(c => [c, graded.filter(g => g.class === c).length]));
    const years = events.map(e => e.year);
    return { county, zip, total: events.length, since: Math.min(...years), last10: years.filter(y => y >= thisYear - 10).length, last_year: Math.max(...years),
      decades: Object.fromEntries(Object.entries(decades).sort()), severity: severity.slice(0, 6), classes, major_count: classes.Major + classes.Severe, events: graded };
  }
  async function floodHistory(lat, lon) {
    const key = `${r3(lat)},${r3(lon)}`, c = CACHE.history[key];
    if (key in CACHE.history && (c === null || 'classes' in c)) return c;
    const h = await computeHistory(lat, lon);
    CACHE.history[key] = h;
    store.set(HIST, { ...store.get(HIST, {}), [key]: h });
    return h;
  }
  const historyReady = (lat, lon) => `${r3(lat)},${r3(lon)}` in CACHE.history;
  function historyPoints(h) {
    if (!h || !h.total) return [0, null];
    let pts = h.last10 >= 5 ? 3 : h.last10 >= 2 ? 2 : 1;
    const worst = Math.max(0, ...h.severity.map(x => x.households));
    pts += worst >= 1000 ? 3 : worst >= 100 ? 1 : 0;
    return [pts, `Flood-prone area: ${h.last10} recorded flood events in ${h.county} County in the last 10 years`];
  }

  const floodZone = (lat, lon) => once(`z|${lat},${lon}`, async () => {
    const url = 'https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query?' + qs({ geometry: `${lon},${lat}`, geometryType: 'esriGeometryPoint', inSR: 4326,
      spatialRel: 'esriSpatialRelIntersects', outFields: 'FLD_ZONE,ZONE_SUBTY,SFHA_TF', returnGeometry: 'false', f: 'json' }).replace(/%2C/g, ',');
    const feats = (await jget(url, 12000)).features || [];
    if (!feats.length) return null;
    const a = feats[0].attributes, high = a.SFHA_TF === 'T';
    return { zone: a.FLD_ZONE, high_risk: high, label: high ? 'High-risk flood zone (1% chance of flooding every year)' : "Outside FEMA's high-risk flood zone" };
  });

  const GAUGES = { t: 0, all: [], busy: null };  // NOAA allows 10 requests per 5 minutes: one big area at most every 5 minutes
  async function allGauges() {
    if (Date.now() - GAUGES.t < 300000 && GAUGES.all.length) return GAUGES.all;
    GAUGES.busy = GAUGES.busy || (async () => {
      try {
        const gs = (await jget('https://api.water.noaa.gov/nwps/v1/gauges?' + qs({ 'bbox.xmin': -100.6, 'bbox.ymin': 28.4, 'bbox.xmax': -93.4, 'bbox.ymax': 31.6, srid: 'EPSG_4326' }), 20000)).gauges || [];
        const out = gs.filter(g => g.latitude != null).map(g => {
          const obs = (g.status || {}).observed || {};
          return { name: g.name, id: g.lid, lat: g.latitude, lon: g.longitude, category: obs.floodCategory || 'unknown', stage: obs.primary, unit: obs.primaryUnit || 'ft' };
        });
        if (out.length) GAUGES.all = out;
      } catch { /* back off for 5 minutes instead of hammering a rate-limited service */ }
      GAUGES.t = Date.now();
    })().finally(() => { GAUGES.busy = null; });
    await GAUGES.busy;
    return GAUGES.all;
  }

  const USGS = {};
  const ABBR = { ck: 'Creek', cr: 'Creek', nr: 'near', rv: 'River', r: 'River', abv: 'above', blw: 'below', fk: 'Fork', at: 'at', w: 'West', e: 'East', n: 'North', s: 'South',
    bayou: 'Bayou', trib: 'tributary', dr: 'Drive', rd: 'Road', hwy: 'Highway', st: 'Street', lk: 'Lake', res: 'Reservoir', tx: 'TX' };
  const cased = w => /[a-z]/i.test(w);
  const plainSite = name => name.split(/(\W+)/).map(w => ABBR[w.toLowerCase()] ?? (cased(w) && (w === w.toUpperCase() || w === w.toLowerCase()) ? title(w) : w)).join('');
  async function usgsGauges(lat, lon) {
    const key = `${r2(lat)},${r2(lon)}`;
    if (USGS[key] && Date.now() - USGS[key][0] < 300000) return USGS[key][1];
    const bbox = [lon - .15, lat - .15, lon + .15, lat + .15].map(v => v.toFixed(3)).join(',');
    const series = (await jget('https://waterservices.usgs.gov/nwis/iv/?' + qs({ format: 'json', parameterCd: '00065', siteStatus: 'active', period: 'PT3H' }) + '&bBox=' + bbox, 12000)).value.timeSeries;
    const out = [];
    for (const ts of series) {
      const vals = ts.values[0].value.filter(v => v.value != null && v.value !== '' && v.value !== '-999999');
      if (vals.length < 2) continue;
      const g = ts.sourceInfo.geoLocation.geogLocation, last = +vals[vals.length - 1].value, change = r2(last - +vals[0].value);
      out.push({ name: plainSite(ts.sourceInfo.siteName), km: r1(km(lat, lon, g.latitude, g.longitude)), source: 'USGS', stage: last, unit: 'ft', change,
        category: change >= .2 ? 'rising' : change <= -.2 ? 'falling' : 'steady' });
    }
    const near = out.filter(g => g.km <= 20).sort((a, b) => a.km - b.km).slice(0, 3);
    USGS[key] = [Date.now(), near];
    return near;
  }
  async function liveGauges(lat, lon) {
    const noaa = (await allGauges()).map(g => ({ ...g, km: r1(km(lat, lon, g.lat, g.lon)), source: 'NOAA' })).filter(g => g.km <= 20).sort((a, b) => a.km - b.km).slice(0, 3);
    if (noaa.length) return noaa;
    try { return await usgsGauges(lat, lon); } catch { return []; }
  }
  async function historyParts(lat, lon) {
    let fh = null, g = [];
    try { fh = await floodHistory(lat, lon); } catch { }
    let [pts, why] = fh ? historyPoints(fh) : [0, null];
    try { g = await liveGauges(lat, lon); } catch { }
    const flooding = g.filter(x => ['minor', 'moderate', 'major'].includes(x.category));
    const rising = g.filter(x => x.category === 'rising' && (x.change || 0) >= 1);
    if (flooding.length) {
      pts += { minor: 2, moderate: 3, major: 4 }[flooding[0].category];
      why = (why ? why + '; ' : '') + `Live river gauge in ${flooding[0].category} flood (NOAA)`;
    } else if (rising.length) {
      pts += 2;
      why = (why ? why + '; ' : '') + `River rising fast nearby: ${rising[0].change} ft in 3 hours (USGS)`;
    }
    return [pts, why];
  }

  // ---------- requests ----------
  const EMERGENCY = [['danger', 'In immediate danger', 40, 20], ['trapped', 'Trapped', 30, 15], ['injured', 'Someone injured', 25, 12],
    ['rising', 'Water rising', 15, 8], ['vulnerable', 'Elderly or disabled person', 10, 5]];
  function score(a, people, histPts, histWhy, verdict) {
    let pts = 0;
    const why = [];
    for (const [k, label, yes, unsure] of EMERGENCY) {
      if (a[k] === 'yes') { pts += yes; why.push(label); }
      else if (a[k] === 'unsure') { pts += unsure; why.push('Not sure: ' + label[0].toLowerCase() + label.slice(1)); }
    }
    if (a.evacuate === 'no') { pts += 15; why.push("Can't leave safely"); }
    else if (a.evacuate === 'unsure') { pts += 8; why.push('Not sure: can leave safely'); }
    pts += Math.min(Math.max(people - 1, 0), 4) * 2;
    if (histPts) { pts += histPts; why.push(histWhy); }
    if (verdict === 'Yes') { pts += 2; why.push('Aerial photo shows this area flooded before'); }
    const level = a.danger === 'yes' || (a.trapped === 'yes' && a.rising === 'yes') || pts >= 65 ? 'Critical' : pts >= 35 ? 'High' : pts >= 15 ? 'Medium' : 'Low';
    return [pts, level, why];
  }
  const LEVELS = { Critical: 0, High: 1, Medium: 2, Low: 3 }, lv = l => LEVELS[l] ?? 9;
  const readRequests = () => store.get(REQ, []).sort((a, b) => (a.status === 'Resolved') - (b.status === 'Resolved') || lv(a.level) - lv(b.level) || b.score - a.score || (a.time > b.time) - (a.time < b.time));
  const writeRequests = rows => store.set(REQ, rows);
  const now = () => new Date().toTimeString().slice(0, 5);

  async function makeReport(lat, lon, label, a, people, name, phone, sample = false, histAt = null) {
    const p = nearest(lat, lon);
    const hist = { id: p.id, place: p.place, event: p.event_name, event_key: p.event, km: r2(km(lat, lon, p.lat, p.lon)) };
    const [verdict, vtext] = [null, ''];  // the photo check needs the AI on the Dell
    const [hl, ho] = histAt || [lat, lon];
    const pending = !sample && !historyReady(hl, ho);
    const [histPts, histWhy] = pending ? [0, null] : await historyParts(hl, ho);
    const [pts, level, why] = score(a, people, histPts, histWhy, verdict);
    return { time: now(), lat, lon, label, answers: a, people, name, phone, score: pts, level, why, history: hist,
      verdict, verdict_text: vtext, hist_pts: histPts, history_pending: pending, status: 'New', sample };
  }
  function rescoreLater(rid, lat, lon) {
    historyParts(lat, lon).then(([histPts, histWhy]) => {
      const rows = readRequests();
      for (const r of rows) if (r.id === rid) {
        [r.score, r.level, r.why] = score(r.answers, r.people, histPts, histWhy, r.verdict);
        r.hist_pts = histPts; r.history_pending = false;
      }
      writeRequests(rows);
    });
  }
  function saveRequest(r, lat, lon) {
    let rows = readRequests();
    const dup = rows.find(x => x.status !== 'Resolved' && !x.sample && Date.now() / 1000 - (x.ts || 0) < 1800 && km(lat, lon, x.lat, x.lon) < 0.05);
    if (dup) { r.id = dup.id; rows = rows.map(x => x.id === dup.id ? r : x); }
    else { r.id = `R${String(rows.length + 1).padStart(3, '0')}`; rows.push(r); }
    writeRequests(rows);
    return [r, !!dup];
  }

  const SAMPLES = [
    [{ danger: 'yes', rising: 'yes', trapped: 'yes', injured: 'no', evacuate: 'no', vulnerable: 'yes' }, 3],
    [{ danger: 'no', rising: 'yes', trapped: 'no', injured: 'yes', evacuate: 'unsure', vulnerable: 'no' }, 2],
    [{ danger: 'unsure', rising: 'yes', trapped: 'no', injured: 'no', evacuate: 'no', vulnerable: 'yes' }, 4],
    [{ danger: 'no', rising: 'no', trapped: 'no', injured: 'no', evacuate: 'yes', vulnerable: 'no' }, 1],
    [{ danger: 'no', rising: 'unsure', trapped: 'yes', injured: 'no', evacuate: 'no', vulnerable: 'no' }, 2],
    [{ danger: 'no', rising: 'no', trapped: 'no', injured: 'no', evacuate: 'unsure', vulnerable: 'yes' }, 2],
    [{ danger: 'no', rising: 'yes', trapped: 'yes', injured: 'no', evacuate: 'no', vulnerable: 'no' }, 2],
    [{ danger: 'unsure', rising: 'yes', trapped: 'no', injured: 'yes', evacuate: 'no', vulnerable: 'no' }, 5],
  ];
  // Fixed random numbers, so every reset puts the demo requests in the same spots (their flood history is saved in demo.json)
  const seeded = a => () => { a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  async function seed(reset) {
    const rows = reset ? [] : readRequests(), rng = seeded(2026), seen = new Set(), ids = [];
    for (const p of PLACES) if (!seen.has(p.place) && p.before) { seen.add(p.place); ids.push(p.id); }
    const pick = ids.slice(0, 6).concat([ids[0], ids[0]]);  // two more near the first spot, so the demo has a real hotspot
    const spots = SAMPLES.map((s, i) => { const p = PLACES.find(x => x.id === pick[i]); return [p, p.lat + rng() * .003 - .0015, p.lon + rng() * .003 - .0015]; });
    const made = await Promise.all(spots.map(([p, lat, lon], i) => makeReport(lat, lon, `Near ${p.place}`, SAMPLES[i][0], SAMPLES[i][1], '', '', true, [p.lat, p.lon])));
    for (const r of made) { r.id = `R${String(rows.length + 1).padStart(3, '0')}`; rows.push(r); }
    writeRequests(rows);
  }
  let seeding = null;
  const ensureSeeded = () => localStorage.getItem(REQ) != null ? Promise.resolve() : (seeding = seeding || seed(false).finally(() => { seeding = null; }));

  function plainBriefing(rows, groups) {
    const first = groups.length ? `Send one crew to the hotspot: ${groups[0]}. ` : '';
    const taken = new Set(groups.length ? groups[0].split(', ') : []);
    const order = rows.filter(r => r.status !== 'Resolved' && !taken.has(r.id)).map(r => `${r.id} (${r.level})`).join(', ');
    return first + (order ? `Then work down the list in this order: ${order}.` : '');
  }

  // ---------- routes (same paths and answers as server.py) ----------
  const isObj = b => b && typeof b === 'object' && !Array.isArray(b);
  const fail = (ms, what) => new Promise((ok, no) => setTimeout(() => no(new Error(what)), ms));
  async function route(method, url, b) {
    await ready;
    const [path, query] = url.split('?'), q = new URLSearchParams(query || '');
    let m;
    if (path === '/api/places') return [200, PLACES];
    if (path === '/api/suggest') {
      const s = (q.get('q') || '').trim().slice(0, 120);
      if (s.length < 3) return [200, []];
      try { return [200, await photon(s)]; } catch { return [200, []]; }
    }
    if (path === '/api/geo') {
      const s = (q.get('q') || '').trim().slice(0, 120);
      if (s.length < 3) return [400, { error: 'Type at least a street, town or ZIP code.' }];
      // Ask both address services at once: Nominatim is better for exact house numbers, Photon is faster
      const safe = fn => Promise.race([fn(s), fail(12000, 'timeout')]).catch(() => null);
      const [full, quick] = await Promise.all([safe(suggest), safe(photon)]);
      if (full == null && quick == null) return [502, { error: "Address search isn't reachable right now. Use your current location or pick on a map." }];
      const hits = full && full.length ? full : quick || [];
      if (!hits.length) return [404, { error: `No match for "${s}" in Texas. Try the street and town, or a ZIP code.` }];
      return [200, hits];
    }
    if (path === '/api/reverse') {
      try { return [200, { label: await reverse(r5(parseFloat(q.get('lat'))), r5(parseFloat(q.get('lon')))) }]; } catch { return [200, { label: 'Pin on the map' }]; }
    }
    if (path === '/api/history') {
      const lat = parseFloat(q.get('lat')), lon = parseFloat(q.get('lon'));
      let h;
      try { h = await floodHistory(lat, lon); } catch (e) { return [502, { error: `FEMA's records didn't load (${e.message}).` }]; }
      if (!h) return [404, { error: 'No FEMA flood history for this spot (outside Texas or no county found).' }];
      const [zone, gauges] = await Promise.all([floodZone(r4(lat), r4(lon)).catch(() => null), liveGauges(lat, lon).catch(() => [])]);
      // What the app writes when the AI is down: the FEMA numbers, then a fixed warning
      const summary = h.total ? `${h.county} County has ${h.total} recorded flood events since ${h.since}, ${h.last10} of them in the last 10 years. Expect deep water, blocked streets and many calls at once.` : null;
      return [200, { ...h, zone, gauges, summary }];
    }
    if (path === '/api/feedback') {
      const rows = store.get(FEED, []);
      if (method !== 'POST') return [200, rows.slice().reverse()];
      const text = String(isObj(b) && b.text || '').trim().slice(0, 1000);
      if (!text) return [400, { error: "Write what's wrong first." }];
      rows.push({ time: now(), name: String(b.name || '').trim().slice(0, 40) || 'Someone', where: String(b.where || '').slice(0, 200), text });
      store.set(FEED, rows);
      return [200, { ok: true }];
    }
    if (path === '/api/requests/samples' && method === 'POST') { await seed(true); return [200, { ok: true }]; }
    await ensureSeeded();
    if (path === '/api/requests') return [200, readRequests()];
    if (path === '/api/request' && method === 'POST') {
      if (!isObj(b)) return [400, { error: "That request couldn't be read. Try sending it again." }];
      const lat = parseFloat(b.lat), lon = parseFloat(b.lon);
      if (!isFinite(lat) || !isFinite(lon)) return [400, { error: 'We need your location first.' }];
      if (!(lat >= 25.8 && lat <= 36.6 && lon >= -106.7 && lon <= -93.5)) return [400, { error: "This service covers Texas only. If you're in danger, call 911." }];
      const answers = isObj(b.answers) ? b.answers : {};
      const a = Object.fromEntries(['danger', 'rising', 'trapped', 'injured', 'evacuate', 'vulnerable'].map(k => [k, String(answers[k] ?? '')]));
      if (!Object.values(a).every(Boolean)) return [400, { error: "Some questions weren't answered. Go back and answer them." }];
      if (Object.values(a).some(v => !['yes', 'no', 'unsure'].includes(v))) return [400, { error: 'Answers must be yes, no or not sure.' }];
      const people = Math.max(1, Math.min(parseInt(b.people) || 1, 50));
      let r = await makeReport(lat, lon, String(b.label ?? '').slice(0, 160), a, people, String(b.name ?? '').slice(0, 60), String(b.phone ?? '').slice(0, 30));
      r.ts = Date.now() / 1000;
      const [saved, dup] = saveRequest(r, lat, lon);
      if (saved.history_pending) rescoreLater(saved.id, lat, lon);
      const pos = readRequests().filter(x => x.status !== 'Resolved').map(x => x.id).indexOf(saved.id) + 1;
      return [200, { id: saved.id, updated: dup, level: saved.level, position: pos }];
    }
    if ((m = path.match(/^\/api\/requests\/(R\d+)\/reply$/)) && method === 'POST') {
      if (!readRequests().some(x => x.id === m[1])) return [404, { error: "That request wasn't found." }];
      // The app's fixed safety message (what it sends when the AI is down)
      return [200, { reply: `Your request is in, case ${m[1]}. Stay out of moving water and never drive through a flooded road. Get to the highest safe place you can, keep your phone charged, and call 911 if you are in immediate danger.` }];
    }
    if ((m = path.match(/^\/api\/requests\/([^/]+)\/status$/)) && method === 'POST') {
      const status = isObj(b) ? b.status : null;
      if (!['New', 'On the way', 'Resolved'].includes(status)) return [400, { error: 'Status must be New, On the way or Resolved.' }];
      const rows = readRequests();
      if (!rows.some(x => x.id === m[1])) return [404, { error: "That request wasn't found." }];
      for (const r of rows) if (r.id === m[1]) r.status = status;
      writeRequests(rows);
      return [200, { ok: true }];
    }
    if (path === '/api/requests/briefing' && method === 'POST') {
      const rows = readRequests().filter(r => r.status !== 'Resolved');
      if (!rows.length) return [200, { answer: 'No open reports.' }];
      const seen = new Set(), groups = [];
      for (const r of rows) {
        if (seen.has(r.id)) continue;
        const g = [r]; seen.add(r.id);
        for (let i = 0; i < g.length; i++) for (const o of rows) if (!seen.has(o.id) && km(g[i].lat, g[i].lon, o.lat, o.lon) <= 1.5) { seen.add(o.id); g.push(o); }
        if (g.length >= 2) groups.push(g.map(x => x.id).join(', '));
      }
      return [200, { answer: plainBriefing(rows, groups), ai: false }];
    }
    return [404, { error: 'Not found.' }];
  }

  window.fetch = (input, init) => {
    const url = typeof input === 'string' ? input : input && input.url || String(input);
    if (!url.startsWith('/api/')) return realFetch(input, init);
    let body = null;
    if (init && init.body != null) { try { body = JSON.parse(init.body); } catch { body = undefined; } }
    return route(((init && init.method) || 'GET').toUpperCase(), url, body)
      .catch(() => [500, { error: 'Something went wrong. Try again.' }])
      .then(([status, data]) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } }));
  };
  ready.then(ensureSeeded).catch(() => { });
})();
