# Phase 1 — Step 2: Build the Static Model-Selection Webpage Viewer

## Goal

Create the first static webpage for your GitHub Pages site.

This page will let a visitor choose one of the already-generated systems:

- Levi
- SPPCS
- CS

Then it will display:

- selected model,
- model-selection summary table,
- repeated-trial summary table if available,
- kernel/f/trajectory figures.

## Important limitation

This static webpage does **not** run Python.

It only reads files already exported by the model-selection pipeline:

```text
docs/assets/model_selection/
    index.json
    levi/
        manifest.json
        selected_model.json
        summary_table.csv
        repeated_trials.csv
        figures/
    sppcs/
        manifest.json
        selected_model.json
        summary_table.csv
        repeated_trials.csv
        figures/
    cs/
        manifest.json
        selected_model.json
        summary_table.csv
        repeated_trials.csv
        figures/
```

If those exported files are missing, the webpage cannot show results.

---

# File 1: `docs/index.html`

## What this file does

This is the main webpage structure.

It defines:

- page title,
- system dropdown,
- selected-model result card,
- summary table area,
- repeated-trial table area,
- figure gallery area.

## Put this in `docs/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <title>Model Selection Results</title>

  <link rel="stylesheet" href="css/model_selection_style.css" />
</head>

<body>
  <header class="site-header">
    <h1>Model Selection for Interacting Particle Systems</h1>
    <p>
      Static viewer for precomputed model-selection results.
      Choose a system to inspect the selected model, summary metrics, and figures.
    </p>
  </header>

  <main class="container">
    <section class="card">
      <h2>Choose system</h2>

      <label for="system-select">System</label>
      <select id="system-select">
        <option value="">Loading systems...</option>
      </select>

      <button id="load-button">Load Results</button>

      <p id="status-message" class="status-message"></p>
    </section>

    <section class="card" id="selected-model-section">
      <h2>Selected Model</h2>
      <div id="selected-model-card">
        <p>No system loaded yet.</p>
      </div>
    </section>

    <section class="card" id="summary-section">
      <h2>Model-Selection Summary</h2>
      <div id="summary-table-container">
        <p>No summary loaded yet.</p>
      </div>
    </section>

    <section class="card" id="repeated-trials-section">
      <h2>Repeated-Trial Summary</h2>
      <div id="repeated-trials-container">
        <p>No repeated-trial summary loaded yet.</p>
      </div>
    </section>

    <section class="card" id="figures-section">
      <h2>Figures</h2>
      <div id="figure-gallery" class="figure-gallery">
        <p>No figures loaded yet.</p>
      </div>
    </section>
  </main>

  <footer class="site-footer">
    <p>Generated from saved model-selection artifacts.</p>
  </footer>

  <script src="js/model_selection_app.js"></script>
</body>
</html>
```

---

# File 2: `docs/css/model_selection_style.css`

## What this file does

This controls the page appearance.

It makes the page readable by adding:

- cards,
- spacing,
- table styling,
- figure grid,
- responsive layout.

## Put this in `docs/css/model_selection_style.css`

```css
:root {
  --bg: #f6f7fb;
  --card-bg: #ffffff;
  --text: #1f2937;
  --muted: #6b7280;
  --border: #d1d5db;
  --accent: #2563eb;
  --accent-dark: #1d4ed8;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}

.site-header {
  padding: 2rem;
  background: #111827;
  color: white;
}

.site-header h1 {
  margin: 0 0 0.5rem 0;
  font-size: 2rem;
}

.site-header p {
  margin: 0;
  color: #d1d5db;
  max-width: 900px;
}

.container {
  width: min(1200px, 94%);
  margin: 2rem auto;
}

.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.25rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 6px 20px rgba(17, 24, 39, 0.06);
}

.card h2 {
  margin-top: 0;
}

label {
  display: block;
  font-weight: 600;
  margin-bottom: 0.35rem;
}

select,
button {
  font: inherit;
}

select {
  padding: 0.55rem;
  min-width: 260px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: white;
}

button {
  margin-left: 0.5rem;
  padding: 0.6rem 1rem;
  border: 0;
  border-radius: 8px;
  background: var(--accent);
  color: white;
  cursor: pointer;
}

button:hover {
  background: var(--accent-dark);
}

.status-message {
  color: var(--muted);
  margin-bottom: 0;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 0.75rem;
}

.metric {
  padding: 0.85rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: #f9fafb;
}

.metric-label {
  display: block;
  color: var(--muted);
  font-size: 0.9rem;
}

.metric-value {
  display: block;
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 0.2rem;
}

.table-wrapper {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
}

th,
td {
  border: 1px solid var(--border);
  padding: 0.5rem;
  text-align: left;
  vertical-align: top;
}

th {
  background: #f3f4f6;
  font-weight: 700;
}

.figure-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1rem;
}

.figure-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.75rem;
  background: #f9fafb;
}

.figure-card img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 8px;
  background: white;
}

.figure-card figcaption {
  margin-top: 0.5rem;
  color: var(--muted);
  font-size: 0.9rem;
  word-break: break-word;
}

.site-footer {
  padding: 1.25rem 2rem;
  color: var(--muted);
  text-align: center;
}
```

---

# File 3: `docs/js/model_selection_app.js`

## What this file does

This is the page logic.

It:

1. loads `assets/model_selection/index.json`,
2. fills the dropdown with available systems,
3. loads the selected system's `manifest.json`,
4. displays selected model information,
5. displays summary table records,
6. displays repeated-trial table records,
7. displays all figure images listed in the manifest.

## Put this in `docs/js/model_selection_app.js`

```javascript
const INDEX_PATH = "assets/model_selection/index.json";

const systemSelect = document.getElementById("system-select");
const loadButton = document.getElementById("load-button");
const statusMessage = document.getElementById("status-message");

const selectedModelCard = document.getElementById("selected-model-card");
const summaryTableContainer = document.getElementById("summary-table-container");
const repeatedTrialsContainer = document.getElementById("repeated-trials-container");
const figureGallery = document.getElementById("figure-gallery");

let systemsIndex = [];

async function fetchJson(path) {
  const response = await fetch(path);

  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }

  return await response.json();
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "—";
    }

    const abs = Math.abs(value);
    if ((abs > 0 && abs < 1e-3) || abs >= 1e4) {
      return value.toExponential(3);
    }

    return value.toFixed(4);
  }

  return String(value);
}

function setStatus(message) {
  statusMessage.textContent = message;
}

function makeTable(records, maxRows = null) {
  if (!records || records.length === 0) {
    return "<p>No table records available.</p>";
  }

  const rows = maxRows ? records.slice(0, maxRows) : records;
  const columns = Object.keys(rows[0]);

  const header = columns
    .map((col) => `<th>${col}</th>`)
    .join("");

  const body = rows
    .map((row) => {
      const cells = columns
        .map((col) => `<td>${formatValue(row[col])}</td>`)
        .join("");

      return `<tr>${cells}</tr>`;
    })
    .join("");

  return `
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>${header}</tr>
        </thead>
        <tbody>
          ${body}
        </tbody>
      </table>
    </div>
  `;
}

function renderSelectedModel(manifest) {
  const selected = manifest.selected_model || {};

  selectedModelCard.innerHTML = `
    <div class="metric-grid">
      <div class="metric">
        <span class="metric-label">System</span>
        <span class="metric-value">${formatValue(manifest.system_name)}</span>
      </div>

      <div class="metric">
        <span class="metric-label">Selected candidate</span>
        <span class="metric-value">${formatValue(selected.candidate)}</span>
      </div>

      <div class="metric">
        <span class="metric-label">Primary metric</span>
        <span class="metric-value">${formatValue(selected.primary_metric)}</span>
      </div>

      <div class="metric">
        <span class="metric-label">Primary value</span>
        <span class="metric-value">${formatValue(selected.primary_value)}</span>
      </div>

      <div class="metric">
        <span class="metric-label">Residual error</span>
        <span class="metric-value">${formatValue(selected.E_res)}</span>
      </div>

      <div class="metric">
        <span class="metric-label">Condition number</span>
        <span class="metric-value">${formatValue(selected.condition_number)}</span>
      </div>

      <div class="metric">
        <span class="metric-label">Energy kernel norm</span>
        <span class="metric-value">${formatValue(selected.phi_E_norm)}</span>
      </div>

      <div class="metric">
        <span class="metric-label">Alignment kernel norm</span>
        <span class="metric-value">${formatValue(selected.phi_A_norm)}</span>
      </div>

      <div class="metric">
        <span class="metric-label">Environment force norm</span>
        <span class="metric-value">${formatValue(selected.f_norm)}</span>
      </div>
    </div>
  `;
}

function renderTables(manifest) {
  summaryTableContainer.innerHTML = makeTable(manifest.summary_records || []);

  if (manifest.repeated_trial_records && manifest.repeated_trial_records.length > 0) {
    repeatedTrialsContainer.innerHTML = makeTable(manifest.repeated_trial_records);
  } else {
    repeatedTrialsContainer.innerHTML = "<p>No repeated-trial records available for this system.</p>";
  }
}

function readableFigureName(path) {
  return path
    .replace(/^figures\//, "")
    .replace(/\.[^.]+$/, "")
    .replaceAll("_", " / ");
}

function renderFigures(manifest, systemName) {
  const figures = manifest.figures || [];

  if (figures.length === 0) {
    figureGallery.innerHTML = "<p>No figures available.</p>";
    return;
  }

  figureGallery.innerHTML = figures
    .filter((path) => {
      const lower = path.toLowerCase();
      return lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".svg");
    })
    .map((path) => {
      const src = `assets/model_selection/${systemName}/${path}`;
      const caption = readableFigureName(path);

      return `
        <figure class="figure-card">
          <img src="${src}" alt="${caption}" loading="lazy" />
          <figcaption>${caption}</figcaption>
        </figure>
      `;
    })
    .join("");
}

async function loadSystem(systemName) {
  if (!systemName) {
    setStatus("Choose a system first.");
    return;
  }

  setStatus(`Loading ${systemName}...`);

  const manifestPath = `assets/model_selection/${systemName}/manifest.json`;
  const manifest = await fetchJson(manifestPath);

  renderSelectedModel(manifest);
  renderTables(manifest);
  renderFigures(manifest, systemName);

  setStatus(`Loaded ${systemName}.`);
}

async function initializePage() {
  try {
    setStatus("Loading available systems...");

    const index = await fetchJson(INDEX_PATH);
    systemsIndex = index.systems || [];

    if (systemsIndex.length === 0) {
      systemSelect.innerHTML = `<option value="">No systems found</option>`;
      setStatus("No systems found in assets/model_selection/index.json.");
      return;
    }

    systemSelect.innerHTML = systemsIndex
      .map((item) => {
        return `<option value="${item.system_name}">${item.system_name}</option>`;
      })
      .join("");

    setStatus("Choose a system and click Load Results.");

    await loadSystem(systemsIndex[0].system_name);
  } catch (error) {
    console.error(error);
    setStatus(error.message);
  }
}

loadButton.addEventListener("click", async () => {
  try {
    await loadSystem(systemSelect.value);
  } catch (error) {
    console.error(error);
    setStatus(error.message);
  }
});

initializePage();
```

---

# How to test locally

## Action

From your project root, run:

```bash
cd docs
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Why

The browser blocks some local file loading when you open `index.html` directly. Running a simple local server makes `fetch()` work correctly.

---

# Expected result

The page should:

1. load `assets/model_selection/index.json`,
2. populate the system dropdown,
3. load the first system automatically,
4. show selected model data,
5. show summary table,
6. show repeated-trial table if present,
7. show saved figures.

---

# If it fails

## Error: `Failed to load assets/model_selection/index.json`

Reason:

The exported website assets are missing.

Fix:

Run your static exporter first:

```python
from ips_learning.model_selection import StaticModelSelectionWebsiteExporter

exporter = StaticModelSelectionWebsiteExporter(
    artifact_root=ARTIFACTS,
    output_root=PROJECT_ROOT / "docs" / "assets" / "model_selection",
)

exporter.export_all([...])
```

## Error: figures do not appear

Reason:

The manifest may list figures, but the figure files may not have been copied.

Check:

```text
docs/assets/model_selection/{system}/figures/
```

## Error: page is blank

Open browser developer tools:

```text
Right click page → Inspect → Console
```

Look for JavaScript errors.
