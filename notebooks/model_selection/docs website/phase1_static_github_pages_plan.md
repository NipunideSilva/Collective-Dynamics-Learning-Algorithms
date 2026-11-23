# Phase 1 — Static GitHub Pages Model-Selection Results Viewer

## Goal

Build a static webpage that displays already-generated model-selection results for:

- Levi
- SPPCS
- CS

This phase does **not** run Python from the webpage. It only displays exported artifacts that were already produced by the model-selection notebooks.

## Why Phase 1 is static

GitHub Pages can serve HTML, CSS, JavaScript, images, CSV files, and JSON files. It cannot execute the Python model-selection pipeline.

Therefore, Phase 1 uses this workflow:

```text
Run notebooks locally
    ↓
Export website-ready artifacts
    ↓
Commit docs/assets/model_selection/... to GitHub
    ↓
GitHub Pages displays the exported results
```

Uploaded-dataset execution requires a backend server and belongs to Phase 2.

## User workflow on the webpage

The webpage should show:

```text
Dataset dropdown:
    Levi
    SPPCS
    CS

Submit button:
    Load selected precomputed result

Output area:
    selected model card
    model-selection summary table
    energy kernel plot
    alignment kernel plot
    f plot, when available
    rollout / phase portrait plot
    repeated-trial summary table
```

## Required exported files

Each system should export a website-ready directory:

```text
docs/assets/model_selection/{system}/
    manifest.json
    selected_model.json
    model_selection_summary.csv
    figures/
        kernels/
            full_model_energy_kernel.png
            full_model_alignment_kernel.png
        environment/
            selected_f.png
        rollouts/
            selected_validation_rollout.png
        scores/
            selection_summary.png
            kernel_norms.png
    repeated_trials/
        repeated_trial_model_selection_metrics.csv
        selection_frequency.csv
        selected_metric_summary.csv
        candidate_metric_summary.csv
```

If a figure does not exist for a system, the webpage should skip it instead of failing.

## Main webpage files

Create:

```text
docs/model-selection.html
docs/assets/model_selection/model-selection-app.js
docs/assets/model_selection/model-selection-style.css
```

The HTML file provides the page structure.
The JavaScript file loads `manifest.json`, CSV tables, and figures.
The CSS file styles the dropdown, cards, tables, and figures.

## Static data loading design

The JavaScript should map system names to manifest paths:

```javascript
const SYSTEMS = {
  levi: "assets/model_selection/levi/manifest.json",
  sppcs: "assets/model_selection/sppcs/manifest.json",
  cs: "assets/model_selection/cs/manifest.json",
};
```

When the user selects a system and clicks Submit:

```text
1. Load manifest.json
2. Load selected_model.json
3. Load model_selection_summary.csv
4. Render selected model card
5. Render candidate comparison table
6. Render figures listed in manifest.json
7. Load repeated-trial tables if available
```

## What happens before the webpage can work

Before building the page, the notebooks must export consistent files.

For each system notebook:

```text
01_sppcs_model_selection.ipynb
02_levi_model_selection.ipynb
03_cs_model_selection.ipynb
```

Run:

```python
website_exporter.export_experiment(...)
```

Then run:

```text
04_repeated_trials_model_selection_summary.ipynb
```

to export repeated-trial summary tables.

## Phase 1 limitations

Phase 1 supports only precomputed datasets.

It does not support:

- uploading a new dataset,
- running model selection from the browser,
- generating new fits or rollouts online,
- long-running Python jobs.

Those require Phase 2 with a FastAPI backend.

## Phase 1 completion criteria

Phase 1 is complete when:

1. `docs/model-selection.html` opens locally.
2. The dropdown lists Levi, SPPCS, and CS.
3. Clicking Submit loads the selected system result.
4. The selected model appears.
5. The candidate summary table appears.
6. Kernel/f/trajectory figures appear when available.
7. Repeated-trial summary tables appear when available.
8. The same page works after pushing to GitHub Pages.

## Building Steps

Step 1 is to standardize the exported website artifact structure for the three systems.

# Phase 1 — Step 1: Standardize Website Artifact Structure

## Step goal

Create one consistent website-ready artifact structure for all precomputed model-selection systems:

- Levi
- SPPCS
- CS

The static GitHub Pages frontend should not need system-specific logic. It should load the same file names and folder structure for every system.

---

## Why this step is needed

The model-selection pipeline produces many research artifacts under paths such as:

```text
artifacts/model_selection/{system}/{profile}/{dataset}/...
artifacts/{system}/figures/...
artifacts/{system}/density_evaluations/...
```

Those paths are useful for research, but they are not ideal for a webpage.

The webpage needs a clean, small, stable export directory:

```text
docs/assets/model_selection/{system}/
```

Each system should contain the same files:

```text
manifest.json
selected_model.json
model_selection_summary.csv
figures/
repeated_trials/
```

Then the website can display results by reading only `manifest.json`.

---

## Standard website export layout

Use this structure:

```text
docs/assets/model_selection/
    levi/
        manifest.json
        selected_model.json
        model_selection_summary.csv
        figures/
            kernels/
                full_model_energy_kernel.png
                full_model_alignment_kernel.png
            environment/
                selected_f.png
            rollouts/
                selected_validation_rollout.png
            scores/
                selection_summary.png
                kernel_norms.png
        repeated_trials/
            repeated_trial_model_selection_metrics.csv
            selection_frequency.csv
            selected_metric_summary.csv
            candidate_metric_summary.csv

    sppcs/
        manifest.json
        selected_model.json
        model_selection_summary.csv
        figures/
            kernels/
            environment/
            rollouts/
            scores/
        repeated_trials/

    cs/
        manifest.json
        selected_model.json
        model_selection_summary.csv
        figures/
            kernels/
            environment/
            rollouts/
            scores/
        repeated_trials/
```

---

## Important rule

The website export should contain only lightweight display artifacts.

Include:

```text
.csv
.json
.png
.svg
.pdf, optional
```

Do not include:

```text
fit_result.pkl
normal_system.npz
coefficients.npz
rollout.npz
large datasets
```

The webpage should display outputs, not reproduce computations.

---

## Required exported files per system

Each system export must include:

### 1. `manifest.json`

This is the main file the webpage reads.

It should contain:

```json
{
  "system_name": "levi",
  "display_name": "Levi / Phototaxis",
  "profile_name": "levi_velocity_grid",
  "dataset_name": "dataset_train_fit_trial_001",
  "selected_model_path": "selected_model.json",
  "summary_table_path": "model_selection_summary.csv",
  "figures": {
    "energy_kernel": "figures/kernels/full_model_energy_kernel.png",
    "alignment_kernel": "figures/kernels/full_model_alignment_kernel.png",
    "environment": "figures/environment/selected_f.png",
    "rollout": "figures/rollouts/selected_validation_rollout.png",
    "selection_summary": "figures/scores/selection_summary.png",
    "kernel_norms": "figures/scores/kernel_norms.png"
  },
  "repeated_trials": {
    "metrics": "repeated_trials/repeated_trial_model_selection_metrics.csv",
    "selection_frequency": "repeated_trials/selection_frequency.csv",
    "selected_metric_summary": "repeated_trials/selected_metric_summary.csv",
    "candidate_metric_summary": "repeated_trials/candidate_metric_summary.csv"
  }
}
```

### 2. `selected_model.json`

This contains the final selected candidate:

```json
{
  "candidate": "second_order_alignment",
  "interaction_type": "alignment",
  "primary_metric": "E_traj_prediction",
  "primary_value": 0.0123,
  "selection_status": "selected"
}
```

### 3. `model_selection_summary.csv`

This is the candidate comparison table. Required columns:

```text
rank
selected
candidate
interaction_type
primary_value
E_res
condition_number
phi_E_norm
phi_A_norm
f_norm
theta_phi_E_norm
theta_phi_A_norm
theta_f_norm
accepted
rejection_reason
```

### 4. Figures

Minimum figures:

```text
figures/kernels/full_model_energy_kernel.png
figures/kernels/full_model_alignment_kernel.png
figures/environment/selected_f.png
figures/rollouts/selected_validation_rollout.png
figures/scores/selection_summary.png
figures/scores/kernel_norms.png
```

For CS, `selected_f.png` can still be exported as a near-zero/noise diagnostic plot.

### 5. Repeated-trial tables

Minimum repeated-trial tables:

```text
repeated_trials/repeated_trial_model_selection_metrics.csv
repeated_trials/selection_frequency.csv
repeated_trials/selected_metric_summary.csv
repeated_trials/candidate_metric_summary.csv
```

---

## Export logic

The export function should:

1. create `docs/assets/model_selection/{system}/`,
2. copy selected model JSON,
3. copy summary CSV,
4. copy figures,
5. copy repeated-trial tables,
6. write a normalized `manifest.json`.

This export step should run after notebooks finish producing artifacts.

---

## Suggested module update

Extend:

```text
src/ips_learning/model_selection/website_export.py
```

Add a function like:

```python
export_static_system_result(
    system_name="levi",
    display_name="Levi / Phototaxis",
    profile_name="levi_velocity_grid",
    dataset_name="dataset_train_fit_trial_001",
    source_root=ARTIFACTS / "model_selection" / "levi" / "levi_velocity_grid" / "dataset_train_fit_trial_001",
    repeated_trial_root=ARTIFACTS / "model_selection" / "levi" / "levi_velocity_grid" / "repeated_trials" / "summaries",
    output_root=PROJECT_ROOT / "docs" / "assets" / "model_selection",
)
```

---

## Validation

After export, validate:

```python
for system in ["levi", "sppcs", "cs"]:
    root = PROJECT_ROOT / "docs" / "assets" / "model_selection" / system

    assert (root / "manifest.json").exists()
    assert (root / "selected_model.json").exists()
    assert (root / "model_selection_summary.csv").exists()
    assert (root / "figures").exists()
    assert (root / "repeated_trials").exists()
```

Also open `manifest.json` and confirm every referenced path exists.

---

After this step, the project has a stable website artifact contract.

The next step is:

```text
Phase 1 — Step 2: Implement the static exporter that creates this structure.
```

# Phase 1 — Step 2: Build the static webpage viewer

## Step goal

Build a static GitHub Pages page that reads:

```text
docs/assets/model_selection/index.json
```

and displays the selected system results for:

```text
Levi
SPPCS
CS
```

This phase does **not** run Python and does **not** support uploaded datasets yet. It only displays precomputed exported artifacts.

---

## What the webpage does

The page should:

1. load the system list from `docs/assets/model_selection/index.json`,
2. populate a dropdown with available systems,
3. load the selected system's `manifest.json`,
4. display:
   - selected model,
   - model-selection summary table,
   - repeated-trial summary table if available,
   - kernel/f/trajectory figures,
5. update the page when the user changes the dropdown.

---

## Expected website artifact structure

The static page assumes this layout:

```text
docs/
  index.html
  assets/
    model_selection/
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
  js/
    model_selection_app.js
  css/
    model_selection_style.css
```

---

## Files to add

Add these files:

```text
docs/index.html
docs/js/model_selection_app.js
docs/css/model_selection_style.css
```

If you already have a GitHub Pages homepage, use:

```text
docs/model_selection.html
```

instead of replacing `docs/index.html`.

---

## Why this step is needed

The Python pipeline already exports website-ready artifacts. The static webpage is only a viewer. Keeping this separate gives the clean workflow:

```text
Python pipeline
  → exports JSON/CSV/figures
  → GitHub Pages loads and displays them
```

This avoids trying to run Python on GitHub Pages, which is not possible.

---

## Implementation summary

### `index.html`

Defines the page layout:

- dataset/system dropdown,
- selected-model card,
- summary tables,
- figure gallery.

### `model_selection_app.js`

Loads:

```text
assets/model_selection/index.json
assets/model_selection/{system}/manifest.json
```

Then renders data dynamically.

### `model_selection_style.css`

Adds readable table, card, and figure-gallery styling.

---

## Validation

After adding files, run a local static server from `docs/`:

```bash
cd docs
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Expected result:

- system dropdown appears,
- selected model card loads,
- model-selection table appears,
- figures display,
- switching systems updates the page.

---

## Stop condition

Stop after confirming the static webpage loads precomputed results for one system.

The next step is:

```text
Phase 1 — Step 3: Improve webpage layout and figure grouping
```


