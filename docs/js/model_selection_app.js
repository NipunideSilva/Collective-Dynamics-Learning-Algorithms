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

  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  if (typeof value === "string") {

  const candidateMap = {
    first_order_energy_only:
      "1st Order: Energy Only",

    first_order_energy_f:
      "1st Order: Energy + f",

    second_order_energy_only:
      "2nd Order: Energy Only",

    second_order_alignment_only:
      "2nd Order: Alignment Only",

    second_order_energy_f:
      "2nd Order: Energy + f",

    second_order_alignment_f:
      "2nd Order: Alignment + f",

    second_order_energy_alignment_only:
      "2nd Order: Energy + Alignment",

    second_order_energy_alignment_f:
      "2nd Order: Energy + Alignment + f",
  };

    if (candidateMap[value]) {
      return candidateMap[value];
    }
  }

  return String(value);
}

function setStatus(message) {
  statusMessage.textContent = message;
}

const COLUMN_LABELS = {
  // Core candidate columns
  candidate: "Candidate Model",
  selected: "Selected",
  rank: "Rank",

  system_order: "System Order",
  interaction_type: "Interaction Type",
  algorithm: "Algorithm",
  include_environment: "Environment Force",
  complexity: "Complexity",

  // Selection metric
  primary_metric: "Selection Metric",
  primary_value:
    "Validation Trajectory Error \\(\\bar E_{\\mathrm{traj}}(M_{\\mathrm{val}})\\)",

  // Train-fit / train diagnostics
  E_traj_reconstruction: "Reconstruction Error \\(E_{\\mathrm{recon}}\\)",
  E_traj_reconstruction_std: "Reconstruction Std",
  E_traj_reconstruction_rel:
    "Relative Reconstruction Error \\(E_{\\mathrm{recon}}^{\\mathrm{rel}}\\)",

  E_traj_prediction: "Train Prediction Error \\(E_{\\mathrm{pred}}\\)",
  E_traj_prediction_std: "Train Prediction Std",

  // Validation diagnostic
  E_traj_full: "Validation Full Error \\(E_{\\mathrm{full}}\\)",
  E_traj_full_std: "Validation Full Std",

  // Residual / framework diagnostics
  E_res: "Residual Error \\(E_{\\mathrm{res}}\\)",
  E_res_rel:
    "Relative Residual Error \\(E_{\\mathrm{res},\\epsilon}^{\\mathrm{rel}}\\)",

  trajectory_reconstruction_rss:
    "Reconstruction RSS \\(E_{\\mathrm{recon}}^{\\mathrm{RSS}}\\)",
  n_likelihood_parameters: "Parameters \\(k_c\\)",
  AIC: "AIC",
  BIC: "BIC",

  // condition_number: "Condition Number \\(\\kappa(A)\\)",

  // Activity diagnostics
  phi_E_norm: "Energy Activity \\(\\|\\tilde{\\phi}^{E}\\|_{L^2(\\rho_R)}\\)",
  phi_A_norm: "Alignment Activity \\(\\|\\tilde{\\phi}^{A}\\|_{L^2(\\rho_R)}\\)",
  f_norm: "Environment Activity \\(\\|\\tilde f\\|_{L^2(\\rho_Z)}\\)",

  // Candidate status
  accepted: "Accepted",
  rejection_reason: "Rejection Reason",

  // Repeated-trial section names
  selection_frequency: "Candidate Selection Frequency",
  order_frequency: "Order Selection Frequency",
  interaction_frequency: "Interaction Selection Frequency",
  environment_frequency: "Environment Selection Frequency",
  selected_metric_summary: "Selected Model Metrics",
  candidate_metric_summary: "Candidate Metric Summary",
  failure_summary: "Failure Summary",

  // Repeated-trial frequency columns
  selected_candidate: "Selected Candidate",
  selected_system_order: "Selected System Order",
  selected_interaction_type: "Selected Interaction Type",
  selected_include_environment: "Selected Environment Force",
  selected_count: "Times Selected",
  selected_fraction: "Selection Frequency",
  num_valid_trials: "Valid Trials",

  // Generic summary columns
  metric: "Metric",
  mean: "Mean \\(\\mu\\)",
  trial_std: "Trial Std \\(\\sigma\\)",
  mean_pm_trial_std: "Mean ± Std \\(\\mu \\pm \\sigma\\)",
  num_trials: "Trials \\(n\\)",

  // Repeated-trial selected metric names
  selected_E_traj_full: "Selected Validation Full Error",
  selected_E_traj_full_std: "Selected Validation Full Std",
  selected_E_traj_reconstruction: "Selected Reconstruction Error",
  selected_E_traj_reconstruction_rel: "Selected Relative Reconstruction Error",
  selected_E_traj_prediction: "Selected Train Prediction Error",
  selected_E_res: "Selected Residual Error",
  selected_E_res_rel: "Selected Relative Residual Error",
  selected_condition_number: "Selected Condition Number",
  selected_complexity: "Selected Complexity",
  selected_phi_E_norm: "Selected Energy Activity",
  selected_phi_A_norm: "Selected Alignment Activity",
  selected_f_norm: "Selected Environment Activity",

  // Candidate metric repeated-trial means
  primary_value_mean: "Mean Validation Error \\(\\mu(E_{\\mathrm{full}})\\)",
  primary_value_trial_std: "Trial Std Validation Error \\(\\sigma(E_{\\mathrm{full}})\\)",
  primary_value_mean_pm_trial_std: "Validation Error \\(\\mu \\pm \\sigma\\)",

  E_traj_full_mean: "Mean Validation Full Error",
  E_traj_full_trial_std: "Trial Std Validation Full Error",
  E_traj_full_mean_pm_trial_std: "Validation Full Error \\(\\mu \\pm \\sigma\\)",

  E_traj_reconstruction_mean: "Mean Reconstruction Error",
  E_traj_reconstruction_trial_std: "Trial Std Reconstruction Error",
  E_traj_reconstruction_mean_pm_trial_std: "Reconstruction Error \\(\\mu \\pm \\sigma\\)",

  E_traj_reconstruction_rel_mean: "Mean Relative Reconstruction Error",
  E_traj_reconstruction_rel_trial_std: "Trial Std Relative Reconstruction Error",
  E_traj_reconstruction_rel_mean_pm_trial_std:
    "Relative Reconstruction Error \\(\\mu \\pm \\sigma\\)",

  E_traj_prediction_mean: "Mean Train Prediction Error",
  E_traj_prediction_trial_std: "Trial Std Train Prediction Error",
  E_traj_prediction_mean_pm_trial_std: "Train Prediction Error \\(\\mu \\pm \\sigma\\)",

  E_res_mean: "Mean Residual Error \\(\\mu(E_{\\mathrm{res}})\\)",
  E_res_trial_std: "Trial Std Residual Error \\(\\sigma(E_{\\mathrm{res}})\\)",
  E_res_mean_pm_trial_std: "Residual Error \\(\\mu \\pm \\sigma\\)",

  E_res_rel_mean: "Mean Relative Residual Error",
  E_res_rel_trial_std: "Trial Std Relative Residual Error",
  E_res_rel_mean_pm_trial_std: "Relative Residual Error \\(\\mu \\pm \\sigma\\)",

  condition_number_mean: "Mean Condition Number",
  condition_number_trial_std: "Trial Std Condition Number",
  condition_number_mean_pm_trial_std: "Condition Number \\(\\mu \\pm \\sigma\\)",

  complexity_mean: "Mean Complexity",
  complexity_trial_std: "Trial Std Complexity",
  complexity_mean_pm_trial_std: "Complexity \\(\\mu \\pm \\sigma\\)",

  phi_E_norm_mean: "Mean Energy Activity",
  phi_E_norm_trial_std: "Trial Std Energy Activity",
  phi_E_norm_mean_pm_trial_std: "Energy Activity \\(\\mu \\pm \\sigma\\)",

  phi_A_norm_mean: "Mean Alignment Activity",
  phi_A_norm_trial_std: "Trial Std Alignment Activity",
  phi_A_norm_mean_pm_trial_std: "Alignment Activity \\(\\mu \\pm \\sigma\\)",

  f_norm_mean: "Mean Environment Activity",
  f_norm_trial_std: "Trial Std Environment Activity",
  f_norm_mean_pm_trial_std: "Environment Activity \\(\\mu \\pm \\sigma\\)",
};

function makeTable(records, maxRows = null, selectedColumns = null) {
  if (!records || records.length === 0) {
    return "<p>No table records available.</p>";
  }

  const rows = maxRows ? records.slice(0, maxRows) : records;
  const availableColumns = Object.keys(rows[0]);

  const columns = selectedColumns
    ? selectedColumns.filter((col) => availableColumns.includes(col))
    : availableColumns;

  const header = columns
    .map((col) => {
      const label = COLUMN_LABELS[col] || col;
      return `<th>${label}</th>`;
    })
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

function modelSelectionDefinitions() {
  return `
    <div class="definition-box">
      <h3>Definitions</h3>

      <p>
      <strong>Fit-window reconstruction:</strong>
      \\[
      E_{\\mathrm{recon}}
      =
      \\frac{1}{M}
      \\sum_{m=1}^{M}
      \\max_{t_\\ell\\in[t_0,T]}
      e_m(t_\\ell).
      \\]
      </p>

      <p>
      <strong>Train prediction:</strong>
      \\[
      E_{\\mathrm{pred}}
      =
      \\frac{1}{M}
      \\sum_{m=1}^{M}
      \\max_{t_\\ell\\in[T,T_f]}
      e_m(t_\\ell).
      \\]
      </p>

      <p>
      <strong>Validation full-horizon error:</strong>
      \\[
      E_{\\mathrm{full}}
      =
      \\frac{1}{M_{\\mathrm{val}}}
      \\sum_m
      \\max_{t_\\ell\\in[t_0,T_f]}
      e_m(t_\\ell).
      \\]
      </p>

      <p>
      where
      \\[
      e_m(t_\\ell)
      =
      \\left(
      \\frac1N
      \\sum_i
      \\|x_i^m(t_\\ell)
      -
      \\hat x_i^m(t_\\ell)
      \\|^2
      \\right)^{1/2}.
      \\]
      </p>

      <p>
        <strong>Residual error:</strong>
        \\[
        E_{\\mathrm{res}}
        =
        \\left(
        \\frac{1}{MLN}
        \\sum_{m,\\ell,i}
        \\|\\mathrm{LHS}_i^{m,\\ell}
        -
        \\widehat{\\mathrm{RHS}}_i^{m,\\ell}\\|^2
        \\right)^{1/2}.
        \\]
      </p>

      <p>
        <strong>Kernel activity:</strong>
        \\[
        \\|\\tilde{\\phi}^{E}\\|_{L^2(\\rho_R)}
        =
        \\left(
        \\int |\\tilde{\\phi}^{E}(r)|^2\\rho_R(r)\\,dr
        \\right)^{1/2},
        \\qquad
        \\|\\tilde{\\phi}^{A}\\|_{L^2(\\rho_R)}
        =
        \\left(
        \\int |\\tilde{\\phi}^{A}(r)|^2\\rho_R(r)\\,dr
        \\right)^{1/2}.
        \\]
      </p>

      <p>
        <strong>Environment-force activity:</strong>
        \\[
        \\|\\tilde f\\|_{L^2(\\rho_Z)}
        =
        \\left(
        \\int \\|\\tilde f(z)\\|^2\\rho_Z(z)\\,dz
        \\right)^{1/2}.
        \\]
        For second-order systems, \\(z=v\\). For first-order systems, \\(z=x\\).
      </p>

      <p>
        <strong>Condition number:</strong>
        \\[
        \\kappa(A)=\\frac{\\sigma_{\\max}(A)}{\\sigma_{\\min}(A)}.
        \\]
        Large values indicate a more ill-conditioned normal equation.
      </p>
    </div>
  `;
}

function repeatedTrialDefinitions() {
  return `
    <div class="definition-box">
      <h3>Repeated-trial definitions</h3>

      <p>
        Repeated-trial summaries report statistics across independently generated
        training trials.
      </p>

      <p>
        <strong>Mean:</strong>
        \\[
        \\mu = \\frac{1}{Q}\\sum_{q=1}^{Q} a_q.
        \\]
      </p>

      <p>
        <strong>Trial standard deviation:</strong>
        \\[
        \\sigma
        =
        \\left(
        \\frac{1}{Q-1}
        \\sum_{q=1}^{Q}(a_q-\\mu)^2
        \\right)^{1/2}.
        \\]
      </p>

      <p>
        <strong>Selection frequency:</strong>
        \\[
        \\mathrm{freq}(c)
        =
        \\frac{\\#\\{q:\\ c \\text{ selected in trial } q\\}}{Q}.
        \\]
      </p>
    </div>
  `;
}

function renderSelectedModel(manifest) {
  const selected = manifest.selected_model || {};


  const modelEquation = `
  <div class="equation-block">
    <p><strong>Candidate model families:</strong></p>

    <p>First-order candidate:</p>
    \\[
    \\dot{x}_i =
    \\tilde{f}(x_i)
    + \\frac{1}{N}\\sum_j \\tilde{\\phi}^{E}(r_{ij})(x_j-x_i).
    \\]

    <p>Second-order full candidate:</p>
    \\[
    \\ddot{x}_i =
    \\tilde{f}(v_i)
    + \\frac{1}{N}\\sum_j \\tilde{\\phi}^{E}(r_{ij})(x_j-x_i)
    + \\frac{1}{N}\\sum_j \\tilde{\\phi}^{A}(r_{ij})(v_j-v_i).
    \\]
  </div>
`;
  selectedModelCard.innerHTML = `
    ${modelEquation}

    <div class="metric-grid">

      <div class="metric">
        <span class="metric-label">System</span>
        <span class="metric-value">
          ${formatValue(manifest.system_name)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">Selected Candidate</span>
        <span class="metric-value">
          ${formatValue(selected.candidate)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">System Order</span>
        <span class="metric-value">
          ${formatValue(selected.system_order)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">Interaction Type</span>
        <span class="metric-value">
          ${formatValue(selected.interaction_type)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">Environment Force</span>
        <span class="metric-value">
          ${formatValue(selected.include_environment)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">Complexity Profile</span>
        <span class="metric-value">
          ${formatValue(selected.complexity)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">Primary Metric</span>
        <span class="metric-value">
          ${formatValue(selected.primary_metric)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">
          Validation Error \\(E_{\\mathrm{full}}\\)
        </span>
        <span class="metric-value">
          ${formatValue(selected.primary_value)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">
          Residual Error \\(E_{\\mathrm{res}}\\)
        </span>
        <span class="metric-value">
          ${formatValue(selected.E_res)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">
          Relative Residual Error
          \\(E_{\\mathrm{res}}^{\\mathrm{rel}}\\)
        </span>
        <span class="metric-value">
          ${formatValue(selected.E_res_rel)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">
          Energy Activity
        </span>
        <span class="metric-value">
          ${formatValue(selected.phi_E_norm)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">
          Alignment Activity
        </span>
        <span class="metric-value">
          ${formatValue(selected.phi_A_norm)}
        </span>
      </div>

      <div class="metric">
        <span class="metric-label">
          Environment Activity
        </span>
        <span class="metric-value">
          ${formatValue(selected.f_norm)}
        </span>
      </div>

    </div>
  `;
}

const THESIS_SELECTION_COLUMNS = [
  "candidate",
  "system_order",
  "E_res_rel",
  "E_traj_reconstruction_rel",
  "primary_value",
  "complexity",
  "E_res",
  "rank",
  "selected",
];

const THESIS_AIC_BIC_COLUMNS = [
  "candidate",
  "system_order",
  "trajectory_reconstruction_rss",
  "n_likelihood_parameters",
  "AIC",
  "BIC",
];

function renderTables(manifest) {
  const summaryRecords = manifest.summary_records || [];

  summaryTableContainer.innerHTML = `
    <h3>Proposed Model Selection</h3>
    ${makeTable(
      summaryRecords,
      null,
      THESIS_SELECTION_COLUMNS
    )}

    <h3>AIC/BIC Comparison</h3>
    ${makeTable(
      summaryRecords,
      null,
      THESIS_AIC_BIC_COLUMNS
    )}

    ${modelSelectionDefinitions()}
  `;

  const repeated = manifest.repeated_trial_records || {};
  const sections = [];

  for (const [name, records] of Object.entries(repeated)) {
    if (!records || records.length === 0) {
      continue;
    }

    sections.push(`
      <h3>${COLUMN_LABELS[name] || name}</h3>
      ${makeTable(records)}
    `);
  }

  if (sections.length > 0) {
    repeatedTrialsContainer.innerHTML =
      sections.join("") + repeatedTrialDefinitions();
  } else {
    repeatedTrialsContainer.innerHTML =
      "<p>No repeated-trial summary available.</p>";
  }
}

function readableFigureName(path) {
  const figureLabels = {
    "figures/kernels/full_model_energy_kernel.png":
      "Energy kernel: true vs learned \\(\\phi^E\\)",

    "figures/kernels/full_model_alignment_kernel.png":
      "Alignment kernel: true vs learned \\(\\phi^A\\)",

    "figures/scores/kernel_norms.png":
      "Candidate activity norms \\(\\|\\tilde{\\phi}^{E}\\|, \\|\\tilde{\\phi}^{A}\\|\\)",

    "figures/scores/selection_summary.png":
      "Candidate validation-score comparison",

    "figures/environment/second_order_alignment_f.png":
      "Learned environment force \\(\\tilde f(v)\\): alignment model",

    "figures/environment/second_order_energy_alignment_f.png":
      "Learned environment force \\(\\tilde f(v)\\): energy + alignment model",

    "figures/rollouts/second_order_alignment_validation_rollout.png":
      "Validation rollout: selected alignment model",

    "figures/rollouts/second_order_energy_alignment_validation_rollout.png":
      "Validation rollout: selected energy + alignment model",
  };

  if (figureLabels[path]) {
    return figureLabels[path];
  }

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

  if (
    window.MathJax &&
    typeof MathJax.typesetPromise === "function"
  ) {
    MathJax.typesetPromise();
  }

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