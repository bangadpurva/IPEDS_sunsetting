const interestOptions = [
  ["technology", "Technology"],
  ["healthcare", "Healthcare"],
  ["business", "Business"],
  ["education", "Education"],
  ["public-service", "Public service"],
  ["creative", "Creative"],
  ["science", "Science"],
  ["trades", "Trades"],
];

let dataset = { summary: {}, programs: [] };
let dimensions = { summary: {}, institution_trends: [], demographics: { gender: [] } };
let activeInterests = new Set(["technology", "healthcare"]);
let backendAvailable = true;
let runPollTimer = null;
let agentRecommendations = null;
let serverMode = location.protocol !== "file:";
let coachSessionId = null;

const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

const searchAliases = {
  "data science": ["computer", "information sciences", "mathematics", "statistics", "analytics"],
  python: ["computer", "mathematics", "statistics"],
  sql: ["computer", "business", "analytics"],
  statistics: ["mathematics", "computer"],
  analytics: ["computer", "business", "mathematics", "statistics"],
  dashboard: ["computer", "business"],
  ai: ["computer", "engineering", "mathematics"],
  "artificial intelligence": ["computer", "engineering", "mathematics"],
  "machine learning": ["computer", "mathematics", "statistics"],
  software: ["computer"],
  developer: ["computer"],
  cybersecurity: ["computer", "homeland"],
  nursing: ["health"],
  medicine: ["health", "biological"],
  "public health": ["health", "public administration"],
  finance: ["business"],
};

const promptAliases = {
  "data science": ["technology", "science"],
  python: ["technology", "science"],
  sql: ["technology", "business"],
  statistics: ["technology", "science"],
  analytics: ["technology", "business"],
  dashboard: ["technology", "business"],
  "machine learning": ["technology", "science"],
  "artificial intelligence": ["technology", "science"],
  ai: ["technology", "science"],
  software: ["technology"],
  developer: ["technology"],
  programming: ["technology"],
  cybersecurity: ["technology"],
  nursing: ["healthcare"],
  medicine: ["healthcare"],
  patient: ["healthcare"],
  clinical: ["healthcare"],
  "public health": ["healthcare", "public-service"],
  finance: ["business"],
  accounting: ["business"],
  management: ["business"],
  marketing: ["business", "creative"],
  teacher: ["education"],
  teaching: ["education"],
  policy: ["public-service"],
  design: ["creative"],
  construction: ["trades"],
};

function number(value) {
  return value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value);
}

function pct(value) {
  const n = number(value);
  return n === null ? "n/a" : `${n > 0 ? "+" : ""}${fmt.format(n)}%`;
}

function money(value) {
  const n = number(value);
  return n === null ? "n/a" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function formatCoachMarkdown(value) {
  const escaped = escapeHtml(value);
  return escaped
    .replace(/\*\*([^*\n][^*]*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br>");
}

function expandedSearchTerms(query) {
  const terms = [query.toLowerCase()];
  Object.entries(searchAliases).forEach(([phrase, mapped]) => {
    if (query.toLowerCase().includes(phrase)) terms.push(...mapped);
  });
  return terms.filter(Boolean);
}

function programMatchesSearch(program, query) {
  if (!query) return true;
  const haystack = `${program.cip2_name || ""} ${program.awlevel_name || ""}`.toLowerCase();
  return expandedSearchTerms(query).some((term) => haystack.includes(term));
}

function riskClass(label) {
  if (label === "High Risk") return "risk-high";
  if (label === "Moderate") return "risk-moderate";
  if (label === "Growth/Stable") return "risk-stable";
  return "";
}

function degreeMatches(program, degree) {
  if (!degree) return true;
  const award = String(program.awlevel_name || "").toLowerCase();
  const aliases = {
    associate: ["associate"],
    bachelor: ["bachelor's degree", "bachelors degree"],
    master: ["master"],
    doctoral: ["doctor", "professional degree"],
    certificate: ["certificate", "award <"],
  };
  return (aliases[degree] || [degree]).some((term) => award.includes(term));
}

function scoreProgram(program) {
  const interests = [...activeInterests];
  const name = String(program.cip2_name || "").toLowerCase();
  const degree = document.querySelector("#degreeSelect").value;
  const risk = document.querySelector("#riskSelect").value;
  const priority = document.querySelector("#prioritySelect").value;
  const keywordMap = {
    technology: ["computer", "engineering", "mathematics"],
    healthcare: ["health", "biological", "psychology"],
    business: ["business", "management", "marketing"],
    education: ["education"],
    "public-service": ["public administration", "social service", "homeland"],
    creative: ["visual", "performing", "communication", "journalism"],
    science: ["physical sciences", "natural resources", "biological"],
    trades: ["construction", "mechanic", "precision", "transportation"],
  };

  let score = 45;
  interests.forEach((interest) => {
    (keywordMap[interest] || [interest]).forEach((word) => {
      if (name.includes(word)) score += 18;
    });
  });
  if (degree && String(program.awlevel_name || "").toLowerCase().includes(degree)) score += 10;

  const bls = number(program.bls_growth_by_degree) ?? number(program.bls_occupational_growth) ?? 0;
  const openings = number(program.bls_annual_openings_mapped) ?? 0;
  const net = number(program.program_net_pct_change) ?? 0;
  score += Math.min(Math.max(bls, -20), 40) * 0.7;
  score += Math.min(openings / 100000, 8);

  if (program.sunset_label === "High Risk") score -= risk === "adventurous" ? 10 : 24;
  if (program.sunset_label === "Moderate") score -= risk === "cautious" ? 10 : 5;
  if (program.sunset_label === "Growth/Stable") score += 8;
  if (program.alignment === "Misaligned") score -= 12;
  if (["Strong", "Moderate"].includes(program.alignment)) score += 8;
  if (priority === "demand") score += Math.min(Math.max(bls, -10), 30) * 0.6;
  if (priority === "stability") score += Math.min(Math.max(net, -20), 20) * 0.4;

  return Math.max(0, Math.min(100, Math.round(score * 10) / 10));
}

function renderChips() {
  const wrap = document.querySelector("#interestChips");
  wrap.innerHTML = "";
  interestOptions.forEach(([value, label]) => {
    const button = document.createElement("button");
    button.className = `chip ${activeInterests.has(value) ? "active" : ""}`;
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => {
      activeInterests.has(value) ? activeInterests.delete(value) : activeInterests.add(value);
      renderChips();
      renderPrograms();
    });
    wrap.appendChild(button);
  });
}

function renderSummary() {
  const s = dataset.summary;
  document.querySelector("#summaryStrip").innerHTML = `
    <div class="metric"><strong>${s.program_count ?? 0}</strong><span>program credential combinations</span></div>
    <div class="metric"><strong>${s.fields ?? 0}</strong><span>IPEDS field families</span></div>
    <div class="metric"><strong>${s.high_risk_count ?? 0}</strong><span>high risk combinations</span></div>
    <div class="metric"><strong>${s.moderate_count ?? 0}</strong><span>moderate risk combinations</span></div>
  `;
}

function setupRuntimeMode() {
  if (!serverMode) {
    document.querySelector("#runtimeWarning").classList.remove("hidden");
    document.querySelector("#runResearch").disabled = true;
    document.querySelector("#refreshData").disabled = true;
  }
}

function renderResearchStrip() {
  const research = dataset.research || {};
  const thresholds = research.thresholds || {};
  const summary = dataset.summary || {};
  const years = research.years || [];
  const studyWindow = years.length ? `${years[0]}-${years[years.length - 1]}` : "2019-2024";
  const source = summary.using_fallback_outputs
    ? "Currently using previously generated research workbooks. Run research rebuilds the canonical outputs."
    : "Using outputs generated by the current research pipeline.";

  document.querySelector("#researchStrip").innerHTML = `
    <div class="research-note featured">
      <b>Evidence base</b>
      <span>Recommendations are based on ${studyWindow} IPEDS completions and BLS occupational projection signals. Sunset risk uses the research model threshold: baseline >= ${thresholds.min_baseline_for_labels ?? "n/a"}, high risk z <= ${thresholds.z_high_risk ?? "n/a"}, moderate z <= ${thresholds.z_moderate ?? "n/a"}.</span>
    </div>
    <div class="research-note compact">
      <b>Data status</b>
      <span>${source}</span>
    </div>
  `;
}

function currentProfileParams() {
  return new URLSearchParams({
    interests: [...activeInterests].join(","),
    degree: document.querySelector("#degreeSelect").value,
    risk: document.querySelector("#riskSelect").value,
    priority: document.querySelector("#prioritySelect").value,
  });
}

async function backendRecommendations() {
  if (!backendAvailable || !serverMode) return null;
  try {
    const response = await fetch(`/api/recommend?${currentProfileParams().toString()}`, { cache: "no-store" });
    if (!response.ok) throw new Error("Recommendation API unavailable");
    const payload = await response.json();
    return payload.recommendations;
  } catch {
    backendAvailable = false;
    return null;
  }
}

function filteredPrograms() {
  const search = document.querySelector("#searchInput").value.toLowerCase();
  const label = document.querySelector("#labelFilter").value;
  const alignment = document.querySelector("#alignmentFilter").value;

  return dataset.programs
    .filter((p) => degreeMatches(p, document.querySelector("#degreeSelect").value))
    .filter((p) => programMatchesSearch(p, search))
    .filter((p) => !label || p.sunset_label === label)
    .filter((p) => !alignment || p.alignment === alignment)
    .map((p) => ({ ...p, advisor_score: scoreProgram(p) }))
    .sort((a, b) => b.advisor_score - a.advisor_score)
    .slice(0, 24);
}

async function renderPrograms() {
  const search = document.querySelector("#searchInput").value.toLowerCase();
  const backend = search || agentRecommendations ? null : await backendRecommendations();
  const basePrograms = agentRecommendations || backend || filteredPrograms();
  const label = document.querySelector("#labelFilter").value;
  const alignment = document.querySelector("#alignmentFilter").value;
  const programs = basePrograms
    .filter((p) => degreeMatches(p, document.querySelector("#degreeSelect").value))
    .filter((p) => programMatchesSearch(p, search))
    .filter((p) => !label || p.sunset_label === label)
    .filter((p) => !alignment || p.alignment === alignment)
    .slice(0, 24);

  document.querySelector("#resultCount").textContent = `${programs.length} shown`;
  document.querySelector("#recommendations").innerHTML = programs
    .map((p) => {
      const score = p.advisor_score;
      return `
        <article class="program-card">
          <div>
            <h3>${p.cip2_name}</h3>
            <p>${p.awlevel_name}</p>
          </div>
          <div class="badges">
            <span class="badge ${riskClass(p.sunset_label)}">${p.sunset_label || "Unlabeled"}</span>
            <span class="badge">${p.alignment || "Alignment pending"}</span>
            <span class="badge">${p.correlation_direction || "BLS link pending"}</span>
          </div>
          <div class="score-row">
            <strong>${score}</strong>
            <div class="score-bar"><span style="width: ${score}%"></span></div>
          </div>
          <div class="stats">
            <span><b>${pct(p.program_net_pct_change)}</b>Program trend</span>
            <span><b>${pct(p.bls_growth_by_degree ?? p.bls_occupational_growth)}</b>BLS growth</span>
            <span><b>${fmt.format(number(p.bls_annual_openings_mapped) ?? 0)}</b>Openings</span>
          </div>
          <p>${p.sunset_label || "Unlabeled"} trend; ${p.alignment || "insufficient"} alignment. Use this as a shortlist signal, then compare schools, cost, geography, and admissions fit.</p>
        </article>
      `;
    })
    .join("");
}

function profileFromPrompt(prompt) {
  const text = prompt.toLowerCase();
  const interests = new Set();
  const reasons = [];
  Object.entries(promptAliases).forEach(([phrase, mapped]) => {
    if (text.includes(phrase)) {
      mapped.forEach((interest) => interests.add(interest));
      reasons.push(`Mapped "${phrase}" to ${mapped.join(", ")}.`);
    }
  });
  interestOptions.forEach(([value]) => {
    if (text.includes(value)) interests.add(value);
  });
  let degree = "";
  if (text.includes("associate")) degree = "associate";
  if (text.includes("bachelor") || text.includes("undergraduate")) degree = "bachelor";
  if (text.includes("master") || text.includes("graduate")) degree = "master";
  if (text.includes("doctoral") || text.includes("phd")) degree = "doctoral";
  if (text.includes("certificate")) degree = "certificate";
  let risk = "balanced";
  if (/(stable|safe|low risk|secure)/.test(text)) risk = "cautious";
  if (/(emerging|new|experimental|pivot)/.test(text)) risk = "adventurous";
  let priority = "balanced";
  if (/(job|jobs|demand|salary|career|employment)/.test(text)) priority = "demand";
  if (/(program stability|not declining|stable program)/.test(text)) priority = "stability";
  if (!interests.size) {
    ["technology", "business", "healthcare"].forEach((interest) => interests.add(interest));
    reasons.push("Started with broad high-demand areas because no field was explicit.");
  }
  return { interests: [...interests], degree, risk, priority, reasons };
}

function promptIntent(prompt) {
  const text = prompt.toLowerCase();
  if (/(skill|skills|good at|experience with|i know)/.test(text)) return "skills-to-degrees";
  if (/(job|career|role|occupation|in demand|salary)/.test(text)) return "job-to-degree";
  if (/(choose|compare|path|degree|institution|college|university)/.test(text)) return "path-analysis";
  return "general-coaching";
}

function intentMessage(intent) {
  return {
    "skills-to-degrees": "I treated this as a skills-to-degree question.",
    "job-to-degree": "I treated this as a job-to-degree question and prioritized labor demand.",
    "path-analysis": "I treated this as a path comparison question.",
    "general-coaching": "I treated this as an open coaching question.",
  }[intent];
}

function showCoachLoading() {
  document.querySelector("#coachResult").classList.remove("hidden");
  document.querySelector("#coachMode").textContent = "AI Coach";
  document.querySelector("#coachResultTitle").textContent = "Thinking through your options";
  document.querySelector("#coachStatus").textContent = "Thinking";
  document.querySelector("#coachAnswer").textContent = "Reviewing your prompt against IPEDS/BLS-backed pathways...";
  document.querySelector("#coachInterpretation").textContent = "";
  document.querySelector("#askAdvisor").disabled = true;
  document.querySelector("#askAdvisor").textContent = "Thinking...";
}

function showCoachResult({ mode, answer, profile, reasoning, nextQuestion }) {
  document.querySelector("#coachResult").classList.remove("hidden");
  document.querySelector("#coachMode").textContent = mode === "llm" ? "Local LLM Coach" : "Rules Coach";
  document.querySelector("#coachResultTitle").textContent = "Coach recommendation";
  document.querySelector("#coachStatus").textContent = "Done";
  document.querySelector("#coachAnswer").innerHTML = `<p>${formatCoachMarkdown(answer)}</p>${nextQuestion ? `<p class="coach-next"><b>Next:</b> ${escapeHtml(nextQuestion)}</p>` : ""}`;
  const interpreted = profile
    ? `Interpreted as: ${(profile.interests || []).join(", ")}${profile.degree_level ? ` · ${profile.degree_level}` : ""}.`
    : "";
  document.querySelector("#coachInterpretation").textContent = [interpreted, ...(reasoning || []).slice(0, 2)].filter(Boolean).join(" ");
  document.querySelector("#askAdvisor").disabled = false;
  document.querySelector("#askAdvisor").textContent = "Ask coach";
}

function showCoachJobs(jobs) {
  if (!jobs || !jobs.length) return;
  const wrap = document.createElement("div");
  wrap.className = "coach-jobs";
  wrap.innerHTML = `
    <div class="coach-jobs-title">Related job designations from BLS mappings</div>
    ${jobs.slice(0, 8).map((job) => `
    <div class="coach-job">
      <b>${escapeHtml(job.title)}</b>
      <span>${escapeHtml(job.related_field || "")}${job.projected_growth !== null ? ` · ${pct(job.projected_growth)} growth` : ""}${job.annual_openings !== null ? ` · ${fmt.format(number(job.annual_openings) ?? 0)} openings` : ""}</span>
      ${(job.skills || []).length ? `<span>Skills: ${(job.skills || []).slice(0, 4).map(escapeHtml).join(", ")}</span>` : ""}
    </div>
  `).join("")}`;
  document.querySelector("#coachAnswer").appendChild(wrap);
}

function showCoachInstitutions(institutions) {
  if (!institutions || !institutions.length) return;
  const wrap = document.createElement("div");
  wrap.className = "coach-jobs";
  wrap.innerHTML = `
    <div class="coach-jobs-title">Institutions matching your available location and cost constraints</div>
    ${institutions.slice(0, 6).map((school) => `
      <div class="coach-job">
        <b>${escapeHtml(school.institution_name || school.scorecard_name || `UNITID ${school.unitid}`)}</b>
        <span>${escapeHtml([school.city || school.scorecard_city, school.state || school.scorecard_state].filter(Boolean).join(", "))}${school.average_net_price !== null && school.average_net_price !== undefined ? ` · ${money(school.average_net_price)} avg. net price` : ""}${school.completion_rate !== null && school.completion_rate !== undefined ? ` · ${pct(number(school.completion_rate) * 100)} completion` : ""}</span>
      </div>
    `).join("")}`;
  document.querySelector("#coachAnswer").appendChild(wrap);
}

async function askAdvisor() {
  const prompt = document.querySelector("#studentPrompt").value.trim();
  if (!prompt) return;
  showCoachLoading();
  const parsed = profileFromPrompt(prompt);
  activeInterests = new Set(parsed.interests);
  document.querySelector("#degreeSelect").value = parsed.degree;
  document.querySelector("#riskSelect").value = parsed.risk;
  document.querySelector("#prioritySelect").value = parsed.priority;
  document.querySelector("#searchInput").value = "";
  renderChips();

  try {
    if (!serverMode) throw new Error("Server unavailable in file mode");
    const response = await fetch("/api/chat", {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, session_id: coachSessionId }),
    });
    if (response.ok) {
      const payload = await response.json();
      coachSessionId = payload.session_id || coachSessionId;
      agentRecommendations = payload.recommendations;
      showCoachResult({
        mode: payload.mode,
        answer: payload.coach_answer || payload.reasoning[0],
        profile: payload.profile,
        reasoning: payload.reasoning,
        nextQuestion: payload.next_question,
      });
      showCoachJobs(payload.job_designations);
      showCoachInstitutions(payload.institutions);
      await renderPrograms();
      return;
    }
  } catch {
    // Static file mode uses the local parser below.
  }

  agentRecommendations = dataset.programs
    .filter((program) => degreeMatches(program, parsed.degree))
    .map((program) => ({ ...program, advisor_score: scoreProgram(program) }))
    .sort((a, b) => b.advisor_score - a.advisor_score)
    .slice(0, 12);
  showCoachResult({
    mode: "rules",
    answer: intentMessage(promptIntent(prompt)),
    profile: {
      interests: parsed.interests,
      degree_level: parsed.degree,
    },
    reasoning: parsed.reasons,
  });
  await renderPrograms();
}

async function loadData() {
  const response = await fetch("data/programs.json", { cache: "no-store" });
  dataset = await response.json();
  const dimensionsResponse = await fetch("data/dimensions.json", { cache: "no-store" });
  dimensions = await dimensionsResponse.json();
  if (serverMode) {
    try {
      const capabilities = await fetch("/api/capabilities", { cache: "no-store" }).then((response) => response.json());
      document.querySelector("#refreshData").disabled = !capabilities.admin;
      document.querySelector("#runResearch").disabled = !capabilities.admin;
      if (!capabilities.admin) {
        document.querySelector("#refreshData").title = "Administrative data refresh is disabled on this deployment";
        document.querySelector("#runResearch").title = "Administrative research execution is disabled on this deployment";
      }
    } catch {
      // Static hosting has no capabilities endpoint.
    }
  }
  renderSummary();
  renderResearchStrip();
  renderPrograms();
  populateJobFilters();
  renderJobSignals();
  renderDimensions();
}

function populateJobFilters() {
  setOptions("#jobDegreeFilter", [...new Set((dataset.programs || []).map((row) => row.awlevel_name))], "All degree levels");
}

function alignmentRank(value) {
  return { Strong: 4, Moderate: 3, Weak: 2, Misaligned: 1 }[value] || 0;
}

function renderJobSignals() {
  const query = document.querySelector("#jobSearch").value.toLowerCase();
  const degree = document.querySelector("#jobDegreeFilter").value;
  const sort = document.querySelector("#jobSort").value;
  const rows = (dataset.programs || [])
    .filter((row) => row.bls_growth_by_degree !== null || row.bls_occupational_growth !== null)
    .filter((row) => !query || programMatchesSearch(row, query))
    .filter((row) => !degree || row.awlevel_name === degree)
    .map((row) => ({
      ...row,
      growth_value: number(row.bls_growth_by_degree) ?? number(row.bls_occupational_growth) ?? 0,
      openings_value: number(row.bls_annual_openings_mapped) ?? 0,
    }));
  rows.sort((a, b) => {
    if (sort === "openings") return b.openings_value - a.openings_value;
    return b.growth_value - a.growth_value;
  });
  document.querySelector("#jobSignals").innerHTML = rows.slice(0, 14).map((row) => `
    <div class="table-row job-row">
      <div>
        <b>${escapeHtml(row.cip2_name)}</b>
        <span>Related degree: ${escapeHtml(row.awlevel_name)}</span>
        <div class="job-titles">
          ${(row.job_designations || []).slice(0, 4).map((job) => `<span title="${escapeHtml(job.typical_education || "")}">${escapeHtml(job.title)}</span>`).join("") || "<span>Designations pending</span>"}
        </div>
      </div>
      <div><b>${pct(row.growth_value)}</b><span>BLS growth</span></div>
      <div><b>${fmt.format(row.openings_value)}</b><span>openings</span></div>
    </div>
  `).join("");
}

function renderDimensions() {
  const summary = dimensions.summary || {};
  document.querySelector("#dimensionSummary").textContent =
    `${summary.institutions ?? 0} institutions; ${summary.geography_note ?? "geography pending"}`;
  document.querySelector("#institutionGeoNote").textContent = summary.geography_note ?? "";
  if (summary.has_scorecard) {
    document.querySelector("#institutionGeoNote").textContent += ` College Scorecard outcomes cached ${summary.scorecard_retrieved_at || "date unavailable"}.`;
  } else {
    document.querySelector("#institutionGeoNote").textContent += " Add a free College Scorecard API key and run the cache command to include price and outcomes.";
  }
  populateDimensionFilters();
  renderInstitutionTrends();
  renderGenderMix();
}

function setOptions(selectId, values, defaultLabel) {
  const select = document.querySelector(selectId);
  const current = select.value;
  select.innerHTML = `<option value="">${defaultLabel}</option>` + values
    .filter(Boolean)
    .sort((a, b) => String(a).localeCompare(String(b)))
    .map((value) => `<option value="${value}">${value}</option>`)
    .join("");
  select.value = values.includes(current) ? current : "";
}

function populateDimensionFilters() {
  const institutionRows = dimensions.institution_trends || [];
  const demographicRows = [
    ...(dimensions.demographics?.gender || []),
    ...(dimensions.demographics?.race || []),
  ];
  setOptions("#stateFilter", [...new Set(institutionRows.map((row) => row.state))], "All states");
  setOptions("#institutionFieldFilter", [...new Set(institutionRows.map((row) => row.cip2_name))], "All fields");
  setOptions("#demographicFieldFilter", [...new Set(demographicRows.map((row) => row.cip2_name))], "All fields");
  setOptions("#demographicAwardFilter", [...new Set(demographicRows.map((row) => row.awlevel_name))], "All award levels");
}

function renderInstitutionTrends() {
  const direction = document.querySelector("#trendDirection").value;
  const query = document.querySelector("#institutionSearch").value.toLowerCase();
  const state = document.querySelector("#stateFilter").value;
  const field = document.querySelector("#institutionFieldFilter").value;
  const maxCostValue = document.querySelector("#institutionCostFilter").value;
  const maxCost = maxCostValue ? number(maxCostValue) : null;
  const rows = (dimensions.institution_trends || [])
    .filter((row) => row.trend_direction === direction)
    .filter((row) => !query || `${row.institution_name || ""} ${row.unitid}`.toLowerCase().includes(query))
    .filter((row) => !state || row.state === state)
    .filter((row) => !field || row.cip2_name === field)
    .filter((row) => maxCost === null || (number(row.average_net_price) !== null && number(row.average_net_price) <= maxCost))
    .slice(0, 10);
  document.querySelector("#institutionTrends").innerHTML = rows
    .map((row) => `
      <div class="table-row">
        <div>
          <b>${row.institution_name || `UNITID ${row.unitid}`}</b>
          <span>${[row.city, row.state].filter(Boolean).join(", ") || "Institution name/geography pending"} · ${row.cip2_name} / ${row.awlevel_name}</span>
          ${row.average_net_price !== null && row.average_net_price !== undefined ? `<span>Scorecard: ${money(row.average_net_price)} avg. net price · ${pct((number(row.completion_rate) ?? 0) * 100)} completion · ${money(row.median_earnings_10yr)} median earnings</span>` : ""}
        </div>
        <div><b>${fmt.format(number(row.change_2019_2024) ?? 0)}</b><span>change</span></div>
        <div><b>${pct(row.pct_change_2019_2024)}</b><span>2019-24</span></div>
      </div>
    `)
    .join("");
}

function renderGenderMix() {
  const type = document.querySelector("#demographicType").value;
  const field = document.querySelector("#demographicFieldFilter").value;
  const award = document.querySelector("#demographicAwardFilter").value;
  document.querySelector("#demographicTitle").textContent = type === "race" ? "Race / ethnicity mix" : "Gender mix";
  document.querySelector("#demographicLegend").innerHTML = type === "race"
    ? `<span><i class="swatch race-1"></i>Largest group</span><span><i class="swatch race-2"></i>Second</span><span><i class="swatch race-3"></i>Third</span>`
    : `<span><i class="swatch women"></i>Women</span><span><i class="swatch men"></i>Men</span>`;
  const rows = (dimensions.demographics?.[type] || [])
    .filter((row) => !field || row.cip2_name === field)
    .filter((row) => !award || row.awlevel_name === award)
    .slice(0, 10);
  document.querySelector("#genderMix").innerHTML = rows
    .map((row) => {
      if (type === "race") {
        const raceKeys = Object.keys(row).filter((key) =>
          !["cip2", "cip2_name", "awlevel", "awlevel_name", "total"].includes(key)
        );
        const top = raceKeys
          .map((key) => [key, number(row[key]) ?? 0])
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3);
        const total = number(row.total) ?? 0;
        return `
          <div class="table-row demographic-row">
            <div>
              <b>${row.cip2_name}</b>
              <span>${row.awlevel_name}</span>
            </div>
            <div class="race-stack">
              ${top.map(([name, value], index) => `
                <div class="race-line">
                  <span>${name}</span>
                  <div class="mix-bar race-${index + 1}"><span style="width:${total ? Math.round((value / total) * 100) : 0}%"></span></div>
                  <b>${total ? fmt.format((value / total) * 100) : "0"}%</b>
                </div>
              `).join("")}
            </div>
            <div class="total-cell"><b>${fmt.format(total)}</b><span>total</span></div>
          </div>
        `;
      }
      const womenShare = number(row.women_share) ?? 0;
      const menShare = number(row.men_share) ?? 0;
      return `
        <div class="table-row demographic-row">
          <div>
            <b>${row.cip2_name}</b>
            <span>${row.awlevel_name}</span>
          </div>
          <div class="mix-bars labeled">
            <div class="mix-line">
              <span class="mix-label">Women</span>
              <div class="mix-bar women" title="Women"><span style="width:${Math.round(womenShare * 100)}%"></span></div>
              <b>${fmt.format(womenShare * 100)}%</b>
            </div>
            <div class="mix-line">
              <span class="mix-label">Men</span>
              <div class="mix-bar men" title="Men"><span style="width:${Math.round(menShare * 100)}%"></span></div>
              <b>${fmt.format(menShare * 100)}%</b>
            </div>
          </div>
          <div class="total-cell"><b>${fmt.format(number(row.total) ?? 0)}</b><span>total</span></div>
        </div>
      `;
    })
    .join("");
}

async function refreshData(runResearch) {
  const button = runResearch ? document.querySelector("#runResearch") : document.querySelector("#refreshData");
  const original = button.textContent;
  if (runResearch) {
    await startResearchRun();
    return;
  }
  button.textContent = "Refreshing...";
  button.disabled = true;
  const response = await fetch("/api/refresh").catch(() => null);
  if (response && !response.ok) {
    const payload = await response.json().catch(() => ({}));
    alert(payload.error || "Refresh failed");
  }
  backendAvailable = true;
  await loadData();
  button.textContent = original;
  button.disabled = false;
}

function renderRunStatus(job) {
  const panel = document.querySelector("#runStatus");
  panel.classList.remove("hidden");
  document.querySelector("#runStage").textContent = job.stage || "Running";
  document.querySelector("#runMessage").textContent = job.message || "";
  document.querySelector("#runState").textContent = job.status || "running";
  document.querySelector("#runLog").innerHTML = (job.logs || [])
    .slice(-18)
    .map((line) => `<div>${line.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]))}</div>`)
    .join("");
}

async function pollResearchRun() {
  const response = await fetch("/api/run-status", { cache: "no-store" });
  const payload = await response.json();
  const job = payload.job || {};
  renderRunStatus(job);
  if (!job.running) {
    clearInterval(runPollTimer);
    runPollTimer = null;
    document.querySelector("#runResearch").textContent = "Run research";
    document.querySelector("#runResearch").disabled = false;
    if (job.status === "completed") {
      backendAvailable = true;
      await loadData();
    }
  }
}

async function startResearchRun() {
  const button = document.querySelector("#runResearch");
  button.textContent = "Starting...";
  button.disabled = true;
  const response = await fetch("/api/run-research", { cache: "no-store" });
  const payload = await response.json();
  renderRunStatus(payload.job || {});
  button.textContent = "Running...";
  if (runPollTimer) clearInterval(runPollTimer);
  runPollTimer = setInterval(() => {
    pollResearchRun().catch(() => null);
  }, 2000);
  await pollResearchRun();
}

setupRuntimeMode();
document.querySelectorAll("select, input").forEach((el) => el.addEventListener("input", () => renderPrograms()));
document.querySelector("#refreshData").addEventListener("click", async () => {
  await refreshData(false);
});
document.querySelector("#runResearch").addEventListener("click", async () => {
  await refreshData(true);
});
document.querySelector("#askAdvisor").addEventListener("click", askAdvisor);
document.querySelector("#studentPrompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) askAdvisor();
});
document.querySelectorAll(".coach-examples button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelector("#studentPrompt").value = button.dataset.prompt;
    document.querySelector("#studentPrompt").focus();
  });
});
document.querySelector("#searchInput").addEventListener("input", () => {
  agentRecommendations = null;
});
document.querySelector("#trendDirection").addEventListener("input", renderInstitutionTrends);
document.querySelector("#institutionSearch").addEventListener("input", renderInstitutionTrends);
document.querySelector("#stateFilter").addEventListener("input", renderInstitutionTrends);
document.querySelector("#institutionFieldFilter").addEventListener("input", renderInstitutionTrends);
document.querySelector("#institutionCostFilter").addEventListener("input", renderInstitutionTrends);
document.querySelector("#demographicType").addEventListener("input", renderGenderMix);
document.querySelector("#demographicFieldFilter").addEventListener("input", renderGenderMix);
document.querySelector("#demographicAwardFilter").addEventListener("input", renderGenderMix);
document.querySelector("#jobSearch").addEventListener("input", renderJobSignals);
document.querySelector("#jobDegreeFilter").addEventListener("input", renderJobSignals);
document.querySelector("#jobSort").addEventListener("input", renderJobSignals);
document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab-button").forEach((tab) => tab.classList.remove("active"));
    document.querySelectorAll(".view-panel").forEach((panel) => panel.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#${button.dataset.view}`).classList.add("active");
  });
});

renderChips();
loadData().catch((error) => {
  document.querySelector("#recommendations").innerHTML = `<p>${error.message}</p>`;
});
