// Boots Pyodide, pip-installs the real ragevallab wheel, and drives it.
// Nothing here reimplements the library — every number on the page comes back
// from the actual Python running in WebAssembly.

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const statusText = $("statusText");

function setStatus(text, state) {
  statusText.textContent = text;
  statusEl.className = "status" + (state ? " " + state : "");
}

const tone = (v) => (v >= 0.9 ? "good" : v >= 0.6 ? "warn" : "bad");
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let py = null;

async function boot() {
  try {
    setStatus("Booting Python (WebAssembly)…");
    py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v314.0.2/full/" });

    setStatus("Installing the ragevallab wheel…");
    await py.loadPackage("micropip");
    const micropip = py.pyimport("micropip");
    // WHEEL_URL is written by the deploy workflow (it knows the built filename).
    await micropip.install(window.WHEEL_URL || "./ragevallab-0.1.0-py3-none-any.whl");

    setStatus("Indexing the corpus…");
    await py.runPythonAsync(`
import json
from ragevallab.data import SAMPLE_DOCS, EVAL_SET, PLANTED
from ragevallab.pipeline import RagPipeline
from ragevallab.evals import faithfulness, evaluate, FAITHFULNESS_THRESHOLD, _content_tokens
from ragevallab.cli import run_eval
import ragevallab

# One pipeline, reused by both panels — same object the CLI builds.
PIPE = RagPipeline().ingest(SAMPLE_DOCS)

def corpus_json():
    return json.dumps(SAMPLE_DOCS)

def eval_json():
    run = run_eval(k=4, out="/tmp/eval_run.json")
    return json.dumps(run.to_dict())

def score_json(question, answer):
    """Retrieve real context for the question, then score the user's answer against it."""
    ans = PIPE.answer(question, k=4)
    supported = set()
    for c in ans.contexts:
        supported |= set(_content_tokens(c))
    toks = [{"t": t, "grounded": t in supported} for t in _content_tokens(answer)]
    f = faithfulness(answer, ans.contexts)
    return json.dumps({
        "faithfulness": round(f, 3),
        "threshold": FAITHFULNESS_THRESHOLD,
        "flagged": f < FAITHFULNESS_THRESHOLD,
        "tokens": toks,
        "retrieved": ans.retrieved,
        "contexts": ans.contexts,
        "version": ragevallab.__version__,
    })
`);

    const version = await py.runPythonAsync("ragevallab.__version__");
    setStatus(`Ready — ragevallab ${version} installed and running in this tab`, "ready");
    $("runEval").disabled = false;
    $("score").disabled = false;
    $("fool").disabled = false;
    renderCorpus(JSON.parse(await py.runPythonAsync("corpus_json()")));
  } catch (err) {
    setStatus("Failed to boot: " + err, "err");
    console.error(err);
  }
}

function renderCorpus(docs) {
  $("corpus").innerHTML =
    "<div style='color:var(--fg-dim);margin-bottom:8px'>The indexed corpus — your answer is scored against whatever this retrieves:</div>" +
    Object.entries(docs)
      .map(([id, text]) => `<div><b>${esc(id)}#0</b> — ${esc(text)}</div>`)
      .join("");
}

// Hand the run we just produced to the companion dashboard. Both demos are
// served from egnaro9.github.io, so they share an origin and can pass the JSON
// through localStorage — the eval never round-trips through a server.
const HANDOFF_KEY = "ragevallab:eval_run";
const DASHBOARD_URL = "https://egnaro9.github.io/eval-dashboard/?from=rag-eval-lab";

function offerHandoff(run) {
  try {
    localStorage.setItem(HANDOFF_KEY, JSON.stringify(run));
  } catch {
    return; // storage unavailable (private mode) — just don't offer it
  }
  $("handoff").innerHTML =
    `<a class="handoff-btn" href="${DASHBOARD_URL}" target="_blank" rel="noopener">` +
    `Open this run in the dashboard →</a>` +
    `<div class="handoff-note">Sends the run you just generated to ` +
    `<a href="https://github.com/egnaro9/eval-dashboard" target="_blank" rel="noopener">eval-dashboard</a>` +
    ` — the other half of the pipeline. Same origin, so it's handed over directly.</div>`;
}

async function runEval() {
  const btn = $("runEval");
  btn.disabled = true;
  btn.textContent = "Running…";
  $("evalOut").textContent = "$ python -m ragevallab.cli eval\n";
  try {
    const run = JSON.parse(await py.runPythonAsync("eval_json()"));
    const m = run.metrics;
    offerHandoff(run);

    $("evalOut").textContent +=
      `run: ${run.run}\n` +
      Object.entries(m).map(([k, v]) => `  ${k.padStart(14)}: ${v}`).join("\n") +
      `\n\n${m.flagged_cases} flagged case(s):\n` +
      run.cases.filter((c) => c.flagged)
        .map((c) => `  ! ${c.q}\n    answer: ${c.answer}\n    faithfulness=${c.scores.faithfulness}`)
        .join("\n");

    const cards = [
      ["Faithfulness", pct(m.faithfulness), tone(m.faithfulness)],
      ["Precision@k", pct(m["precision@k"]), tone(m["precision@k"])],
      ["Recall@k", pct(m["recall@k"]), tone(m["recall@k"])],
      ["Citation rate", pct(m.citation_rate), tone(m.citation_rate)],
      ["Flagged", `${m.flagged_cases} / ${m.n_cases}`, m.flagged_cases > 0 ? "bad" : "good"],
    ];
    $("evalCards").innerHTML = cards
      .map(([k, v, t]) => `<div class="card"><div class="k">${k}</div><div class="v ${t}">${v}</div></div>`)
      .join("");

    $("evalTable").innerHTML =
      `<table><thead><tr><th>Question</th><th>Answer</th><th>Retrieved</th><th>Faithful</th><th>Status</th></tr></thead><tbody>` +
      run.cases.map((c) => `
        <tr class="${c.flagged ? "flagged" : ""}">
          <td>${esc(c.q)}</td>
          <td style="color:var(--fg-dim)">${esc(c.answer)}${c.note ? `<div style="color:var(--red);font-size:12px;margin-top:4px">${esc(c.note)}</div>` : ""}</td>
          <td class="mono" style="color:var(--amber-ink)">${c.retrieved.join(", ")}</td>
          <td class="mono ${tone(c.scores.faithfulness)}">${pct(c.scores.faithfulness, 0)}</td>
          <td>${c.flagged ? '<span class="bad">🚩 flagged</span>' : '<span class="good">✓ ok</span>'}</td>
        </tr>`).join("") +
      `</tbody></table>`;
    btn.textContent = "▶ Run it again";
  } catch (err) {
    $("evalOut").textContent += "\nError: " + err;
  }
  btn.disabled = false;
}

async function score() {
  const btn = $("score");
  btn.disabled = true;
  const q = $("q").value.trim();
  const a = $("a").value.trim();
  if (!q || !a) { btn.disabled = false; return; }
  try {
    const r = JSON.parse(
      await py.runPythonAsync(`score_json(${JSON.stringify(q)}, ${JSON.stringify(a)})`)
    );
    const ungrounded = r.tokens.filter((t) => !t.grounded);
    $("verdict").innerHTML = `
      <div class="verdict ${r.flagged ? "bad" : "ok"}">
        <h3>${r.flagged ? "🚩 Flagged as a hallucination" : "✓ Grounded"}</h3>
        <div style="font-family:var(--mono);font-size:13px;color:var(--fg-dim)">
          faithfulness = <span class="${tone(r.faithfulness)}"><b>${r.faithfulness}</b></span>
          &nbsp;·&nbsp; threshold = ${r.threshold}
          &nbsp;·&nbsp; retrieved: <span style="color:var(--amber-ink)">${r.retrieved.join(", ")}</span>
        </div>
        <div style="margin-top:12px">${r.tokens
          .map((t) => `<span class="tok ${t.grounded ? "g" : "u"}">${esc(t.t)}</span>`)
          .join("")}</div>
        <div style="margin-top:10px;color:var(--fg-dim);font-size:13.5px">
          ${ungrounded.length === 0
            ? "Every content word in your answer appears in the retrieved context."
            : `<b>${ungrounded.length} of ${r.tokens.length}</b> content words aren't supported by anything retrieved${ungrounded.length ? " — struck through above" : ""}.`}
        </div>
      </div>`;
  } catch (err) {
    $("verdict").innerHTML = `<div class="verdict bad">Error: ${esc(err)}</div>`;
  }
  btn.disabled = false;
}

const HALLUCINATIONS = [
  ["Which planet is the hottest in the Solar System?", "Neptune is the hottest planet because of its volcanic geysers."],
  ["What is the tallest volcano in the Solar System?", "The tallest volcano is Mount Kilimanjaro, located on Titan."],
  ["Which planet is famous for its ring system?", "Uranus is famous for its rings, which are made of solid diamond."],
  ["What is the largest planet?", "The largest planet is Kepler-9, roughly triple the mass of the Sun."],
];
let hIdx = 0;
function giveHallucination() {
  const [q, a] = HALLUCINATIONS[hIdx++ % HALLUCINATIONS.length];
  $("q").value = q;
  $("a").value = a;
  score();
}

$("runEval").addEventListener("click", runEval);
$("score").addEventListener("click", score);
$("fool").addEventListener("click", giveHallucination);
boot();
