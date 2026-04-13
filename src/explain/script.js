async function run() {
  const text = document.getElementById("text").value;

  const res = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
  });

  const data = await res.json();
  render(data);
}
  const vec = (v, limit = 6) => {
    if (!Array.isArray(v)) return "[0, 0, 0, 0, 0, 0, ...]";

    const body = v
      .slice(0, limit)
      .map(x => Number(x).toFixed(3))
      .join(", ");

    return v.length > limit ? `[${body}, ...]` : `[${body}]`;
  };

  const matrix = (v, rowLimit = 5, colLimit = 10) => {
    if (!Array.isArray(v)) return "[]";

    const rows = [];
    for (let r = 0; r < rowLimit; r++) {
      const row = Array.isArray(v[r]) ? v[r] : [];
      const cells = [];
      for (let c = 0; c < colLimit; c++) {
        const n = Number(row[c]);
        cells.push(Number.isNaN(n) ? "0.000" : n.toFixed(3));
      }
      rows.push(`[${cells.join(", ")}...]`);
    }

    return `[<br>&nbsp;&nbsp;${rows.join(",<br>&nbsp;&nbsp;")}<br>&nbsp;&nbsp;...]`;
  };

// =====================
// FORMAT VECTOR SAFE
// =====================
function formatVector(v, limit = 6, precision = 3) {
  if (!Array.isArray(v)) return "[]";

  const clean = v
    .filter(x => x !== null && x !== undefined)
    .slice(0, limit)
    .map(x => {
      const num = Number(x);
      return Number.isNaN(num) ? "0.000" : num.toFixed(precision);
    });

  if (clean.length === 0) return "[]";

  return v.length > limit
    ? `${clean.join(", ")}, ...`
    : clean.join(", ");
}

// Return bracketed vector string with fixed precision and zero fallback
function displayVec(v, limit = 5, precision = 3) {
  if (!Array.isArray(v) || v.length === 0) {
    return `[${Array(limit).fill('0.000').join(', ')}, ...]`;
  }

  const body = v
    .slice(0, limit)
    .map(x => {
      const n = Number(x);
      return Number.isNaN(n) ? '0.000' : n.toFixed(precision);
    })
    .join(', ');

  return v.length > limit ? `[${body}, ...]` : `[${body}]`;
}

// =====================
// BUILD LSTM INPUT TENSOR
// =====================
function buildLSTMInput(vectors) {
  if (!Array.isArray(vectors)) return "";

  return vectors
    .slice(0, 6) // 👈 FIX QUAN TRỌNG Ở ĐÂY (word level)
    .map(v => {
      if (!Array.isArray(v)) return "[]";

      const limited = v
        .slice(0, 6)
        .map(x => {
          const num = Number(x);
          return isNaN(num) ? "0.000" : num.toFixed(3);
        });

      return `[${limited.join(", ")}...]`;
    })
    .join(",<br>") + (vectors.length > 6 ? "<br>..." : "");
}
// =====================
// WORD VECTOR RENDER
function renderVectors(tokens, vectors) {
  return (tokens || [])
    .slice(0, 6) // 👈 giới hạn 6 từ
    .map((t, i) => {
      const v = vectors?.[i] || [];

      const formatted = v
        .slice(0, 6)
        .map(x => {
          const num = Number(x);
          return isNaN(num) ? "0.000" : num.toFixed(3);
        })
        .join(", ");

      return `
        <div class="vector-row">
          <b>${t}</b> → [${formatted}]
        </div>
      `;
    })
    .join("") 
    + ((tokens || []).length > 6 ? "<div>...</div>" : "");
}
// =====================
// STEP 1 LSTM EXPLAIN
// =====================
function renderStep1(inputVector, h_prev, c_prev, t = 0) {

  const isFirstStep = t === 0;
 
  const vec = (v, limit = 5) => {
    if (!Array.isArray(v) || v.length === 0) {
      return `[${Array(limit).fill('0.000').join(', ')}, ...]`;
    }

    const body = v.slice(0, limit).map(x => {
      const n = Number(x);
      return Number.isNaN(n) ? '0.000' : n.toFixed(3);
    }).join(', ');

    return v.length > limit ? `[${body}, ...]` : `[${body}]`;
  };

  const hState = vec(isFirstStep ? [] : (h_prev || []));
  const cState = vec(isFirstStep ? [] : (c_prev || []));

  return `
    <div class="lstm-step output-step">
      <h3>📍 Bước ${t + 1}: Nhận input tại thời điểm t = ${t}</h3>

      <p>
        <b>Nội dung:</b> xₜ là input mới,
        hₜ₋₁ và Cₜ₋₁ là bộ nhớ từ bước trước.
      </p>

      <div class="code">
        <b>xₜ (input vector):</b><br>
            ${Array.isArray(inputVector)
            ? `[${inputVector
                .slice(0, 6)
                .map(x => {
                    const num = Number(x);
                    return Number.isNaN(num) ? "0.000" : num.toFixed(3);
                })
                .join(", ")}, ...]`
            : "[]"}
        </div>

      <div class="code">
        <b>hₜ₋₁ (hidden state):</b><br>
        ${hState}
      </div>

      <div class="code">
        <b>Cₜ₋₁ (cell state):</b><br>
        ${cState}
      </div>

      <p>
        &rarr; Ở <b>t = 0</b>, LSTM chưa có memory nên khởi tạo toàn bộ bằng 0.
        Sau đó nó bắt đầu học dần qua từng từ.
      </p>
    </div>
  `;
}

// =====================
// MAIN RENDER
// =====================
function render(data) {
  const div = document.getElementById("result");

  if (data.error) {
    div.innerHTML = `<div class="card">❌ ${data.error}</div>`;
    return;
  }
  const tokenCount = (data.tokens || []).length;
  const labelClass = data.label === "spam" ? "spam" : "ham";

  const lstmInput = buildLSTMInput(data.vectors);

  const step1 = renderStep1(
    data.vectors?.[0] || [],
    data.hidden_prev || [],
    data.cell_prev || []
  );
  const step2 = renderStep2(
    data.vectors?.[0],
    data.hidden_prev,
    data.cell_prev,
    data.W_f,
    data.b_f,
    0
  );
  const step3 = renderStep3(
  data.vectors?.[0] || [],
  data.hidden_prev || [],
  data.W_i,
  data.b_i,
  0
);
  const step4 = renderStep4(
    data.vectors?.[0] || [],
    data.hidden_prev || [],
    data.W_c,
    data.b_c,
    0
  );
  const forgetStep = computeForgetGate(
    data.W_f,
    data.b_f,
    data.hidden_prev || [],
    data.vectors?.[0] || []
  );
  const inputStep = computeInputGate(data.W_i, data.b_i, data.vectors?.[0] || []);
  const candidateStep = computeCandidateMemory(
    data.W_c,
    data.b_c,
    data.hidden_prev || [],
    data.vectors?.[0] || []
  );
  const step5 = renderStep5(
    forgetStep.full,
    data.cell_prev || [],
    inputStep.i_demo,
    candidateStep.c_tilde,
    0
  );
  const cellStateResult = computeCellState(
    forgetStep.full,
    data.cell_prev || [],
    inputStep.i_demo,
    candidateStep.c_tilde
  );
  const step6 = renderStep6(
    data.vectors?.[0] || [],
    data.hidden_prev || [],
    cellStateResult.C_t,
    data.W_o,
    data.b_o,
    0
  );
  div.innerHTML = `
    <div class="card">
      <h2>📌 Prediction</h2>
      <p><b>Tokens:</b> ${tokenCount}</p>
      <p><b>Probability:</b> ${data.prob}</p>
      <p><b>Label:</b> <span class="${labelClass}">${data.label.toUpperCase()}</span></p>

      <h3>Tokens</h3>
        <div class="code" style="margin-bottom: 8px;">${(data.tokens || []).join(", ")}</div>

      <h3> Word2Vec</h3>
      ${renderVectors(data.tokens, data.vectors)}

      <h3>LSTM Flow</h3>
      <div class="flow">
        <div class="node f">Forget</div>
        <div class="node i">Input</div>
        <div class="node c">Cell</div>
        <div class="node o">Output</div>
        <div class="node h">Hidden</div>
      </div>

      <div class="lstm-box">
        <b>Input to LSTM (sequence tensor)</b>
        <div class="code">
          [<br>
          ${lstmInput}
          <br>]
        </div>
      </div>


      ${step1}
      ${step2}
      ${step3}
      ${step4}

      <div class="card" style="margin-top: 16px; border: 2px solid #6c5ce7;">
        <h3>✅ Bước 5: Cell State Update</h3>
        <p>Bước này tổng hợp các gate trước đó để cập nhật bộ nhớ dài hạn.</p>
        ${step5}
      </div>

      <div class="card output-card" style="margin-top: 16px;">
        <h3>✅ Bước 6: Output Gate & Hidden State</h3>
        <p>Bước này quyết định phần nào của Cₜ được xuất ra thành hₜ.</p>
        ${step6}
      </div>

    </div>
  `;

  if (window.MathJax) {
    MathJax.typeset();
  }
}

function computeForgetGate(W_f, b_f, h_prev, x_t) {

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  const h = Array.isArray(h_prev) ? h_prev : [];
  const x = Array.isArray(x_t) ? x_t : [];

  const input = [...h, ...x]; // concat

  const hidden = b_f.length; // output size

  const f_t = [];

  for (let i = 0; i < hidden; i++) {

    let z = 0;

    for (let j = 0; j < input.length; j++) {
      z += W_f[i][j] * input[j];
    }

    z += b_f[i];

    f_t.push(sigmoid(z));
  }

  return {
    full: f_t,
    display: f_t.slice(0, 6)
  };
}

function formatVectorHidden(v, count = 5) {
  if (!Array.isArray(v)) return "[]";

  const dots = Array.from({ length: count }, () => "..").join(", ");

  return `[${dots}${v.length > count ? ", ..." : ""}]`;
}

function renderStep2(inputVector, h_prev, c_prev, W_f, b_f, t = 0) {

  const isFirstStep = t === 0;

  // =====================
  // FORMAT
  // =====================
  const vec = (v, limit = 6) => {
    if (!Array.isArray(v)) return "[]";

    const body = v
      .slice(0, limit)
      .map(x => Number(x).toFixed(3))
      .join(", ");

    return v.length > limit ? `[${body}, ...]` : `[${body}...]`;
  };
    const formula = `
    fₜ = σ(W_f · [hₜ₋₁, xₜ] + b_f)
  `;

  const stepByStep = `
    Step 1: z = W_f · [hₜ₋₁, xₜ] + b_f<br>
    Step 2: fₜ = σ(z)
  `;
  // flatten safety
  const flat = (arr) =>
    Array.isArray(arr) ? arr.flat(Infinity).map(x => Number(x) || 0) : [];

  const h = isFirstStep ? (h_prev || []).map(() => 0) : flat(h_prev);
  const x = flat(inputVector);

  const hx = [...h, ...x];

  // =====================
  // DOT PRODUCT
  // =====================
  const dot = (W, X) => {
    const w = flat(W);

    let sum = 0;
    const n = Math.min(w.length, X.length);

    for (let i = 0; i < n; i++) {
      sum += w[i] * X[i];
    }

    return sum;
  };

const z = W_f.map((row, i) => {
  return dot(row, hx) + (b_f?.[i] ?? 0);
});
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const f = z.map(sigmoid);

  // =====================
  // DISPLAY
  // =====================
  return `
    <div class="lstm-step">

      <h3>❌ Forget Gate (fₜ)</h3>
       <div class="code">
          <b>Công thức:</b><br>
          ${formula}
        </div>

        <div class="code">
          <b>Các bước tính:</b><br>
          ${stepByStep}
        </div>


        </div>
      <div class="code">
        <b>xₜ:</b> ${vec(x)}<br>
        <b>hₜ₋₁:</b> ${displayVec(h, 5, 3)}
      </div>

      <div class="code">
        <b>b_f:</b>
        ${vec(b_f)}
      </div>

      <div class="code">
        <b>W_f:</b>
        ${vec(flat(W_f).slice(0, 20))}
      </div>

      <div class="code">
      step 1:<br>
      z =  ${vec(flat(W_f).slice(0, 20))}<br> · [${displayVec(h, 5, 3)} , ${vec(x)}] <br>+ ${vec(b_f)}<br>
      z = ${vec(z)}<br><br>
      
      <br>step 2: <br> fₜ = [${computeForgetGate(W_f, b_f, h_prev, inputVector).display.map(x => x.toFixed(3)).join(", ")}...]
      </div>
      <p>
        - fₜ gần 0 → quên mạnh<br>
        - fₜ gần 1 → giữ memory
      </p>

    </div>
  `;
}

function computeInputGate(W_i, b_i, x_t) {

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  const toNumber = (value) => {
    const num = Number(value);
    return Number.isNaN(num) ? 0 : num;
  };

  const x = Array.isArray(x_t) ? x_t.slice(0, 10).map(toNumber) : [];
  const W = Array.isArray(W_i) && Array.isArray(W_i[0]) ? W_i : [];
  const b_demo = Array.isArray(b_i)
    ? b_i.slice(0, 5).map(toNumber)
    : [toNumber(b_i)];

  const demoDim = Math.min(5, x.length, W.length, b_demo.length);

  if (W.length === 0 || x.length === 0 || demoDim === 0) {
    return { z_demo: [], i_demo: [], x_demo: x, w_demo: [], b_demo: [] };
  }

  const w_demo = W.slice(0, 5).map((row) =>
    Array.isArray(row)
      ? row.slice(0, 10).map(toNumber)
      : []
  );

  const z_demo = new Array(demoDim);

  for (let i = 0; i < demoDim; i++) {
    const row = Array.isArray(W[i]) ? W[i] : [];
    let z = b_demo[i] ?? 0;

    for (let j = 0; j < Math.min(10, row.length, x.length); j++) {
      z += toNumber(row[j]) * x[j];
    }

    z_demo[i] = z;
  }

  return {
    z_demo,
    i_demo: z_demo.map(sigmoid),
    x_demo: x.slice(0, Math.min(10, x.length)),
    w_demo,
    b_demo: b_demo.slice(0, demoDim)
  };
}

function renderStep3(inputVector, h_prev, W_i, b_i, t = 0) {

  // ===== FORMAT SAFE =====
  const vec = (v, limit = 6) => {
    if (!Array.isArray(v)) return "[]";

    const body = v
      .slice(0, limit)
      .map(x => Number(x).toFixed(3))
      .join(", ");

    return v.length > limit ? `[${body}, ...]` : `[${body}...]`;
  };

const matrix = (v, rowLimit = 5, colLimit = 5) => {
  if (!Array.isArray(v)) return "[]";

  const indent = "&nbsp;&nbsp;&nbsp;&nbsp;"; // 4 spaces

  const rows = [];
  for (let r = 0; r < rowLimit; r++) {
    const row = Array.isArray(v[r]) ? v[r] : [];
    const cells = [];

    for (let c = 0; c < colLimit; c++) {
      const val = row[c];
      const n = Number(val);
      cells.push(Number.isNaN(n) ? "0.000" : n.toFixed(3));
    }

    rows.push(
      `${indent}[${cells.join(", ")}${
        Array.isArray(v[r]) && v[r].length > colLimit ? ", ..." : "..."
      }]`
    );
  }

  const moreRows = v.length > rowLimit ? `,<br>${indent}...` : "";

  return `[<br>
${rows.join(",<br>")}${moreRows}
<br>&nbsp;&nbsp;&nbsp...]`;
};

  const h = Array.isArray(h_prev) ? h_prev : [];
  const x = Array.isArray(inputVector) ? inputVector : [];

  // ===== COMPUTE =====
  const result = computeInputGate(W_i, b_i, x);

  // ===== DISPLAY =====
  const x_display = vec(result.x_demo, 10);
  const w_display = matrix(result.w_demo, 5, 10);
  const b_display = vec(result.b_demo);
  const z_display = vec(result.z_demo);
  const i_display = vec(result.i_demo);
  const formula = `
    zₜ = W_i · [hₜ₋₁, xₜ] + b_i
    <br>
    iₜ = &sigma;(zₜ)
  `;
  const stepByStep = `
    Step 1: zₜ = W_i · [hₜ₋₁, xₜ] + b_i <br>
    Step 2: iₜ = &sigma;(zₜ)
  `;

  return `
    <div class="lstm-step">

      <h3>➕ Input Gate</h3>

      <div class="code">
        <b>Công thức:</b><br>
        ${formula}
      </div>

      <div class="code">
        <b>Các bước tính:</b><br>
        ${stepByStep}
      </div>

      <p>
        Với <b>W_i</b> cắt lấy ma trận 5x10 góc trên - trái.
      </p>

      <div class="code">
      Step 1:<br>  
      <b>xₜ:</b> ${x_display}<br>
        <b>W_i:</b> ${w_display}<br>
        <b>b_i:</b> ${b_display} <br> <br>
        zₜ = ${w_display} <br>·[${displayVec(h, 5, 3)}, ${x_display}] <br>+ ${b_display}<br>
        zₜ = ${z_display}
      </div>

      <div class="code">
        Step 2:<br>
        <b>iₜ = &sigma;(zₜ):</b> ${i_display}
      </div>

      <p>
        - iₜ gần 0 → bỏ qua input<br>
        - iₜ gần 1 → ghi mạnh vào bộ nhớ
      </p>

    </div>
  `;
}

function computeCandidateMemory(W_c, b_c, h_prev, x_t) {

  const tanh = (x) => Math.tanh(x);

  // ===== SAFE INPUT =====
  const h = Array.isArray(h_prev) ? h_prev : [];
  const x = Array.isArray(x_t) ? x_t : [];
  const input = [...h, ...x];

  // ===== VALIDATE WEIGHT =====
  const W = Array.isArray(W_c) && Array.isArray(W_c[0]) ? W_c : [];

  if (W.length === 0 || input.length === 0) {
    return { c_tilde: [], z_list: [] };
  }

  const hiddenDim = W.length;

  const c_tilde = new Array(hiddenDim);
  const z_list = new Array(hiddenDim);

  for (let i = 0; i < hiddenDim; i++) {

    const Wi = W[i] || [];

    let z = 0;

    for (let j = 0; j < input.length; j++) {
      const w = Wi[j] ?? 0;
      const xj = input[j] ?? 0;
      z += w * xj;
    }

    const b = Array.isArray(b_c)
      ? (b_c[i] ?? 0)
      : (b_c ?? 0);

    z += b;

    z_list[i] = z;
    c_tilde[i] = tanh(z);
  }

  return { c_tilde, z_list };
}

function renderStep4(inputVector, h_prev, W_c, b_c, t = 0) {

  const isFirstStep = t === 0;

  // =====================
  // FORMAT
  // =====================
  const vec = (v, limit = 6) => {
    if (!Array.isArray(v)) return "[]";

    const body = v
      .slice(0, limit)
      .map(x => Number(x).toFixed(3))
      .join(", ");

    return v.length > limit ? `[${body}, ...]` : `[${body}...]`;
  };

  const matrix = (v, rowLimit = 5, colLimit = 10) => {
    if (!Array.isArray(v)) return "[]";

    const rows = [];
    for (let r = 0; r < rowLimit; r++) {
      const row = Array.isArray(v[r]) ? v[r] : [];
      const cells = [];
      for (let c = 0; c < colLimit; c++) {
        const n = Number(row[c]);
        cells.push(Number.isNaN(n) ? "0.000" : n.toFixed(3));
      }
      rows.push(`[${cells.join(", ")}...]`);
    }

    return `[<br>&nbsp;&nbsp;${rows.join(",<br>&nbsp;&nbsp;")}<br>&nbsp;&nbsp;...]`;
  };

  // =====================
  // SAFE STATE
  // =====================
  const h = Array.isArray(h_prev)
    ? (isFirstStep ? new Array(h_prev.length).fill(0) : h_prev)
    : [];

  const x = Array.isArray(inputVector) ? inputVector : [];

  // =====================
  // COMPUTE
  // =====================
  let result = { c_tilde: [], z_list: [] };

  try {
    result = computeCandidateMemory(W_c, b_c, h, x);
  } catch (e) {
    console.error("CandidateMemory error:", e);
  }

  // =====================
  // DISPLAY SAFE
  // =====================
  const c_display = vec(result.c_tilde);
  const z_display = vec(result.z_list);
  const formula = `
    zₜ = W_c · [hₜ₋₁, xₜ] + b_c <br>
    C̃ₜ = tanh(zₜ)
  `;
  const stepByStep = `
    Step 1: ghép hₜ₋₁ với xₜ, zₜ = W_c · [hₜ₋₁, xₜ] + b_c<br>
    Step 2: C̃ₜ = tanh(zₜ)<br>
  `;

  return `
    <div class="lstm-step">

      <h3>Candidate Memory (C̃ₜ)</h3>

      <div class="code">
        <b>Công thức:</b><br>
        ${formula}
      </div>

      <div class="code">
        <b>Các bước tính:</b><br>
        ${stepByStep}
      </div>

      <div class="code">
        <b>xₜ:</b> ${vec(x)}<br>
        <b>hₜ₋₁:</b> ${displayVec(h, 5, 3)}<br>
        <b>W_c (5x10):</b> ${matrix(W_c, 5, 10)}<br>
        <b>b_c:</b> ${vec(b_c, 5)}
      </div>

      <div class="code">
        Step 1: <br>
        zₜ = ${matrix(W_c, 5, 10)}<br> · [${displayVec(h, 5, 3)}, ${vec(x)}] <br>+ ${vec(result.z_list)}<br>
        <b>zₜ = </b>
        ${z_display}
      </div>

      <div class="code">
        Step 2: <br>
        <b>C̃ₜ = tanh(${z_display})</b><br>
        C̃ₜ = ${c_display}
      </div>

      <p>
        &rarr; C̃ₜ là thông tin mới được tạo ra từ input hiện tại<br>
        &rarr; Sẽ được lọc bởi Input Gate trước khi ghi vào memory
      </p>

    </div>
  `;
}
function computeCellState(f_t, C_prev, i_t, c_tilde) {

  const safeArray = (arr, len) => {
    if (!Array.isArray(arr)) return new Array(len).fill(0);
    return arr;
  };

  const dim = Math.max(
    f_t?.length || 0,
    C_prev?.length || 0,
    i_t?.length || 0,
    c_tilde?.length || 0
  );

  const f = safeArray(f_t, dim);
  const C_old = safeArray(C_prev, dim);
  const i = safeArray(i_t, dim);
  const c_new = safeArray(c_tilde, dim);

  const C_t = new Array(dim);
  const forget_part = new Array(dim);
  const input_part = new Array(dim);

  for (let k = 0; k < dim; k++) {

    const f_val = f[k] ?? 0;
    const C_val = C_old[k] ?? 0;
    const i_val = i[k] ?? 0;
    const c_val = c_new[k] ?? 0;

    forget_part[k] = f_val * C_val;
    input_part[k] = i_val * c_val;

    C_t[k] = forget_part[k] + input_part[k];
  }

  return { C_t, forget_part, input_part };
}
function renderStep5(f_t, C_prev, i_t, c_tilde, t = 0) {

  const isFirstStep = t === 0;

  // ===== FORMAT =====
  const vec = (v, limit = 6) => {
    if (!Array.isArray(v)) return "[]";

    const body = v
      .slice(0, limit)
      .map(x => Number(x).toFixed(3))
      .join(", ");

    return v.length > limit ? `[${body}, ...]` : `[${body}...]`;
  };

  // ===== SAFE STATE =====
  const C_old = Array.isArray(C_prev)
    ? (isFirstStep ? new Array(C_prev.length).fill(0) : C_prev)
    : [];

  // ===== COMPUTE =====
  let result = {
    C_t: [],
    forget_part: [],
    input_part: []
  };

  try {
    result = computeCellState(f_t, C_old, i_t, c_tilde);
  } catch (e) {
    console.error("CellState error:", e);
  }

  const formula = `
    Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
  `;
  const stepByStep = `
    Step 1: Lấy phần nhớ cũ: fₜ ⊙ Cₜ₋₁<br>
    Step 2: Lấy phần thông tin mới: iₜ ⊙ C̃ₜ<br>
  `;

  return `
    <div class="lstm-step">

      <h3>Cell State Update (Cₜ)</h3>

      <div class="code">
        <b>Công thức:</b><br>
        ${formula}
      </div>

      <div class="code">
        <b>Tách bước tính:</b><br>
        ${stepByStep}
      </div>

      <div class="code">
        <b>Cₜ₋₁:</b> ${displayVec(C_old, 5, 3)}<br>
        <b>fₜ:</b> ${vec(f_t)}<br>
        <b>iₜ:</b> ${vec(i_t)}<br>
        <b>C̃ₜ:</b> ${vec(c_tilde)}
      </div>

      <div class="code">
        <b>Step 1 (fₜ ⊙ Cₜ₋₁):</b><br>
        ${displayVec(C_old, 5, 3)} ⊙ ${vec(f_t)}
        <br>
        = ${vec(result.forget_part)}
      </div>

      <div class="code">
        <b>Step 2 (iₜ ⊙ C̃ₜ):</b><br>
        ${vec(i_t)} ⊙ ${vec(c_tilde)}
        <br>
        = ${vec(result.input_part)}
      </div>

      <div class="code">
        <b>Cₜ = </b> ${vec(result.forget_part)} + ${vec(result.input_part)}<br>
        <b>Cₜ = </b> ${vec(result.C_t)}
      </div>

      <p>
        &rarr; Giữ lại trí nhớ quan trọng từ quá khứ<br>
        &rarr; Thêm thông tin mới có chọn lọc<br>
        &rarr; Đây là “bộ nhớ dài hạn” của LSTM
      </p>

    </div>
  `;
}
function computeOutputGate(W_o, b_o, h_prev, x_t, C_t) {

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const tanh = (x) => Math.tanh(x);

  // ===== INPUT =====
  const h = Array.isArray(h_prev) ? h_prev : [];
  const x = Array.isArray(x_t) ? x_t : [];
  const input = [...h, ...x];

  const W = Array.isArray(W_o) && Array.isArray(W_o[0]) ? W_o : [];

  if (W.length === 0 || input.length === 0) {
    return { o_t: [], h_t: [], z_list: [] };
  }

  const hiddenDim = W.length;

  const o_t = new Array(hiddenDim);
  const h_t = new Array(hiddenDim);
  const z_list = new Array(hiddenDim);

  for (let i = 0; i < hiddenDim; i++) {

    const Wi = W[i] || [];

    let z = 0;

    for (let j = 0; j < input.length; j++) {
      z += (Wi[j] ?? 0) * (input[j] ?? 0);
    }

    const b = Array.isArray(b_o)
      ? (b_o[i] ?? 0)
      : (b_o ?? 0);

    z += b;

    const o = sigmoid(z);

    const c_val = C_t?.[i] ?? 0;
    const h_val = o * tanh(c_val);

    z_list[i] = z;
    o_t[i] = o;
    h_t[i] = h_val;
  }

  return { o_t, h_t, z_list };
}

function renderStep6(inputVector, h_prev, C_t, W_o, b_o, t = 0) {

  const isFirstStep = t === 0;

  // ===== FORMAT =====
  const vec = (v, limit = 6) => {
    if (!Array.isArray(v)) return "[]";

    const body = v
      .slice(0, limit)
      .map(x => Number(x).toFixed(3))
      .join(", ");

    return v.length > limit ? `[${body}, ...]` : `[${body}...]`;
  };

  // ===== STATE =====
  const h = Array.isArray(h_prev)
    ? (isFirstStep ? new Array(h_prev.length).fill(0) : h_prev)
    : [];

  const x = Array.isArray(inputVector) ? inputVector : [];

  let result = { o_t: [], h_t: [], z_list: [] };

  try {
    result = computeOutputGate(W_o, b_o, h, x, C_t);
  } catch (e) {
    console.error("OutputGate error:", e);
  }

  const formula = `
    oₜ = &sigma;(W_o · [hₜ₋₁, xₜ] + b_o) <br>
    hₜ = oₜ ⊙ tanh(Cₜ)
  `;

  const stepByStep = `
    Step 1: zₜ = W_o · [hₜ₋₁, xₜ] + b_o<br>
    Step 2: oₜ = &sigma;(zₜ)<br>
    Step 3: hₜ = oₜ ⊙ tanh(Cₜ)
  `;

  return `
    <div class="lstm-step output-step">

      <h3>📤 Output Gate (hₜ)</h3>

      <div class="code">
        <b>Công thức:</b><br>
        ${formula}
      </div>

      <div class="code">
        <b>Các bước tính:</b><br>
        ${stepByStep}
      </div>

      <div class="code">
        <b>xₜ:</b> ${vec(x)}<br>
        <b>hₜ₋₁:</b> ${displayVec(h, 5, 3)}<br>
        <b>Cₜ:</b> ${vec(C_t)}<br>
        <b>W_o:</b> ${matrix(W_o, 5, 10)}<br>
        <b>b_o:</b> ${vec(b_o)}
      </div>

      <div class="code">
        <b>Step 1:</b><br>
        zₜ =  ${matrix(W_o, 5, 10)} <br> · [${displayVec(h, 5, 3)}, ${vec(x)}] <br>+ ${vec(b_o)}<br>
        zₜ = ${vec(result.z_list)}
      </div>

      <div class="code">
        <b>Step 2:</b><br>
        oₜ = &sigma;(zₜ)<br>
        oₜ = ${vec(result.o_t)}
      </div>

      <div class="code">
        <b>Step 3:</b><br>
        hₜ = ${vec(result.o_t)} ⊙ tanh(${vec(C_t)})<br>
        hₜ = ${vec(result.h_t)}
      </div>

      <p>
        &rarr; oₜ chọn phần nào của memory được xuất ra<br>
        &rarr; hₜ là output cuối cùng của LSTM tại thời điểm t
      </p>

    </div>
  `;
}