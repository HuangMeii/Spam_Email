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

  const hState = isFirstStep ? "0 (initial state)" : formatVector(h_prev);
  const cState = isFirstStep ? "0 (initial state)" : formatVector(c_prev);

  return `
    <div class="lstm-step">
      <h3>📍 Bước ${t + 1}: Nhận input tại thời điểm t = ${t}</h3>

      <p>
        <b>Ý nghĩa:</b> xₜ là input mới,
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
        👉 Ở <b>t = 0</b>, LSTM chưa có memory nên khởi tạo toàn bộ bằng 0.
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
    data.vectors?.[0] || [],
    data.hidden_prev || [],
    data.cell_prev || [],
    data.forget_gate || 0.8,
    0
    );
  div.innerHTML = `
    <div class="card">
      <h2>📌 Prediction</h2>
      <p><b>Tokens:</b> ${tokenCount}</p>
      <p><b>Probability:</b> ${data.prob}</p>
      <p><b>Label:</b> <span class="${labelClass}">${data.label.toUpperCase()}</span></p>

      <h3>🔤 Tokens</h3>
      <div class="code">${(data.tokens || []).join(", ")}</div>

      <h3>📊 Word2Vec (5 dims)</h3>
      ${renderVectors(data.tokens, data.vectors)}

      <h3>🔄 LSTM Flow</h3>
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

      <div class="code">
        $$C_t = f_t \\cdot C_{t-1} + i_t \\cdot \\tilde{C}_t$$
        <br>
        $$h_t = o_t \\cdot \\tanh(C_t)$$
      </div>

      ${step1}
      ${step2}
    </div>
  `;

  if (window.MathJax) {
    MathJax.typeset();
  }
}
function renderStep2(inputVector, h_prev, c_prev, forgetGate, t = 0) {

  const isFirstStep = t === 0;

  // =====================
  // SAFE FORMAT FUNCTION
  // =====================
    const fmt = (v) => {
    if (!Array.isArray(v)) return "[]";

    return v
        .slice(0, 6)
        .map(x => {
        const num = Number(x);
        return isNaN(num) ? "0.000" : num.toFixed(3);
        })
        .join(", ") + ", ...";
    };

  // =====================
  // STATE HANDLING
  // =====================
  const hState = isFirstStep ? "0 (initial hidden state)" : fmt(h_prev);
  const cState = isFirstStep ? "0 (initial cell state)" : fmt(c_prev);

  // forget gate normalize
  const fState = (() => {
    if (Array.isArray(forgetGate)) return fmt(forgetGate);
    if (forgetGate !== undefined && forgetGate !== null)
      return Number(forgetGate).toFixed(3);
    return "0.000";
  })();

  return `
    <div class="lstm-step">

      <h3>❌ Bước 2: Forget Gate (fₜ) – quên gì?</h3>

      <p>
        <b>Ý nghĩa:</b> Forget Gate quyết định thông tin nào trong <b>Cₜ₋₁</b> được giữ lại hoặc loại bỏ.
      </p>

      <div class="code">
        <b>Công thức:</b><br>
        fₜ = σ(W_f · [hₜ₋₁, xₜ] + b_f)
      </div>

      <div class="code">
        <b>xₜ (input):</b><br>
        [${fmt(inputVector)}]
      </div>

      <div class="code">
        <b>hₜ₋₁ (hidden state):</b><br>
        [${hState}]
      </div>

      <div class="code">
        <b>Cₜ₋₁ (cell state):</b><br>
        [${cState}]
      </div>

      <div class="code">
        <b>fₜ (forget gate output):</b><br>
        ${fState}
      </div>

      <p>
        👉 <b>Giải thích:</b><br>
        • fₜ ≈ 0 → quên gần hết thông tin cũ<br>
        • fₜ ≈ 1 → giữ lại toàn bộ memory
      </p>

    </div>
  `;
}