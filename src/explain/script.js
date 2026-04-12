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
// RENDER TOKENS + VECTOR
// =====================
function renderVectors(tokens, vectors) {
  return tokens.map((t, i) => `
    <div class="vector-row">
      <b>${t}</b> → [${
        vectors[i]
          .map(v => Number(v.toFixed(3)))  
          .join(", ")
      }]
    </div>
  `).join("");
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

  const labelClass = data.label === "spam" ? "spam" : "ham";

  div.innerHTML = `
    <div class="card">
      <h2>📌 Prediction</h2>
      <p><b>Probability:</b> ${data.prob}</p>
      <p><b>Label:</b> <span class="${labelClass}">${data.label.toUpperCase()}</span></p>

      <h3>🔤 Tokens</h3>
      <div class="code">${data.tokens.join(", ")}</div>

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

      <div class="code">
        $$C_t = f_t \\cdot C_{t-1} + i_t \\cdot \\tilde{C}_t$$
        <br>
        $$h_t = o_t \\cdot \\tanh(C_t)$$
      </div>
    </div>
  `;

  // render lại MathJax
  if (window.MathJax) {
    MathJax.typeset();
  }
}