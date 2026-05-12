async function loadData() {
  const res = await fetch("leaderboard_data.json", { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`failed to load leaderboard_data.json: ${res.status}`);
  }
  return res.json();
}

function num(v) {
  return typeof v === "number" ? v : Number.NEGATIVE_INFINITY;
}

function containsAny(haystack, needle) {
  if (!needle) return true;
  return haystack.toLowerCase().includes(needle.toLowerCase());
}

function renderTable(rows) {
  const tbody = document.querySelector("#leaderboardTable tbody");
  tbody.innerHTML = "";
  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="rank-chip">${idx + 1}</span></td>
      <td><div class="num-strong">${r.method}</div><div class="muted">${r.model_size || ""}</div></td>
      <td>${r.vlm}</td>
      <td>${r.vla}</td>
      <td class="num-strong">${r.overall_tsr.toFixed(1)}</td>
      <td class="num-strong">${r.overall_csr.toFixed(1)}</td>
      <td>${r.transfer_tsr.toFixed(1)} / ${r.transfer_csr.toFixed(1)}</td>
      <td>${r.occlusion_tsr.toFixed(1)} / ${r.occlusion_csr.toFixed(1)}</td>
      <td>${r.counting_tsr.toFixed(1)} / ${r.counting_csr.toFixed(1)}</td>
      <td>${r.sequence_tsr.toFixed(1)} / ${r.sequence_csr.toFixed(1)}</td>
      <td>${r.run_date}</td>
      <td class="muted">${r.notes}</td>
    `;
    tbody.appendChild(tr);
  });
}

function applyState(allRows) {
  const protocol = document.getElementById("protocolSelect").value;
  const metric = document.getElementById("sortSelect").value;
  const query = document.getElementById("searchInput").value.trim();

  let rows = allRows.filter((r) => protocol === "all" || r.protocol === protocol);
  rows = rows.filter((r) =>
    containsAny(
      [r.method, r.vlm, r.vla, r.notes, r.protocol].join(" "),
      query
    )
  );
  rows.sort((a, b) => num(b[metric]) - num(a[metric]));
  renderTable(rows);
}

function initProtocolOptions(rows) {
  const select = document.getElementById("protocolSelect");
  const protocols = [...new Set(rows.map((r) => r.protocol))].sort();
  select.innerHTML = "";
  const all = document.createElement("option");
  all.value = "all";
  all.textContent = "All protocols";
  select.appendChild(all);
  protocols.forEach((p) => {
    const op = document.createElement("option");
    op.value = p;
    op.textContent = p;
    select.appendChild(op);
  });
}

async function main() {
  try {
    const data = await loadData();
    const rows = data.entries || [];
    initProtocolOptions(rows);
    document.getElementById("updatedAt").textContent = `Updated: ${data.updated_at}`;

    ["protocolSelect", "sortSelect", "searchInput"].forEach((id) => {
      document.getElementById(id).addEventListener("input", () => applyState(rows));
    });
    applyState(rows);
  } catch (err) {
    const tbody = document.querySelector("#leaderboardTable tbody");
    tbody.innerHTML = `<tr><td colspan="12">Failed to load leaderboard data: ${err.message}</td></tr>`;
  }
}

main();
