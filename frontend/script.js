const API_BASE = "http://localhost:8000/api";
let currentEventSource = null;
let activeQueryId = null;
let currentQueryText = "";

// Initialize Icons
lucide.createIcons();
loadHistory();

// --- 1. HISTORY MANAGEMENT ---
async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/queries?limit=20`);
    const data = await res.json();
    renderHistoryList(data.queries.reverse());
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

function renderHistoryList(queries) {
  const list = document.getElementById("history-list");
  list.innerHTML = "";

  queries.forEach((q) => {
    const div = document.createElement("div");
    div.className = `history-item ${q.query_id === activeQueryId ? "active" : ""}`;
    div.onclick = () => loadQuery(q.query_id);

    const time = new Date(q.started_at).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    const statusClass = `status-${q.status}`;

    div.innerHTML = `
              <div class="history-meta">
                  <span>${time}</span>
                  <span class="status-badge ${statusClass}">${q.status}</span>
              </div>
              <div class="history-query">${q.query}</div>
          `;
    list.appendChild(div);
  });
}

// --- 2. QUERY SUBMISSION ---
async function submitQuery() {
  const input = document.getElementById("query-input");
  const btn = document.getElementById("submit-btn");
  const query = input.value.trim();

  if (!query) return;

  currentQueryText = query;
  input.value = "";
  btn.disabled = true;

  // Prepare UI for new query — keep previous logs visible until stream starts
  document.getElementById("welcome-message").style.display = "none";
  document.getElementById("content-container").style.display = "block";
  document.getElementById("markdown-output").innerHTML =
    '<span style="color:#64748b">Processing query...</span>';

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    activeQueryId = data.query_id;

    // Start listening to the stream
    connectToStream(activeQueryId);

    // Refresh history list to show the new "pending" item
    loadHistory();
  } catch (err) {
    console.error(err);
    logLine("System Error: Failed to submit query", "error");
    btn.disabled = false;
  }
}

// --- 3. LOADING EXISTING QUERY ---
async function loadQuery(id) {
  if (currentEventSource) currentEventSource.close();
  activeQueryId = id;

  // Set up the layout without wiping existing content
  document.getElementById("welcome-message").style.display = "none";
  document.getElementById("content-container").style.display = "block";

  // Highlight in sidebar
  loadHistory();

  try {
    // Fetch full details
    const res = await fetch(`${API_BASE}/query/${id}`);
    const data = await res.json();

    // Store query text for log filtering
    currentQueryText = data.query || "";

    // If completed, render static data
    if (data.status === "completed" || data.status === "failed") {
      renderResult(data.result);

      // Clear and re-render artifacts for this query
      document.getElementById("artifacts-grid").innerHTML = "";
      document.getElementById("exports-section").style.display = "none";
      renderExports(data.new_exports || []);

      // Load logs for this query
      document.getElementById("logs").innerHTML = "";
      const logRes = await fetch(`${API_BASE}/query/${id}/logs`);
      const logData = await logRes.json();
      logData.logs.forEach((l) => logLine(l));

      if (data.status === "completed")
        logLine("--- Execution Completed ---", "success");
      if (data.status === "failed")
        logLine("--- Execution Failed ---", "error");
    } else {
      // It's still running — show pending state and stream
      document.getElementById("markdown-output").innerHTML =
        '<span style="color:#64748b">Waiting for result...</span>';
      connectToStream(id);
    }
  } catch (err) {
    console.error(err);
  }
}

// --- 4. STREAMING LOGIC (SSE) ---
function connectToStream(id) {
  if (currentEventSource) currentEventSource.close();

  // Clear logs and artifacts for the new stream
  document.getElementById("logs").innerHTML = "";
  document.getElementById("artifacts-grid").innerHTML = "";
  document.getElementById("exports-section").style.display = "none";

  logLine(`--- Stream connected ---`, "info");
  const es = new EventSource(`${API_BASE}/query/${id}/stream`);
  currentEventSource = es;

  es.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    const { type, data } = payload;

    switch (type) {
      case "log":
        // Filter out lines that are just the user's prompt/query
        if (currentQueryText && typeof data === "string") {
          const trimmed = data.trim().toLowerCase();
          const queryLower = currentQueryText.trim().toLowerCase();
          if (trimmed === queryLower) break;
          if (
            trimmed.includes(queryLower) &&
            trimmed.length < queryLower.length + 40
          )
            break;
        }
        logLine(data);
        break;
      case "result":
        renderResult(data);
        break;
      case "exports":
        renderExports(data);
        break;
      case "status":
        if (data === "completed")
          logLine("--- Execution Completed ---", "success");
        if (data === "failed") logLine("--- Execution Failed ---", "error");
        break;
      case "done":
        es.close();
        document.getElementById("submit-btn").disabled = false;
        loadHistory(); // Update status icons in sidebar
        // Results, logs, and artifacts remain visible — no reset
        break;
      case "error":
        logLine(`Error: ${data}`, "error");
        break;
    }
  };

  es.onerror = (err) => {
    console.error("SSE Error", err);
    es.close();
    document.getElementById("submit-btn").disabled = false;
  };
}

// --- 5. RENDER HELPERS ---
function resetUI() {
  document.getElementById("welcome-message").style.display = "none";
  document.getElementById("content-container").style.display = "block";
  document.getElementById("markdown-output").innerHTML =
    '<span style="color:#64748b">Waiting for result...</span>';
  document.getElementById("logs").innerHTML = "";
  document.getElementById("artifacts-grid").innerHTML = "";
  document.getElementById("exports-section").style.display = "none";
}

function logLine(text, type = "normal") {
  const container = document.getElementById("logs");
  const div = document.createElement("div");
  div.className = `log-line log-${type}`;
  div.textContent = `> ${text}`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function renderResult(markdown) {
  if (!markdown) return;
  document.getElementById("markdown-output").innerHTML = marked.parse(markdown);
}

function renderExports(files) {
  if (!files || files.length === 0) return;

  const container = document.getElementById("artifacts-grid");
  document.getElementById("exports-section").style.display = "block";

  files.forEach((file) => {
    if (document.getElementById(`artifact-${file.filename}`)) return;

    const card = document.createElement("div");
    card.className = "artifact-card";
    card.id = `artifact-${file.filename}`;

    const isImage = ["png", "jpg", "jpeg", "svg"].includes(file.type);
    const isGeoVector = ["geojson", "json", "kml"].includes(file.type);
    const isTextData = ["csv", "txt"].includes(file.type);
    const isRaster = ["tif", "tiff"].includes(file.type);

    let previewHtml = "";

    if (isImage) {
      fetch(`${API_BASE}/exports/${file.filename}/preview`)
        .then((r) => r.json())
        .then((d) => {
          const img = card.querySelector("img");
          if (img) {
            img.src = d.data_uri;
            img.style.opacity = "1";
          }
        })
        .catch(() => {});
      previewHtml = `<div class="artifact-preview" style="min-height:150px; background:#000; display:flex; align-items:center; justify-content:center;">
                       <img alt="Loading preview..." style="opacity:0.3; max-width:100%;">
                     </div>`;
    } else if (isGeoVector) {
      const mapId = `map-${file.filename.replace(/[^a-zA-Z0-9]/g, "_")}`;
      previewHtml = `<div class="artifact-preview">
                       <div id="${mapId}" style="height:350px; width:100%; background:#1a1a2e;"></div>
                       <div class="geo-toggle" style="text-align:center; padding:0.4rem; border-top:1px solid var(--border);">
                         <button onclick="toggleGeoView(this, '${mapId}')" style="background:none; border:1px solid var(--border); color:var(--text-secondary); border-radius:4px; padding:0.25rem 0.75rem; cursor:pointer; font-size:0.7rem;">Show Raw JSON</button>
                       </div>
                       <pre class="geo-preview" style="display:none; max-height:300px; overflow:auto; padding:1rem; font-size:0.7rem; color:#10b981;"></pre>
                     </div>`;

      // Fetch GeoJSON and render on Leaflet map
      fetch(`${API_BASE}/exports/${file.filename}/preview`)
        .then((r) => r.json())
        .then((d) => {
          if (!d.data) return;
          const geojsonData =
            typeof d.data === "object" ? d.data : JSON.parse(d.data);

          // Fill the raw JSON fallback
          const pre = card.querySelector(".geo-preview");
          if (pre)
            pre.textContent = JSON.stringify(geojsonData, null, 2).substring(
              0,
              5000,
            );

          // Initialize Leaflet map
          const mapEl = card.querySelector(`#${mapId}`);
          if (!mapEl) return;

          const map = L.map(mapEl, {
            attributionControl: false,
            zoomControl: true,
          });
          L.tileLayer(
            "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            {
              maxZoom: 19,
              attribution: "",
            },
          ).addTo(map);

          const geoLayer = L.geoJSON(geojsonData, {
            style: function () {
              return {
                color: "#6366f1",
                weight: 2,
                fillColor: "#818cf8",
                fillOpacity: 0.35,
              };
            },
            pointToLayer: function (feature, latlng) {
              return L.circleMarker(latlng, {
                radius: 6,
                fillColor: "#6366f1",
                color: "#fff",
                weight: 1,
                fillOpacity: 0.8,
              });
            },
            onEachFeature: function (feature, layer) {
              if (feature.properties) {
                let popup =
                  '<div style="max-height:200px; overflow:auto; font-size:0.75rem;">';
                for (const [k, v] of Object.entries(feature.properties)) {
                  popup += `<b>${k}:</b> ${v}<br>`;
                }
                popup += "</div>";
                layer.bindPopup(popup);
              }
            },
          }).addTo(map);

          // Fit map to data bounds
          try {
            const bounds = geoLayer.getBounds();
            if (bounds.isValid()) {
              map.fitBounds(bounds, { padding: [20, 20] });
            } else {
              map.setView([20, 78], 5);
            }
          } catch (e) {
            map.setView([20, 78], 5);
          }

          // Store map reference for resize
          mapEl._leafletMap = map;
          setTimeout(() => map.invalidateSize(), 200);
        })
        .catch((err) => {
          console.error("GeoJSON map error:", err);
          const mapEl = card.querySelector(`#${mapId}`);
          if (mapEl)
            mapEl.innerHTML =
              '<div style="padding:1rem; color:var(--danger);">Failed to load map preview</div>';
        });
    } else if (isTextData) {
      fetch(`${API_BASE}/exports/${file.filename}/preview`)
        .then((r) => r.json())
        .then((d) => {
          const pre = card.querySelector(".text-preview");
          if (pre && d.data) {
            pre.textContent = String(d.data).substring(0, 3000);
            pre.style.color = "#e2e8f0";
          }
        })
        .catch(() => {});
      previewHtml = `<div class="artifact-preview">
                       <pre class="text-preview" style="max-height:300px; overflow:auto; padding:1rem; font-size:0.7rem; color:#94a3b8;">Loading preview...</pre>
                     </div>`;
    } else if (isRaster) {
      previewHtml = `<div class="artifact-preview" style="padding:1rem; text-align:center;">
                       <div style="font-size:2rem; margin-bottom:0.5rem;">🗺️</div>
                       <div style="color:var(--text-secondary); font-size:0.8rem;">Raster: ${file.filename}</div>
                       <div style="color:var(--text-secondary); font-size:0.75rem;">Size: ${(file.size_bytes / 1024).toFixed(1)} KB</div>
                       <a href="${API_BASE}/exports/${file.filename}" target="_blank" style="color:var(--accent); font-size:0.8rem; text-decoration:none; margin-top:0.5rem; display:inline-block;">Download to view</a>
                     </div>`;
    } else {
      previewHtml = `<div class="artifact-preview">
                       <pre style="padding:1rem;">📄 ${file.filename}\nSize: ${(file.size_bytes / 1024).toFixed(1)} KB</pre>
                     </div>`;
    }

    card.innerHTML = `
              <div class="artifact-header">
                  <span>${file.filename}</span>
                  <a href="${API_BASE}/exports/${file.filename}" target="_blank" style="color:var(--accent); text-decoration:none;">
                      <i data-lucide="download" style="width:14px; height:14px;"></i>
                  </a>
              </div>
              ${previewHtml}
          `;
    container.appendChild(card);
    lucide.createIcons();
  });
}

// --- 6. TOGGLE MAP / RAW JSON VIEW ---
function toggleGeoView(btn, mapId) {
  const card = btn.closest(".artifact-card");
  const mapEl = card.querySelector(`#${mapId}`);
  const pre = card.querySelector(".geo-preview");
  if (!mapEl || !pre) return;

  if (pre.style.display === "none") {
    // Show raw JSON, hide map
    pre.style.display = "block";
    mapEl.style.display = "none";
    btn.textContent = "Show Map";
  } else {
    // Show map, hide raw JSON
    pre.style.display = "none";
    mapEl.style.display = "block";
    btn.textContent = "Show Raw JSON";
    // Refresh map tiles after re-display
    if (mapEl._leafletMap) {
      setTimeout(() => mapEl._leafletMap.invalidateSize(), 100);
    }
  }
}
