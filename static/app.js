const form = document.getElementById("search-form");
const input = document.getElementById("search-input");
const btn = document.getElementById("search-btn");
const loading = document.getElementById("loading");
const resultsEl = document.getElementById("results");
const errorEl = document.getElementById("error");
const intentBadge = document.getElementById("intent-badge");
const resultsCount = document.getElementById("results-count");
const summary = document.getElementById("summary");
const instructorGroups = document.getElementById("instructor-groups");

function hide(...els) {
    els.forEach((el) => el.classList.add("hidden"));
}

function show(...els) {
    els.forEach((el) => el.classList.remove("hidden"));
}

function scoreClass(score) {
    if (score >= 8) return "score-high";
    if (score >= 5) return "score-mid";
    return "score-low";
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function renderResults(data) {
    intentBadge.textContent = data.intent;
    resultsCount.textContent = `${data.total_results} clips from ${data.instructor_count} instructors`;
    summary.textContent = data.summary;

    instructorGroups.innerHTML = "";

    const instructors = Object.entries(data.results);
    instructors.forEach(([name, clips], idx) => {
        const group = document.createElement("div");
        group.className = "instructor-group" + (idx === 0 ? " open" : "");

        const header = document.createElement("div");
        header.className = "instructor-header";
        header.innerHTML = `
            <div>
                <span class="instructor-name">${escapeHtml(name)}</span>
                <span class="clip-count">${clips.length} clip${clips.length !== 1 ? "s" : ""}</span>
            </div>
            <span class="chevron">&#9660;</span>
        `;
        header.addEventListener("click", () => group.classList.toggle("open"));

        const body = document.createElement("div");
        body.className = "instructor-body";

        clips.forEach((clip) => {
            const row = document.createElement("div");
            row.className = "result-row";
            row.innerHTML = `
                <div class="result-top">
                    <span class="video-title">${escapeHtml(clip.video_title)}</span>
                    <span class="timestamp">${escapeHtml(clip.timestamp)} - ${escapeHtml(clip.end_time)}</span>
                    <span class="score-badge ${scoreClass(clip.relevance_score)}">${clip.relevance_score}/10</span>
                </div>
                <div class="excerpt">${escapeHtml(clip.text)}</div>
            `;
            body.appendChild(row);
        });

        group.appendChild(header);
        group.appendChild(body);
        instructorGroups.appendChild(group);
    });

    show(resultsEl);
}

async function doSearch(query) {
    hide(resultsEl, errorEl);
    show(loading);
    btn.disabled = true;

    try {
        const res = await fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
        });

        const data = await res.json();

        hide(loading);
        btn.disabled = false;

        if (data.error) {
            errorEl.textContent = data.error;
            show(errorEl);
            return;
        }

        renderResults(data);
    } catch (err) {
        hide(loading);
        btn.disabled = false;
        errorEl.textContent = "Network error. Is the server running?";
        show(errorEl);
    }
}

form.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (query.length < 3) return;
    doSearch(query);
});

document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
        const query = chip.dataset.query;
        input.value = query;
        doSearch(query);
    });
});
