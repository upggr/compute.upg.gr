/* Sourced shortlist UI. No ML, geometry computation, or Lean execution. */
(function () {
    'use strict';
    const catalog = JSON.parse(document.getElementById('handoff-data').textContent);
    const records = catalog.records;
    const byId = new Map(records.map(record => [record.record_id, record]));
    const selected = new Set();
    const el = id => document.getElementById(`handoff-${id}`);
    let activeId = null;
    let downloading = false;

    function visibleRecords() {
        const filtered = records.filter(record => el('filter').value !== 'small' || Math.abs(record.geometry.euler_char) < 100);
        const sort = el('sort').value;
        const key = record => {
            const g = record.geometry;
            if (sort === 'moduli') return g.h11 + g.h21;
            if (sort === 'vertices') return g.vertex_count;
            return Math.abs(g.euler_char);
        };
        return filtered.sort((a, b) => key(a) - key(b) || a.geometry.h11 - b.geometry.h11 ||
            a.geometry.h21 - b.geometry.h21 || a.record_id.localeCompare(b.record_id));
    }

    function selectionStatus() {
        const visible = new Set(visibleRecords().map(r => r.record_id));
        const hidden = [...selected].filter(id => !visible.has(id)).length;
        el('selection').textContent = `${selected.size} selected · ${visible.size} shown of ${records.length}` +
            (hidden ? ` · ${hidden} selected outside this view (will also be downloaded)` : '') +
            '. View # is not an ML rank.';
        el('download').disabled = !selected.size || downloading;
    }

    function inspect(record, focus = false) {
        activeId = record.record_id;
        const g = record.geometry;
        el('detail-title').textContent = `(h¹¹, h²¹) = (${g.h11}, ${g.h21}) · χ = ${g.euler_char}`;
        el('record-id').textContent = record.record_id;
        el('detail-summary').textContent = `${g.vertex_count} integer vertices in 4 columns. Hodge labels and reflexivity have not been recomputed here.`;
        el('checks').replaceChildren();
        for (const check of record.checks.items) {
            const li = document.createElement('li');
            li.textContent = `${check.passed ? 'PASS' : 'FAIL'} — ${check.label}`;
            li.dataset.outcome = check.passed ? 'pass' : 'fail';
            el('checks').append(li);
        }
        el('missing').replaceChildren();
        for (const missing of record.not_supplied) {
            const li = document.createElement('li');
            li.textContent = missing;
            el('missing').append(li);
        }
        el('provenance').replaceChildren();
        for (const [label, value] of [
            ['File / pointer', `${record.provenance.source_file} ${record.provenance.source_json_pointer}`],
            ['Source slice', record.provenance.source_slice],
            ['Reference', record.provenance.reference],
            ['Source-record SHA-256', record.provenance.source_record_sha256],
        ]) {
            const dt = document.createElement('dt');
            const dd = document.createElement('dd');
            dt.textContent = label;
            dd.textContent = value || 'Not supplied';
            el('provenance').append(dt, dd);
        }
        const context = el('hodge-link');
        context.hidden = !record.hodge_context_url;
        if (record.hodge_context_url) context.href = record.hodge_context_url;
        else context.removeAttribute('href');
        el('vertices').textContent = g.polytope_vertices.map(row => `[ ${row.join(', ')} ]`).join('\n');
        el('record-json').textContent = JSON.stringify(record, null, 2);
        for (const row of el('rows').children) row.dataset.active = String(row.dataset.id === activeId);
        if (focus) {
            const panel = document.querySelector('.handoff-inspector');
            panel.focus({ preventScroll: true });
            if (window.matchMedia('(max-width: 1000px)').matches) panel.scrollIntoView({ block: 'start' });
        }
    }

    function render() {
        const visible = visibleRecords();
        el('rows').replaceChildren();
        visible.forEach((record, index) => {
            const g = record.geometry;
            const row = document.createElement('tr');
            row.dataset.id = record.record_id;
            row.dataset.active = String(record.record_id === activeId);
            const selectCell = document.createElement('td');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = selected.has(record.record_id);
            checkbox.disabled = !record.checks.passed;
            checkbox.setAttribute('aria-label', `Select record with Hodge numbers ${g.h11}, ${g.h21}`);
            checkbox.addEventListener('change', () => {
                if (checkbox.checked) selected.add(record.record_id);
                else selected.delete(record.record_id);
                selectionStatus();
            });
            selectCell.append(checkbox);
            row.append(selectCell);
            for (const text of [index + 1, `(${g.h11}, ${g.h21})`, g.euler_char, g.vertex_count, record.checks.passed ? 'Pass' : 'Fail']) {
                const td = document.createElement('td');
                td.textContent = String(text);
                row.append(td);
            }
            const actionCell = document.createElement('td');
            const button = document.createElement('button');
            button.type = 'button';
            button.textContent = 'Inspect';
            button.setAttribute('aria-label', `Inspect record with Hodge numbers ${g.h11}, ${g.h21}`);
            button.addEventListener('click', () => inspect(record, true));
            actionCell.append(button);
            row.append(actionCell);
            el('rows').append(row);
        });
        if (visible.length && !visible.some(r => r.record_id === activeId)) inspect(visible[0]);
        selectionStatus();
    }

    el('filter').addEventListener('change', render);
    el('sort').addEventListener('change', render);
    el('select').addEventListener('click', () => {
        selected.clear();
        for (const record of visibleRecords().filter(r => r.checks.passed).slice(0, Number(el('top').value))) selected.add(record.record_id);
        render();
    });
    el('clear').addEventListener('click', () => { selected.clear(); render(); });

    el('download').addEventListener('click', async () => {
        // Preserve the displayed ordering, with any hidden selections afterwards.
        const visibleIds = visibleRecords().map(r => r.record_id);
        const ids = [...visibleIds.filter(id => selected.has(id)), ...[...selected].filter(id => !visibleIds.includes(id))];
        if (!ids.length || ids.some(id => !byId.has(id))) return;
        downloading = true;
        selectionStatus();
        el('download-status').textContent = 'Preparing the selected records and source snapshot…';
        try {
            const response = await fetch('/api/handoff/batch', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ record_ids: ids, source_sha256: catalog.source.sha256 }),
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.error || `Download failed (${response.status})`);
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'upg-ks-handoff.zip';
            document.body.append(link);
            link.click();
            link.remove();
            setTimeout(() => URL.revokeObjectURL(url), 30000);
            el('download-status').textContent = `Downloaded ${ids.length} selected record(s). ZIP SHA-256: ${response.headers.get('X-Content-SHA256')}.`;
        } catch (error) {
            el('download-status').textContent = error.message;
        } finally {
            downloading = false;
            selectionStatus();
        }
    });

    el('copy').addEventListener('click', async () => {
        const url = new URL('/handoff', window.location.origin).href;
        try {
            await navigator.clipboard.writeText(url);
            el('share-status').textContent = 'Page link copied. It opens the default view, not your current selection.';
        } catch (_) {
            el('share-status').textContent = `Copy this link: ${url}`;
        }
    });

    el('calibration-load').addEventListener('click', async () => {
        const button = el('calibration-load');
        button.disabled = true;
        el('calibration-status').textContent = 'Loading the reproducible Python check results…';
        try {
            const response = await fetch('/api/handoff/calibration');
            if (!response.ok) throw new Error(`Calibration request failed (${response.status})`);
            const data = await response.json();
            el('calibration-checks').replaceChildren();
            for (const check of data.report.positive.checks) {
                const li = document.createElement('li');
                li.textContent = `${check.passed ? 'PASS' : 'FAIL'} — ${check.statement}`;
                el('calibration-checks').append(li);
            }
            const rejected = data.report.negative_control_rejected_as_expected;
            el('calibration-status').textContent = `Python checks: ${data.report.positive.passed ? 'pass' : 'fail'}. Negative control: ${rejected ? 'rejected as expected' : 'unexpected result'}. Lean verification: NOT performed.`;
            el('calibration-results').hidden = false;
        } catch (error) {
            el('calibration-status').textContent = error.message;
        } finally {
            button.disabled = false;
        }
    });

    function checkEuler() {
        const value = el('euler').value.trim();
        const result = el('euler-result');
        if (!/^-?\d{1,20}$/.test(value)) {
            result.dataset.outcome = 'fail';
            result.textContent = 'Enter an integer (up to 20 digits), not a decimal or expression.';
            return;
        }
        const supplied = BigInt(value);
        const expected = 2n * (1n - 101n);
        const pass = supplied === expected;
        result.dataset.outcome = pass ? 'pass' : 'fail';
        result.textContent = `${pass ? 'PASS' : 'FAIL'} — supplied χ = ${supplied}; expected χ = ${expected}. Browser arithmetic only; not a Lean proof.`;
    }
    el('euler').addEventListener('input', checkEuler);
    el('corrupt').addEventListener('click', () => { el('euler').value = '-198'; checkEuler(); });
    el('restore').addEventListener('click', () => { el('euler').value = '-200'; checkEuler(); });
    render();
    checkEuler();
}());
