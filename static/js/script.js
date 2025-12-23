// Sample data for results table
const sampleResults = [
    { rank: 1, candidate_id: "CY_147823", score: 0.9847, verified: true, notes: "High confidence" },
    { rank: 2, candidate_id: "CY_092341", score: 0.9721, verified: true, notes: "Verified match" },
    { rank: 3, candidate_id: "CY_183294", score: 0.9658, verified: true, notes: "" },
    { rank: 4, candidate_id: "CY_045619", score: 0.9543, verified: true, notes: "Strong signal" },
    { rank: 5, candidate_id: "CY_128473", score: 0.9421, verified: false, notes: "Needs review" },
    { rank: 6, candidate_id: "CY_076234", score: 0.9385, verified: true, notes: "" },
    { rank: 7, candidate_id: "CY_201394", score: 0.9312, verified: true, notes: "Confirmed" },
    { rank: 8, candidate_id: "CY_159472", score: 0.9287, verified: true, notes: "" },
    { rank: 9, candidate_id: "CY_083124", score: 0.9156, verified: true, notes: "" },
    { rank: 10, candidate_id: "CY_194735", score: 0.9043, verified: false, notes: "Edge case" },
    { rank: 11, candidate_id: "CY_067281", score: 0.8987, verified: true, notes: "" },
    { rank: 12, candidate_id: "CY_142398", score: 0.8921, verified: true, notes: "" },
    { rank: 13, candidate_id: "CY_185749", score: 0.8876, verified: true, notes: "" },
    { rank: 14, candidate_id: "CY_093827", score: 0.8754, verified: true, notes: "" },
    { rank: 15, candidate_id: "CY_127463", score: 0.8698, verified: false, notes: "Low signal" },
    { rank: 16, candidate_id: "CY_074159", score: 0.8621, verified: true, notes: "" },
    { rank: 17, candidate_id: "CY_198234", score: 0.8543, verified: true, notes: "" },
    { rank: 18, candidate_id: "CY_136472", score: 0.8487, verified: true, notes: "" },
    { rank: 19, candidate_id: "CY_081923", score: 0.8421, verified: true, notes: "" },
    { rank: 20, candidate_id: "CY_165847", score: 0.8376, verified: false, notes: "Ambiguous" }
];

// Populate the results table
function populateResultsTable() {
    const tbody = document.getElementById('results-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';

    sampleResults.forEach(result => {
        const row = document.createElement('tr');

        const verifiedClass = result.verified ? 'verified-true' : 'verified-false';
        const verifiedText = result.verified ? 'Yes' : 'No';

        row.innerHTML = `
            <td>${result.rank}</td>
            <td><code>${result.candidate_id}</code></td>
            <td>${result.score.toFixed(4)}</td>
            <td class="${verifiedClass}">${verifiedText}</td>
            <td>${result.notes}</td>
        `;

        tbody.appendChild(row);
    });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', populateResultsTable);
} else {
    populateResultsTable();
}
