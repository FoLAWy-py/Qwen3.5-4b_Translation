from witrans_tools.acceptance import CATEGORIES, DIRECTIONS, quality_gate

def corpus():
    return [{"id": f"{c}-{i}-{t}", "group_id": f"{c}-{i}", "category": c, "target_lang": t,
        "verdict": "pass", "format_valid": True, "ended": True, "language_correct": True}
        for c in CATEGORIES for i in range(60) for t in DIRECTIONS]

def test_small_or_correlated_samples_cannot_pass_numeric_gate():
    assert not quality_gate(corpus()[:16])["semantic_numeric_gate"]
    rows = corpus()
    for row in rows:
        row['group_id'] = "same-family"
    assert "fewer_than_300_source_groups" in quality_gate(rows)['failures']

def test_critical_and_concentrated_major_errors_cannot_hide_in_average():
    rows = corpus()
    rows[0]['verdict'] = 'critical'
    assert "critical_error_present" in quality_gate(rows)['failures']
    rows[0]['verdict'] = 'major'
    rows[2]['verdict'] = 'major'
    assert "quality:daily/en" in quality_gate(rows)['failures']

def test_numeric_pass_never_implies_full_release_approval():
    report = quality_gate(corpus())
    assert report['semantic_numeric_gate']
    assert not report['release_approved']
