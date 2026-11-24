use std::fmt::Write as _;

pub fn fmt_percent(value: f64) -> String {
    format!("{:.2}%", value * 100.0)
}

pub fn write_metric_definitions_section(out: &mut String) {
    writeln!(out, "## Metric Definitions").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "All ratio-style sync metrics below are reported as percentages."
    )
    .unwrap();
    writeln!(
        out,
        "For `Simul`, the denominator is the union of cycles where at least one hart is active; idle gaps where neither hart is active are excluded."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- `Simul` = simultaneous active-cycle ratio within that active-cycle union"
    )
    .unwrap();
    writeln!(
        out,
        "- `Reg` = ratio of simultaneous cycles where the two harts touch overlapping register sets"
    )
    .unwrap();
    writeln!(out, "- `Mem` = ratio of both-memory simultaneous cycles where the two harts hit the same address").unwrap();
    writeln!(
        out,
        "- `Score` = weighted combined ratio across the synchronization patterns above"
    )
    .unwrap();
    writeln!(out).unwrap();
}
