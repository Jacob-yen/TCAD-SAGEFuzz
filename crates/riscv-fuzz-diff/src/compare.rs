//! 用于执行输出和性能快照的严格结构化比较器。
//! Exact structural comparators for execution outputs and perf snapshots.
//!
//! 这些辅助函数不参与 testcase 归约。
//! These helpers do not participate in testcase reduction.
//! 它们服务于“对两个既有执行结果做直接同/异判断”的调用场景。
//! They are used when callers want a direct "same or different" decision between two
//! already-produced execution outputs.

use std::collections::BTreeSet;

use super::{normalize::logical_instruction_effects_unfiltered, view::DiffExecutionView};

#[derive(Debug, Clone)]
pub struct OutputMismatch {
    pub instruction_index: Option<usize>,
    pub message: String,
}

fn perf_note_is_collection_issue(note: &str) -> bool {
    let lower = note.to_ascii_lowercase();
    lower.starts_with("perf: cva6 synthesized")
        || lower.starts_with("perf: cva6 missing hart")
        || lower.starts_with("perf: rocket synthesized")
        || lower.starts_with("perf: rocket fixed")
        || lower.starts_with("perf: rocket missing hart")
        || lower.starts_with("perf: rocket non-monotonic")
        || lower.starts_with("perf: missing rvfi trace")
        || lower.starts_with("perf: failed to parse rvfi trace")
        || lower.starts_with("perf: failed to load perf symbols")
}

pub fn collect_perf_collection_issue_notes(output: &DiffExecutionView) -> Vec<String> {
    output
        .perf
        .notes
        .iter()
        .filter(|note| perf_note_is_collection_issue(note))
        .cloned()
        .collect()
}

fn fmt_reg_pairs(pairs: &[(String, u64)], max: usize) -> String {
    if pairs.is_empty() {
        return "[]".to_string();
    }
    let shown = pairs.len().min(max);
    let mut parts = Vec::with_capacity(shown);
    for (name, value) in pairs.iter().take(shown) {
        parts.push(format!("{name}=0x{value:x}"));
    }
    if pairs.len() > shown {
        parts.push(format!("...(+{} more)", pairs.len() - shown));
    }
    format!("[{}]", parts.join(", "))
}

fn fmt_mem_pairs(pairs: &[(u64, u8)], max: usize) -> String {
    if pairs.is_empty() {
        return "[]".to_string();
    }
    let shown = pairs.len().min(max);
    let mut parts = Vec::with_capacity(shown);
    for (addr, value) in pairs.iter().take(shown) {
        parts.push(format!("0x{addr:x}=0x{value:02x}"));
    }
    if pairs.len() > shown {
        parts.push(format!("...(+{} more)", pairs.len() - shown));
    }
    format!("[{}]", parts.join(", "))
}

pub fn compare_execution_views(
    a: &DiffExecutionView,
    b: &DiffExecutionView,
) -> Result<(), OutputMismatch> {
    // 这里故意保持严格：diff crate 用它做校验，而不是做 best-effort 排查。
    // This is intentionally strict: the diff crate uses it for verification, not for
    // best-effort triage.
    // 任何结构性不一致都会被报告为硬差异。
    // Any structural mismatch is reported as a hard difference.
    if a.isa_base != b.isa_base {
        return Err(OutputMismatch {
            instruction_index: None,
            message: format!("isa_base mismatch: {:?} vs {:?}", a.isa_base, b.isa_base),
        });
    }

    if a.logical_instruction_count() != b.logical_instruction_count() {
        return Err(OutputMismatch {
            instruction_index: None,
            message: format!(
                "instruction_count mismatch: {} vs {}",
                a.logical_instruction_count(),
                b.logical_instruction_count()
            ),
        });
    }

    let mut exc_a = a
        .logical_view
        .iter()
        .flat_map(|entry| {
            entry
                .exceptions
                .iter()
                .map(move |cause| (entry.instruction_index, cause.code))
        })
        .collect::<Vec<_>>();
    let mut exc_b = b
        .logical_view
        .iter()
        .flat_map(|entry| {
            entry
                .exceptions
                .iter()
                .map(move |cause| (entry.instruction_index, cause.code))
        })
        .collect::<Vec<_>>();
    exc_a.sort();
    exc_b.sort();
    if exc_a != exc_b {
        return Err(OutputMismatch {
            instruction_index: None,
            message: format!(
                "exception mismatch: left={} right={} (first_left={:?}, first_right={:?})",
                exc_a.len(),
                exc_b.len(),
                exc_a.first(),
                exc_b.first()
            ),
        });
    }

    for idx in 0..a.logical_instruction_count() {
        let left = a
            .logical_instruction(idx)
            .expect("trace instruction count checked above");
        let right = b
            .logical_instruction(idx)
            .expect("trace instruction count checked above");
        let (regs_a, mem_a) = logical_instruction_effects_unfiltered(left);
        let (regs_b, mem_b) = logical_instruction_effects_unfiltered(right);
        if regs_a != regs_b {
            return Err(OutputMismatch {
                instruction_index: Some(idx),
                message: format!(
                    "register_changes mismatch at #{idx}: left={} right={}",
                    fmt_reg_pairs(&regs_a, 12),
                    fmt_reg_pairs(&regs_b, 12)
                ),
            });
        }

        if mem_a != mem_b {
            return Err(OutputMismatch {
                instruction_index: Some(idx),
                message: format!(
                    "memory_changes mismatch at #{idx}: left={} right={}",
                    fmt_mem_pairs(&mem_a, 16),
                    fmt_mem_pairs(&mem_b, 16)
                ),
            });
        }
    }

    Ok(())
}

pub fn compare_perf_metrics(
    a: &DiffExecutionView,
    b: &DiffExecutionView,
) -> Result<(), OutputMismatch> {
    // perf 对比会先拒绝已知的采集问题，
    // Perf comparison rejects known collection issues up front,
    // 这样“匹配”才真正表示两边指标都存在且可以直接比较。
    // so that a "match" means the metrics were both present and directly comparable.
    let left_issues = collect_perf_collection_issue_notes(a);
    if !left_issues.is_empty() {
        return Err(OutputMismatch {
            instruction_index: None,
            message: format!(
                "perf collection issue on left output: {}",
                left_issues.join(" | ")
            ),
        });
    }

    let right_issues = collect_perf_collection_issue_notes(b);
    if !right_issues.is_empty() {
        return Err(OutputMismatch {
            instruction_index: None,
            message: format!(
                "perf collection issue on right output: {}",
                right_issues.join(" | ")
            ),
        });
    }

    if a.perf.global_cycles != b.perf.global_cycles {
        return Err(OutputMismatch {
            instruction_index: None,
            message: format!(
                "perf global_cycles mismatch: left={:?} right={:?}",
                a.perf.global_cycles, b.perf.global_cycles
            ),
        });
    }

    let mut harts: BTreeSet<u32> = BTreeSet::new();
    harts.extend(a.perf.per_hart.keys().copied());
    harts.extend(b.perf.per_hart.keys().copied());

    if harts.is_empty() {
        return Err(OutputMismatch {
            instruction_index: None,
            message: "perf per_hart metrics missing on both outputs".to_string(),
        });
    }

    for hart in harts {
        let Some(left) = a.perf.per_hart.get(&hart) else {
            return Err(OutputMismatch {
                instruction_index: None,
                message: format!("perf per_hart missing on left for hart {}", hart),
            });
        };
        let Some(right) = b.perf.per_hart.get(&hart) else {
            return Err(OutputMismatch {
                instruction_index: None,
                message: format!("perf per_hart missing on right for hart {}", hart),
            });
        };

        if left.mcycle_start != right.mcycle_start {
            return Err(OutputMismatch {
                instruction_index: None,
                message: format!(
                    "perf hart {} mcycle_start mismatch: left={:?} right={:?}",
                    hart, left.mcycle_start, right.mcycle_start
                ),
            });
        }
        if left.mcycle_end != right.mcycle_end {
            return Err(OutputMismatch {
                instruction_index: None,
                message: format!(
                    "perf hart {} mcycle_end mismatch: left={:?} right={:?}",
                    hart, left.mcycle_end, right.mcycle_end
                ),
            });
        }
        if left.minstret_start != right.minstret_start {
            return Err(OutputMismatch {
                instruction_index: None,
                message: format!(
                    "perf hart {} minstret_start mismatch: left={:?} right={:?}",
                    hart, left.minstret_start, right.minstret_start
                ),
            });
        }
        if left.minstret_end != right.minstret_end {
            return Err(OutputMismatch {
                instruction_index: None,
                message: format!(
                    "perf hart {} minstret_end mismatch: left={:?} right={:?}",
                    hart, left.minstret_end, right.minstret_end
                ),
            });
        }

        let left_mcycle_delta = match (left.mcycle_start, left.mcycle_end) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        };
        let right_mcycle_delta = match (right.mcycle_start, right.mcycle_end) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        };
        if left_mcycle_delta != right_mcycle_delta {
            return Err(OutputMismatch {
                instruction_index: None,
                message: format!(
                    "perf hart {} mcycle delta mismatch: left={:?} right={:?}",
                    hart, left_mcycle_delta, right_mcycle_delta
                ),
            });
        }

        let left_minstret_delta = match (left.minstret_start, left.minstret_end) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        };
        let right_minstret_delta = match (right.minstret_start, right.minstret_end) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        };
        if left_minstret_delta != right_minstret_delta {
            return Err(OutputMismatch {
                instruction_index: None,
                message: format!(
                    "perf hart {} minstret delta mismatch: left={:?} right={:?}",
                    hart, left_minstret_delta, right_minstret_delta
                ),
            });
        }

        if let (Some(mcycle), Some(minstret)) = (left_mcycle_delta, left_minstret_delta)
            && minstret > mcycle
        {
            return Err(OutputMismatch {
                instruction_index: None,
                message: format!(
                    "perf hart {} invalid left metrics: minstret delta {} > mcycle delta {}",
                    hart, minstret, mcycle
                ),
            });
        }
        if let (Some(mcycle), Some(minstret)) = (right_mcycle_delta, right_minstret_delta)
            && minstret > mcycle
        {
            return Err(OutputMismatch {
                instruction_index: None,
                message: format!(
                    "perf hart {} invalid right metrics: minstret delta {} > mcycle delta {}",
                    hart, minstret, mcycle
                ),
            });
        }
    }

    Ok(())
}
