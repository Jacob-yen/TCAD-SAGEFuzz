use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Write as _,
    fs,
    path::Path,
};

use riscv_fuzz_plugin_api::{ISABase, RegisterValue, parse_register_name};
use riscv_fuzz_testcase_engine::testcase::GeneratedTestCase;
use riscv_instruction::RegisterBank;
use riscv_instruction_types::{
    LoadImmediatePurpose, PseudoInstruction, XLenAddress, XLenImmediate,
};

use super::replay::infer_target_hart_for_test_index;
use super::{
    CoverageMode, DiffAnalysisError, WriteDifference, WriteProbeOutcome, copy_reg_instruction,
    detect_unique_writes, isa_profile, iteration_summary_path, load_byte_instruction,
    run_iteration_with_per_impl_testcases, write_execution_state_report, x_reg,
};
use crate::normalize::{logical_instruction_at_location, test_instruction_location};
use crate::report_io::write_string_report;

#[derive(Debug, Clone)]
enum ProbeCaptureKind {
    CopyReg { src: String },
    LoadByte { offset: u64 },
}

#[derive(Debug, Clone)]
struct ProbeCaptureStep {
    test_index: usize,
    asm: String,
    kind: ProbeCaptureKind,
}

#[derive(Debug, Clone)]
struct CaptureObservation {
    step: ProbeCaptureStep,
    global_index: usize,
    per_impl: BTreeMap<String, Option<u64>>,
    is_complete: bool,
    is_mismatch: bool,
}

struct PreparedWriteProbe {
    canonical_testcase: GeneratedTestCase,
    per_impl_testcases: HashMap<String, GeneratedTestCase>,
    capture_steps: Vec<ProbeCaptureStep>,
    reg_probes: Vec<String>,
    mem_probes: Vec<u64>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_write_probe_verification(
    difference: &WriteDifference,
    replay_testcase: &GeneratedTestCase,
    replay_per_impl_testcases: &HashMap<String, GeneratedTestCase>,
    impl_order: &[String],
    replay_dir: &Path,
    target_index: usize,
    coverage_mode: CoverageMode,
    emit_execution_output_json: bool,
    emit_execution_report_md: bool,
    max_registers: usize,
    max_memory_bytes: usize,
) -> Result<WriteProbeOutcome, DiffAnalysisError> {
    let report_path = replay_dir.join("report_02_write_probe.md");
    let probe_dir = replay_dir.join("probe_verify");
    fs::create_dir_all(&probe_dir)?;

    let (temp_start, temp_end) = replay_testcase
        .config
        .random_config
        .memory_access_config
        .memory_base_register_number_range;
    let temp_regs: Vec<u8> = (temp_start..=temp_end).collect();
    if temp_regs.len() < 2 {
        let reason = format!(
            "insufficient temp registers for probing: need >=2, got {:?}",
            replay_testcase
                .config
                .random_config
                .memory_access_config
                .memory_base_register_number_range
        );
        let outcome = WriteProbeOutcome::Inconclusive { reason };
        write_probe_report(
            &report_path,
            replay_testcase,
            impl_order,
            target_index,
            None,
            None,
            &[],
            &[],
            &[],
            &outcome,
        )?;
        return Ok(outcome);
    }

    let probe_reg = temp_regs[0];
    let addr_reg = temp_regs[1];
    let probe_reg_name = format!("x{}", probe_reg);
    let addr_reg_name = format!("x{}", addr_reg);

    let prepared = prepare_write_probe(
        difference,
        replay_testcase,
        replay_per_impl_testcases,
        impl_order,
        target_index,
        probe_reg,
        addr_reg,
        max_registers,
        max_memory_bytes,
    )?;

    if prepared.reg_probes.is_empty() && prepared.mem_probes.is_empty() {
        let outcome = WriteProbeOutcome::Inconclusive {
            reason: "no mismatching integer registers or memory bytes to probe".to_string(),
        };
        write_probe_report(
            &report_path,
            replay_testcase,
            impl_order,
            target_index,
            Some((&probe_reg_name, &addr_reg_name)),
            None,
            &[],
            &[],
            &[],
            &outcome,
        )?;
        return Ok(outcome);
    }

    let probe_execution = run_iteration_with_per_impl_testcases(
        &prepared.canonical_testcase,
        &prepared.per_impl_testcases,
        impl_order,
        &probe_dir,
        coverage_mode,
        emit_execution_output_json,
        emit_execution_report_md,
    )?;
    let probe_outputs = probe_execution.outputs;

    let summary_path = iteration_summary_path(&probe_dir);
    write_execution_state_report(
        &summary_path,
        &prepared.canonical_testcase,
        &probe_outputs,
        impl_order,
        Some(&probe_execution.timings),
        None,
    )?;

    let probe_diffs =
        detect_unique_writes(impl_order, &prepared.canonical_testcase, &probe_outputs)?;
    let target_mismatch_persists = probe_diffs.iter().any(|d| d.test_index == target_index);

    let mut observations: Vec<CaptureObservation> = Vec::new();
    let mut any_complete = false;
    let mut any_mismatch = false;
    for step in &prepared.capture_steps {
        let location =
            test_instruction_location(impl_order, &prepared.canonical_testcase, step.test_index)?;
        let mut global_index = Some(location.global_index);
        let mut per_impl: BTreeMap<String, Option<u64>> = BTreeMap::new();
        for impl_ref in impl_order {
            let out = probe_outputs.get(impl_ref).ok_or_else(|| {
                DiffAnalysisError::MissingInstructions {
                    core_id: impl_ref.to_string(),
                }
            })?;
            let instruction =
                logical_instruction_at_location(out, location).ok_or_else(|| {
                    DiffAnalysisError::TestInstructionCountMismatch {
                        details: format!(
                            "probe output missing register changes at test index {} (hart {} local {}) for {}",
                            step.test_index,
                            location.hart_id,
                            location.hart_local_instruction_index,
                            impl_ref
                        ),
                    }
                })?;
            match global_index {
                Some(existing) if existing != instruction.instruction_index => {
                    return Err(DiffAnalysisError::TestInstructionCountMismatch {
                        details: format!(
                            "probe test index {} maps to different global indices: {} vs {} in {}",
                            step.test_index, existing, instruction.instruction_index, impl_ref
                        ),
                    });
                }
                Some(_) => {}
                None => global_index = Some(instruction.instruction_index),
            }
            let changes = instruction.register_writes.as_slice();
            per_impl.insert(
                impl_ref.clone(),
                last_written_reg_value(changes, &probe_reg_name),
            );
        }
        let global_index =
            global_index.ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "probe output missing all implementations for test index {}",
                    step.test_index
                ),
            })?;

        let is_complete = per_impl.values().all(|v| v.is_some());
        let is_mismatch = if is_complete {
            any_complete = true;
            let mut iter = per_impl.values().filter_map(|v| *v);
            let first = iter
                .next()
                .expect("complete per-implementation capture set must contain at least one value");
            let mismatch = iter.any(|v| v != first);
            if mismatch {
                any_mismatch = true;
            }
            mismatch
        } else {
            false
        };

        observations.push(CaptureObservation {
            step: step.clone(),
            global_index,
            per_impl,
            is_complete,
            is_mismatch,
        });
    }

    let outcome = if any_mismatch {
        WriteProbeOutcome::Confirmed
    } else if !any_complete {
        WriteProbeOutcome::Inconclusive {
            reason: "probe observations missing register writes".to_string(),
        }
    } else if !target_mismatch_persists {
        WriteProbeOutcome::MismatchDisappeared
    } else {
        WriteProbeOutcome::Refuted
    };

    write_probe_report(
        &report_path,
        replay_testcase,
        impl_order,
        target_index,
        Some((&probe_reg_name, &addr_reg_name)),
        Some(target_mismatch_persists),
        &prepared.reg_probes,
        &prepared.mem_probes,
        &observations,
        &outcome,
    )?;

    Ok(outcome)
}

#[allow(clippy::too_many_arguments)]
fn prepare_write_probe(
    difference: &WriteDifference,
    replay_testcase: &GeneratedTestCase,
    replay_per_impl_testcases: &HashMap<String, GeneratedTestCase>,
    impl_order: &[String],
    target_index: usize,
    probe_reg: u8,
    addr_reg: u8,
    max_registers: usize,
    max_memory_bytes: usize,
) -> Result<PreparedWriteProbe, DiffAnalysisError> {
    let mut reg_probes = collect_mismatching_integer_registers(difference, impl_order);
    if reg_probes.len() > max_registers {
        reg_probes.truncate(max_registers);
    }

    let mut mem_probes = collect_mismatching_memory_offsets(difference, impl_order);
    if let Some((start, end)) = replay_testcase.mem_range.values().next().copied() {
        let mem_size = end.saturating_sub(start).saturating_add(1);
        mem_probes.retain(|off| *off < mem_size);
    }
    if mem_probes.len() > max_memory_bytes {
        mem_probes.truncate(max_memory_bytes);
    }

    let target_hart = infer_target_hart_for_test_index(replay_testcase, target_index)?;
    let capture_steps = build_capture_steps(
        replay_testcase.config.isa_base,
        probe_reg,
        addr_reg,
        target_index,
        &reg_probes,
        &mem_probes,
    )?;

    let canonical_impl =
        impl_order
            .first()
            .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
                details: "write probe requires at least one implementation".to_string(),
            })?;
    let canonical_mem_addrs =
        probe_absolute_mem_addresses(replay_testcase, canonical_impl, &mem_probes)?;
    let mut canonical_testcase = replay_testcase.clone();
    append_probe_instructions(
        &mut canonical_testcase,
        target_hart,
        replay_testcase.config.isa_base,
        probe_reg,
        addr_reg,
        &reg_probes,
        &canonical_mem_addrs,
    )?;

    let mut per_impl_testcases = HashMap::new();
    for impl_ref in impl_order {
        let replay_impl_testcase = replay_per_impl_testcases.get(impl_ref).ok_or_else(|| {
            DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            }
        })?;
        let mem_addrs = probe_absolute_mem_addresses(replay_impl_testcase, impl_ref, &mem_probes)?;
        let mut probe_testcase = replay_impl_testcase.clone();
        append_probe_instructions(
            &mut probe_testcase,
            target_hart,
            replay_impl_testcase.config.isa_base,
            probe_reg,
            addr_reg,
            &reg_probes,
            &mem_addrs,
        )?;
        per_impl_testcases.insert(impl_ref.clone(), probe_testcase);
    }

    Ok(PreparedWriteProbe {
        canonical_testcase,
        per_impl_testcases,
        capture_steps,
        reg_probes,
        mem_probes,
    })
}

fn build_capture_steps(
    isa_base: ISABase,
    probe_reg: u8,
    addr_reg: u8,
    target_index: usize,
    reg_probes: &[String],
    mem_probes: &[u64],
) -> Result<Vec<ProbeCaptureStep>, DiffAnalysisError> {
    let mut capture_steps = Vec::new();
    let mut next_test_index = target_index + 1;
    for src in reg_probes {
        let copy_instruction = copy_reg_instruction(isa_base, probe_reg, src)?;
        capture_steps.push(ProbeCaptureStep {
            test_index: next_test_index,
            asm: copy_instruction.to_string(),
            kind: ProbeCaptureKind::CopyReg { src: src.clone() },
        });
        next_test_index += 1;
    }
    for &off in mem_probes {
        let load_instruction = load_byte_instruction(isa_base, probe_reg, addr_reg);
        capture_steps.push(ProbeCaptureStep {
            test_index: next_test_index + 1,
            asm: load_instruction.to_string(),
            kind: ProbeCaptureKind::LoadByte { offset: off },
        });
        next_test_index += 2;
    }
    Ok(capture_steps)
}

fn append_probe_instructions(
    testcase: &mut GeneratedTestCase,
    target_hart: u32,
    isa_base: ISABase,
    probe_reg: u8,
    addr_reg: u8,
    reg_probes: &[String],
    mem_addrs: &[u64],
) -> Result<(), DiffAnalysisError> {
    for src in reg_probes {
        testcase.push_test_instruction_to_hart(
            target_hart as usize,
            copy_reg_instruction(isa_base, probe_reg, src)?,
        );
    }

    for &addr in mem_addrs {
        testcase.push_test_pseudo_to_hart(
            target_hart as usize,
            li_address_pseudo(addr_reg, addr, isa_base),
        );
        testcase.push_test_instruction_to_hart(
            target_hart as usize,
            load_byte_instruction(isa_base, probe_reg, addr_reg),
        );
    }

    Ok(())
}

fn probe_absolute_mem_addresses(
    testcase: &GeneratedTestCase,
    impl_ref: &str,
    mem_probes: &[u64],
) -> Result<Vec<u64>, DiffAnalysisError> {
    let (mem_start, mem_end) = testcase
        .mem_range
        .get(impl_ref)
        .copied()
        .or_else(|| testcase.mem_range.values().next().copied())
        .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
            details: format!("missing mem_range for probe testcase {impl_ref}"),
        })?;

    let mem_size = mem_end.saturating_sub(mem_start).saturating_add(1);
    let mut result = Vec::with_capacity(mem_probes.len());
    for &offset in mem_probes {
        if offset >= mem_size {
            return Err(DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "probe memory offset 0x{offset:x} exceeds mem_range size 0x{mem_size:x} for {impl_ref}"
                ),
            });
        }
        result.push(mem_start.checked_add(offset).ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "probe memory address overflow for offset 0x{offset:x} in {impl_ref}"
                ),
            }
        })?);
    }
    Ok(result)
}

fn li_address_pseudo(rd: u8, address: u64, isa_base: ISABase) -> PseudoInstruction {
    let address = XLenAddress::from_u64_for(isa_profile(isa_base), address)
        .expect("probe address must fit active xlen");
    PseudoInstruction::LoadImmediate {
        rd: x_reg(rd),
        immediate: XLenImmediate::from_address(address),
        purpose: LoadImmediatePurpose::Generic,
    }
}

fn last_written_reg_value(changes: &[RegisterValue], reg_name: &str) -> Option<u64> {
    let target = parse_register_name(reg_name)?;
    changes
        .iter()
        .filter(|entry| parse_register_name(&entry.name) == Some(target))
        .map(|entry| entry.value)
        .next_back()
}

pub(super) fn collect_mismatching_integer_registers(
    difference: &WriteDifference,
    impl_order: &[String],
) -> Vec<String> {
    let mut names = BTreeSet::new();
    for impl_ref in impl_order {
        if let Some(entries) = difference.register_changes.get(impl_ref) {
            for (name, _) in entries {
                names.insert(name.clone());
            }
        }
    }

    let mut result = Vec::new();
    for name in names {
        let trimmed = name.trim();
        let Some(parsed) = parse_register_name(trimmed) else {
            continue;
        };
        if parsed.bank != RegisterBank::Integer {
            continue;
        }
        if parsed.index == 0 {
            continue;
        }
        let canonical = format!("x{}", parsed.index);

        let mut baseline: Option<Option<u64>> = None;
        let mut differs = false;
        for impl_ref in impl_order {
            let value = difference
                .register_changes
                .get(impl_ref)
                .and_then(|entries| {
                    entries
                        .iter()
                        .find(|(n, _)| parse_register_name(n) == Some(parsed))
                })
                .map(|(_, v)| *v);
            match &mut baseline {
                Some(existing) => {
                    if *existing != value {
                        differs = true;
                    }
                }
                None => baseline = Some(value),
            }
        }
        if differs {
            result.push(canonical);
        }
    }

    result.sort_by_key(|name| {
        parse_register_name(name)
            .map(|parsed| parsed.index as u32)
            .unwrap_or(u32::MAX)
    });
    result
}

pub(super) fn collect_mismatching_memory_offsets(
    difference: &WriteDifference,
    impl_order: &[String],
) -> Vec<u64> {
    let mut addrs = BTreeSet::new();
    for impl_ref in impl_order {
        if let Some(entries) = difference.memory_changes.get(impl_ref) {
            for (addr, _) in entries {
                addrs.insert(*addr);
            }
        }
    }

    let mut result = Vec::new();
    for addr in addrs {
        let mut baseline: Option<Option<u8>> = None;
        let mut differs = false;
        for impl_ref in impl_order {
            let value = difference
                .memory_changes
                .get(impl_ref)
                .and_then(|entries| entries.iter().find(|(a, _)| *a == addr))
                .map(|(_, v)| *v);
            match &mut baseline {
                Some(existing) => {
                    if *existing != value {
                        differs = true;
                    }
                }
                None => baseline = Some(value),
            }
        }
        if differs {
            result.push(addr);
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn write_probe_report(
    path: &Path,
    replay_testcase: &GeneratedTestCase,
    impl_order: &[String],
    target_index: usize,
    probe_regs: Option<(&str, &str)>,
    target_mismatch_persists: Option<bool>,
    reg_probes: &[String],
    mem_probes: &[u64],
    observations: &[CaptureObservation],
    outcome: &WriteProbeOutcome,
) -> Result<(), DiffAnalysisError> {
    let mut content = String::new();
    writeln!(&mut content, "# 写入差异探针验证报告\n").unwrap();
    writeln!(&mut content, "- 目标测试指令索引：{}", target_index).unwrap();

    if !impl_order.is_empty() {
        let inst = replay_testcase
            .traceable_test_pair_at(target_index)
            .map(|(line, _)| line)
            .unwrap_or_else(|| "-".to_string());
        writeln!(&mut content, "- 目标指令：`{}`", inst).unwrap();
    }

    if let Some((probe, addr)) = probe_regs {
        writeln!(&mut content, "- 探针寄存器：值=`{probe}`，地址=`{addr}`").unwrap();
    }

    if let Some(parent) = path.parent() {
        let probe_dir = parent.join("probe_verify");
        writeln!(&mut content, "- 探针输出目录：`{}`", probe_dir.display()).unwrap();
    }

    if let Some(persist) = target_mismatch_persists {
        writeln!(
            &mut content,
            "- 原差异在探针程序中是否仍存在：{}",
            if persist { "是" } else { "否" }
        )
        .unwrap();
    }

    if !reg_probes.is_empty() {
        writeln!(
            &mut content,
            "- 被探测整数寄存器：{}",
            reg_probes.join(", ")
        )
        .unwrap();
    }
    if !mem_probes.is_empty() {
        let mem_fmt = mem_probes
            .iter()
            .map(|v| format!("0x{:x}", v))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(&mut content, "- 被探测内存字节偏移：{mem_fmt}").unwrap();
    }

    writeln!(&mut content, "\n- 结论：{}\n", outcome.summary()).unwrap();

    if !observations.is_empty() {
        let probe_reg = probe_regs.map(|(probe, _)| probe).unwrap_or("probe");
        writeln!(&mut content, "## 探针观测（probe_reg 写入值）\n").unwrap();

        write!(&mut content, "| # | Test# | Global# | Probe | ASM |").unwrap();
        for impl_ref in impl_order {
            write!(&mut content, " {} |", impl_ref).unwrap();
        }
        writeln!(&mut content, " Status |").unwrap();

        write!(&mut content, "|---:|---:|---:|---|---|").unwrap();
        for _ in impl_order {
            write!(&mut content, "---|").unwrap();
        }
        writeln!(&mut content, "---|").unwrap();

        for (idx, obs) in observations.iter().enumerate() {
            let probe_desc = match &obs.step.kind {
                ProbeCaptureKind::CopyReg { src } => format!("copy {} -> {}", src, probe_reg),
                ProbeCaptureKind::LoadByte { offset } => {
                    format!("load mem[0x{:x}] -> {}", offset, probe_reg)
                }
            };
            let status = if !obs.is_complete {
                "missing"
            } else if obs.is_mismatch {
                "DIFF"
            } else {
                "same"
            };

            write!(
                &mut content,
                "| {} | {} | {} | {} | `{}` |",
                idx, obs.step.test_index, obs.global_index, probe_desc, obs.step.asm
            )
            .unwrap();
            for impl_ref in impl_order {
                let cell = match obs.per_impl.get(impl_ref).copied().flatten() {
                    Some(v) => format!("`0x{:x}`", v),
                    None => "-".to_string(),
                };
                write!(&mut content, " {} |", cell).unwrap();
            }
            writeln!(&mut content, " {} |", status).unwrap();
        }
    }

    write_string_report(path, content)
}

#[cfg(all(test, feature = "internal-tests"))]
mod tests {
    use super::*;
    use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
    use riscv_fuzz_testcase_engine::testcase::{
        GeneratedTestCase, HartConfig, HartProgram, TestCaseConfig,
    };
    use riscv_instruction::{
        RV32_I_ADDI, RV32Extensions, RV32IInstructions, RV32Instruction, RiscvInstruction,
    };
    use riscv_instruction_types::{
        Immediate, InstructionSequence, IntegerRegister, MemoryAccessConfig, RandomConfig,
        RegisterOperandConfig, ValidatedValue,
    };

    fn addi(rd: u8, rs1: u8, imm: i16) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADDI(RV32_I_ADDI {
            xd: IntegerRegister::new(rd).expect("valid rd"),
            xs1: IntegerRegister::new(rs1).expect("valid rs1"),
            imm: Immediate::<12>::new(imm.into()).expect("valid addi immediate"),
        })))
    }

    fn minimal_testcase(mem_range: HashMap<String, (u64, u64)>) -> GeneratedTestCase {
        GeneratedTestCase {
            config: TestCaseConfig {
                isa_base: ISABase::Rv32,
                extension_insts_count: 0,
                extension_inst_scaling: None,
                mem_size: 64,
                float_extension: Default::default(),
                unaligned_access_required: Some(false),
                random_config: RandomConfig {
                    memory_access_config: MemoryAccessConfig {
                        memory_base_address_range: MemoryAccessConfig::default()
                            .memory_base_address_range,
                        memory_offset_immediate_range: (0, 0),
                        memory_base_register_number_range: (8, 12),
                    },
                    register_operand_config: RegisterOperandConfig {
                        integer_register_number_range: (13, 20),
                        floating_point_register_number_range: (13, 20),
                        vector_register_number_range: (13, 20),
                    },
                    ..Default::default()
                },
                seed: None,
                hart_config: HartConfig::Single,
            },
            harts: vec![HartProgram {
                hart_id: 0,
                init: Vec::new(),
                test: vec![InstructionSequence::new(addi(13, 0, 1))],
                shared_init_sequence_indices: Default::default(),
                shared_test_sequence_indices: Default::default(),
            }],
            mem_range,
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        }
    }

    fn expect_probe_load_address(
        testcase: &GeneratedTestCase,
        li_test_index: usize,
        expected_addr: i64,
    ) {
        let pair = testcase
            .traceable_test_pair_at(li_test_index)
            .expect("missing traceable probe li");
        assert_eq!(pair.0, format!("li x9, {}", expected_addr));

        let next_pair = testcase
            .traceable_test_pair_at(li_test_index + 1)
            .expect("missing traceable probe load");
        assert_eq!(next_pair.0, "lbu x8, 0(x9)");
    }

    #[test]
    fn prepare_write_probe_uses_per_impl_absolute_addresses() {
        let impl_order = vec!["spike".to_string(), "picorv32".to_string()];
        let replay_testcase = minimal_testcase(HashMap::from([
            ("spike".to_string(), (0x8fff_ffbc, 0x8fff_fffb)),
            ("picorv32".to_string(), (0x0001_ffbc, 0x0001_fffb)),
        ]));
        let replay_per_impl_testcases = HashMap::from([
            (
                "spike".to_string(),
                minimal_testcase(HashMap::from([(
                    "spike".to_string(),
                    (0x8fff_ffbc, 0x8fff_fffb),
                )])),
            ),
            (
                "picorv32".to_string(),
                minimal_testcase(HashMap::from([(
                    "picorv32".to_string(),
                    (0x0001_ffbc, 0x0001_fffb),
                )])),
            ),
        ]);

        let difference = WriteDifference {
            test_index: 0,
            global_index: 0,
            register_changes: BTreeMap::new(),
            memory_changes: BTreeMap::from([
                ("spike".to_string(), vec![(0x24, 0x96)]),
                ("picorv32".to_string(), vec![]),
            ]),
            register_contexts: BTreeMap::new(),
            memory_contexts: BTreeMap::new(),
        };

        let prepared = prepare_write_probe(
            &difference,
            &replay_testcase,
            &replay_per_impl_testcases,
            &impl_order,
            0,
            8,
            9,
            8,
            16,
        )
        .expect("prepare probe");

        assert_eq!(prepared.mem_probes, vec![0x24]);
        expect_probe_load_address(&prepared.canonical_testcase, 1, -1879048224);

        let spike = prepared
            .per_impl_testcases
            .get("spike")
            .expect("missing spike testcase");
        expect_probe_load_address(spike, 1, -1879048224);

        let pico = prepared
            .per_impl_testcases
            .get("picorv32")
            .expect("missing picorv32 testcase");
        expect_probe_load_address(pico, 1, 131040);
    }
}
