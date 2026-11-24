use rand::{Rng as _, SeedableRng as _, rngs::StdRng, seq::SliceRandom as _};
use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
use riscv_instruction::{RV64Extensions, RiscvInstruction};
use riscv_instruction_types::{AsmDecodeWithContext, InstructionSequence, IsaDecodeContext};

use crate::{
    GenerationError,
    rvv_catalog_legal::{RvvSamplingProfile, check_asm_under_config, domain_for_asm},
    rvv_constraints::RvvConfigConstraint,
    rvv_legal,
    testcase::{GeneratedTestCase, HartPhase, TestCaseConfig},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum StateKind {
    VecReg,
    Mask,
    Mem,
    Gpr,
    Fpr,
    VCtrl,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Family {
    Config,
    Load,
    Store,
    VectorCompute,
    Reduction,
    Move,
}

struct FamilyRule {
    family: Family,
    consumes: &'static [StateKind],
    produces: &'static [StateKind],
}

const FAMILY_RULES: &[FamilyRule] = &[
    FamilyRule {
        family: Family::Config,
        consumes: &[StateKind::Gpr],
        produces: &[StateKind::VCtrl, StateKind::Gpr],
    },
    FamilyRule {
        family: Family::Load,
        consumes: &[
            StateKind::VCtrl,
            StateKind::Gpr,
            StateKind::Mem,
            StateKind::Mask,
        ],
        produces: &[StateKind::VecReg, StateKind::VCtrl],
    },
    FamilyRule {
        family: Family::Store,
        consumes: &[
            StateKind::VCtrl,
            StateKind::VecReg,
            StateKind::Gpr,
            StateKind::Mask,
        ],
        produces: &[StateKind::Mem],
    },
    FamilyRule {
        family: Family::VectorCompute,
        consumes: &[
            StateKind::VCtrl,
            StateKind::VecReg,
            StateKind::Gpr,
            StateKind::Fpr,
            StateKind::Mask,
        ],
        produces: &[StateKind::VecReg, StateKind::Mask],
    },
    FamilyRule {
        family: Family::Reduction,
        consumes: &[StateKind::VCtrl, StateKind::VecReg, StateKind::Mask],
        produces: &[StateKind::VecReg],
    },
    FamilyRule {
        family: Family::Move,
        consumes: &[
            StateKind::VCtrl,
            StateKind::VecReg,
            StateKind::Gpr,
            StateKind::Fpr,
        ],
        produces: &[StateKind::VecReg, StateKind::Gpr, StateKind::Fpr],
    },
];

#[derive(Default)]
struct FamilyIndex;

impl FamilyIndex {
    fn producers(&self, kind: StateKind) -> Vec<Family> {
        FAMILY_RULES
            .iter()
            .filter(|rule| rule.produces.contains(&kind))
            .map(|rule| rule.family)
            .collect()
    }

    fn consumers(&self, kind: StateKind) -> Vec<Family> {
        FAMILY_RULES
            .iter()
            .filter(|rule| rule.consumes.contains(&kind))
            .map(|rule| rule.family)
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StateInstance {
    kind: StateKind,
    selector: String,
}

impl StateInstance {
    fn vector(base: u8) -> Self {
        Self {
            kind: StateKind::VecReg,
            selector: format!("v{base}"),
        }
    }
}

struct GraphNode {
    id: u32,
    family: Family,
    lines: Vec<String>,
    consumes: Vec<StateInstance>,
    produces: Vec<StateInstance>,
}

struct GraphEdge {
    producer: u32,
    consumer: u32,
    state: StateInstance,
}

#[derive(Default)]
struct StateGraph {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    next_id: u32,
    vector_group_width: u8,
}

impl StateGraph {
    fn node(
        &mut self,
        family: Family,
        lines: Vec<String>,
        consumes: Vec<StateInstance>,
        produces: Vec<StateInstance>,
    ) -> GraphNode {
        let id = self.next_id;
        self.next_id += 1;
        GraphNode {
            id,
            family,
            lines,
            consumes,
            produces,
        }
    }

    fn before(&mut self, anchor: u32, node: GraphNode, state: StateInstance) -> u32 {
        let index = self
            .nodes
            .iter()
            .position(|item| item.id == anchor)
            .expect("anchor exists");
        let id = node.id;
        self.nodes.insert(index, node);
        self.edges.push(GraphEdge {
            producer: id,
            consumer: anchor,
            state,
        });
        id
    }

    fn after(&mut self, anchor: u32, node: GraphNode, state: StateInstance) -> u32 {
        let index = self
            .nodes
            .iter()
            .position(|item| item.id == anchor)
            .expect("anchor exists");
        let id = node.id;
        self.nodes.insert(index + 1, node);
        self.edges.push(GraphEdge {
            producer: anchor,
            consumer: id,
            state,
        });
        id
    }

    fn rendered_lines(&self) -> Vec<String> {
        self.nodes
            .iter()
            .flat_map(|node| node.lines.iter().cloned())
            .collect()
    }

    fn valid_edges(&self) -> bool {
        self.edges.iter().all(|edge| {
            let p = self.nodes.iter().position(|node| node.id == edge.producer);
            let c = self.nodes.iter().position(|node| node.id == edge.consumer);
            matches!((p,c), (Some(p),Some(c)) if p < c)
                && self.nodes[p.unwrap()].produces.contains(&edge.state)
                && self.nodes[c.unwrap()].consumes.contains(&edge.state)
                && self.nodes[p.unwrap() + 1..c.unwrap()].iter().all(|node| {
                    node.produces
                        .iter()
                        .all(|written| !self.overwrites(&edge.state, written))
                })
        })
    }

    fn overwrites(&self, live: &StateInstance, written: &StateInstance) -> bool {
        if live.kind == written.kind && live.selector == written.selector {
            return true;
        }
        let vector_span = |state: &StateInstance| -> Option<(u8, u8)> {
            let width = match state.kind {
                StateKind::VecReg => self.vector_group_width,
                StateKind::Mask => 1,
                _ => return None,
            };
            let base = state.selector.strip_prefix('v')?.parse::<u8>().ok()?;
            Some((base, width))
        };
        if let (Some((left, left_width)), Some((right, right_width))) =
            (vector_span(live), vector_span(written))
        {
            return rvv_legal::groups_overlap(left, left_width, right, right_width);
        }
        false
    }
}

fn error(message: impl Into<String>) -> GenerationError {
    GenerationError::StructuredInstructionBuildFailed {
        message: message.into(),
    }
}

fn minimum_memory_size(profile: RvvSamplingProfile) -> u64 {
    0x200 + u64::from(profile.vlen_bits).max(0x200)
}

fn decode(line: &str) -> Result<InstructionSequence<RiscvInstruction>, GenerationError> {
    RiscvInstruction::decode_asm_with_context(line, IsaDecodeContext::RV64)
        .map(InstructionSequence::new)
        .map_err(|source| {
            error(format!(
                "artifact state-flow cannot decode `{line}`: {source}"
            ))
        })
}

fn scalar_load(sew: u16) -> &'static str {
    match sew {
        8 => "lbu",
        16 => "lhu",
        32 => "lwu",
        64 => "ld",
        _ => unreachable!(),
    }
}

const INTEGER_VX: &[&str] = &[
    "vadd.vx",
    "vsub.vx",
    "vrsub.vx",
    "vand.vx",
    "vor.vx",
    "vxor.vx",
    "vsll.vx",
    "vsrl.vx",
    "vsra.vx",
    "vminu.vx",
    "vmin.vx",
    "vmaxu.vx",
    "vmax.vx",
    "vmul.vx",
    "vmulh.vx",
    "vmulhu.vx",
    "vmulhsu.vx",
    "vdivu.vx",
    "vdiv.vx",
    "vremu.vx",
    "vrem.vx",
    "vsaddu.vx",
    "vsadd.vx",
    "vssubu.vx",
    "vssub.vx",
    "vaaddu.vx",
    "vaadd.vx",
    "vasubu.vx",
    "vasub.vx",
    "vsmul.vx",
];
const INTEGER_VI: &[&str] = &["vadd.vi", "vrsub.vi", "vand.vi", "vor.vi", "vxor.vi"];
const INTEGER_VV: &[&str] = &[
    "vadd.vv",
    "vsub.vv",
    "vand.vv",
    "vor.vv",
    "vxor.vv",
    "vminu.vv",
    "vmin.vv",
    "vmaxu.vv",
    "vmax.vv",
    "vmul.vv",
    "vdivu.vv",
    "vdiv.vv",
    "vremu.vv",
    "vrem.vv",
    "vsaddu.vv",
    "vsadd.vv",
    "vssubu.vv",
    "vssub.vv",
];

fn sampled_alu(
    output: u8,
    input: u8,
    scalar: u8,
    masked: bool,
    cfg: RvvConfigConstraint,
    elen: u16,
    groups: &mut Vec<u8>,
    rng: &mut StdRng,
) -> Result<(String, Option<u8>), GenerationError> {
    let suffix = if masked { ", v0.t" } else { "" };
    let families = [INTEGER_VX, INTEGER_VI, INTEGER_VV]
        .iter()
        .enumerate()
        .filter(|(family, _)| *family != 2 || !groups.is_empty())
        .filter_map(|(family, candidates)| {
            let eligible = candidates
                .iter()
                .copied()
                .filter(|op| rvv_legal::opcode_config_eligible(op, cfg.sew, cfg.lmul, elen))
                .collect::<Vec<_>>();
            (!eligible.is_empty()).then_some((family, eligible))
        })
        .collect::<Vec<_>>();
    if families.is_empty() {
        return Err(error(
            "no eligible integer ALU form in the selected VC domain",
        ));
    }
    let (family, candidates) = &families[rng.random_range(0..families.len())];
    let op = candidates[rng.random_range(0..candidates.len())];
    Ok(match *family {
        0 => (format!("{op} v{output}, v{input}, x{scalar}{suffix}"), None),
        1 => {
            let immediate = rng.random_range(-16..=15);
            (
                format!("{op} v{output}, v{input}, {immediate}{suffix}"),
                None,
            )
        }
        _ => {
            let extra = groups.pop().expect("sampled VV has a spare vector group");
            (
                format!("{op} v{output}, v{input}, v{extra}{suffix}"),
                Some(extra),
            )
        }
    })
}

fn sampled_load(sew: u16, output: u8, rng: &mut StdRng) -> (String, bool) {
    match rng.random_range(0..3) {
        0 => (format!("vle{sew}.v v{output}, (x11)"), false),
        1 => (format!("vlse{sew}.v v{output}, (x11), x9"), false),
        _ => (format!("vle{sew}ff.v v{output}, (x11)"), true),
    }
}

fn sampled_store(sew: u16, input: u8, rng: &mut StdRng) -> String {
    if rng.random_bool(0.5) {
        format!("vse{sew}.v v{input}, (x12)")
    } else {
        format!("vsse{sew}.v v{input}, (x12), x9")
    }
}

fn insert_terminal_observer(
    graph: &mut StateGraph,
    producer: u32,
    input: u8,
    cfg: RvvConfigConstraint,
    output_addr: u64,
    store: bool,
    rng: &mut StdRng,
) {
    let gpr = StateInstance {
        kind: StateKind::Gpr,
        selector: "x17".into(),
    };
    let observed = StateInstance {
        kind: StateKind::Mem,
        selector: format!("{:#x}", output_addr + 64),
    };
    if store {
        let first_element = StateInstance {
            kind: StateKind::Mem,
            selector: format!("{output_addr:#x}"),
        };
        let vector_store = graph.node(
            Family::Store,
            vec![sampled_store(cfg.sew, input, rng)],
            vec![StateInstance::vector(input)],
            vec![first_element.clone()],
        );
        let store_id = graph.after(producer, vector_store, StateInstance::vector(input));
        let observer = graph.node(
            Family::Move,
            vec![
                format!("{} x17, 0(x12)", scalar_load(cfg.sew)),
                "sd x17, 64(x12)".into(),
            ],
            vec![first_element.clone()],
            vec![gpr, observed],
        );
        graph.after(store_id, observer, first_element);
    } else {
        let extraction = graph.node(
            Family::Move,
            vec![format!("vmv.x.s x17, v{input}")],
            vec![StateInstance::vector(input)],
            vec![gpr.clone()],
        );
        let extraction_id = graph.after(producer, extraction, StateInstance::vector(input));
        let observer = graph.node(
            Family::Store,
            vec!["sd x17, 64(x12)".into()],
            vec![gpr.clone()],
            vec![observed],
        );
        graph.after(extraction_id, observer, gpr);
    }
}

fn append_vstart_flow(
    graph: &mut StateGraph,
    target: u8,
    cfg: RvvConfigConstraint,
    output_addr: u64,
) {
    let vector = StateInstance::vector(target);
    let control = StateInstance {
        kind: StateKind::VCtrl,
        selector: "vl/vtype/vstart".into(),
    };
    let initial = graph.node(
        Family::Move,
        vec![format!("vmv.v.i v{target}, 0")],
        vec![],
        vec![vector.clone()],
    );
    let initial_id = initial.id;
    graph.nodes.push(initial);
    let writer = graph.node(
        Family::Config,
        vec!["li x20, 1".into(), "csrrw x0, vstart, x20".into()],
        vec![],
        vec![control.clone()],
    );
    graph.nodes.push(writer);
    let update = graph.node(
        Family::VectorCompute,
        vec![format!("vmv.v.i v{target}, 1")],
        vec![vector.clone(), control.clone()],
        vec![vector.clone(), control.clone()],
    );
    let update_id = update.id;
    graph.nodes.push(update);
    graph.edges.push(GraphEdge {
        producer: initial_id,
        consumer: update_id,
        state: vector.clone(),
    });
    let csr_observer = graph.node(
        Family::Move,
        vec!["csrrs x21, vstart, x0".into(), "sd x21, 88(x12)".into()],
        vec![control.clone()],
        vec![StateInstance {
            kind: StateKind::Gpr,
            selector: "x21".into(),
        }],
    );
    let csr_observer_id = csr_observer.id;
    graph.nodes.push(csr_observer);
    graph.edges.push(GraphEdge {
        producer: update_id,
        consumer: csr_observer_id,
        state: control,
    });
    let element_bytes = u64::from(cfg.sew / 8);
    let second_element = StateInstance {
        kind: StateKind::Mem,
        selector: format!("{:#x}", output_addr + element_bytes),
    };
    let store = graph.node(
        Family::Store,
        vec![format!("vse{}.v v{target}, (x12)", cfg.sew)],
        vec![vector.clone()],
        vec![second_element.clone()],
    );
    let store_id = store.id;
    graph.nodes.push(store);
    graph.edges.push(GraphEdge {
        producer: update_id,
        consumer: store_id,
        state: vector,
    });
    let observer = graph.node(
        Family::Move,
        vec![
            format!("{} x22, {element_bytes}(x12)", scalar_load(cfg.sew)),
            "sd x22, 96(x12)".into(),
        ],
        vec![second_element.clone()],
        vec![StateInstance {
            kind: StateKind::Gpr,
            selector: "x22".into(),
        }],
    );
    let observer_id = observer.id;
    graph.nodes.push(observer);
    graph.edges.push(GraphEdge {
        producer: store_id,
        consumer: observer_id,
        state: second_element,
    });
}

fn insert_load_before(
    graph: &mut StateGraph,
    consumer: u32,
    output: u8,
    cfg: RvvConfigConstraint,
    source_addr: u64,
    rng: &mut StdRng,
) {
    let (line, fof) = sampled_load(cfg.sew, output, rng);
    let control = StateInstance {
        kind: StateKind::VCtrl,
        selector: "vl/vtype/vstart".into(),
    };
    let mut produced = vec![StateInstance::vector(output)];
    if fof {
        produced.push(control.clone());
    }
    let node = graph.node(
        Family::Load,
        vec![line],
        vec![
            StateInstance {
                kind: StateKind::Mem,
                selector: format!("{source_addr:#x}"),
            },
            control.clone(),
        ],
        produced,
    );
    let load_id = graph.before(consumer, node, StateInstance::vector(output));
    if fof {
        let observer = graph.node(
            Family::Move,
            vec!["csrr x19, vl".into(), "sd x19, 80(x12)".into()],
            vec![control.clone()],
            vec![StateInstance {
                kind: StateKind::Gpr,
                selector: "x19".into(),
            }],
        );
        graph.after(load_id, observer, control);
    }
}

fn config_and_groups(
    profile: RvvSamplingProfile,
    range: (u8, u8),
    rng: &mut StdRng,
) -> Result<(RvvConfigConstraint, Vec<u8>), GenerationError> {
    let domain = domain_for_asm("vadd.vx v16, v8, x13", profile, range)
        .ok_or_else(|| error("no VC domain for state-flow anchor"))?;
    let choices = domain
        .configurations
        .iter()
        .filter(|choice| {
            choice.sampled_avl_max.min(profile.memory_avl_cap) >= 2
                && rvv_legal::register_width(choice.sew, choice.lmul, choice.sew)
                    .is_some_and(|width| rvv_legal::legal_group_bases(range, width).len() >= 2)
        })
        .collect::<Vec<_>>();
    if choices.is_empty() {
        return Err(error("no legal configuration with two active elements"));
    }
    let choice = choices[rng.random_range(0..choices.len())];
    let vl = rng.random_range(2..=choice.sampled_avl_max.min(profile.memory_avl_cap).min(4));
    let cfg = RvvConfigConstraint::new(ISABase::Rv64, choice.sew, vl, choice.lmul, "tu", "mu");
    let width = rvv_legal::register_width(cfg.sew, cfg.lmul, cfg.sew)
        .expect("selected VC domain has width");
    let mut groups = rvv_legal::legal_group_bases(range, width);
    groups.shuffle(rng);
    if groups.len() < 2 {
        return Err(error(
            "state-flow needs two disjoint vector groups in the configured range",
        ));
    }
    Ok((cfg, groups))
}

fn validate_vector_lines(
    lines: &[String],
    cfg: RvvConfigConstraint,
    elen: u16,
) -> Result<(), GenerationError> {
    for line in lines
        .iter()
        .filter(|line| line.starts_with('v') && !line.starts_with("vset"))
    {
        let issues = check_asm_under_config(line, cfg, elen);
        if !issues.is_empty() {
            return Err(error(format!("VC rule rejected `{line}`: {issues:?}")));
        }
    }
    Ok(())
}

fn build_graph(
    cfg: RvvConfigConstraint,
    elen: u16,
    source_addr: u64,
    output_addr: u64,
    fp_enabled: bool,
    groups: &mut Vec<u8>,
    rng: &mut StdRng,
) -> Result<StateGraph, GenerationError> {
    let index = FamilyIndex;
    let source = groups
        .pop()
        .ok_or_else(|| error("missing source vector group"))?;
    let target = groups
        .pop()
        .ok_or_else(|| error("missing target vector group"))?;
    let mut graph = StateGraph {
        vector_group_width: rvv_legal::register_width(cfg.sew, cfg.lmul, cfg.sew)
            .ok_or_else(|| error("no legal vector group width"))?,
        ..StateGraph::default()
    };
    let config = graph.node(
        Family::Config,
        vec![
            format!(
                "vsetivli x0, {}, {}",
                cfg.vl,
                cfg.vsetivli_vtype_immediate()
            ),
            "csrrw x0, vstart, x0".into(),
        ],
        vec![],
        vec![StateInstance {
            kind: StateKind::VCtrl,
            selector: "vl/vtype/vstart".into(),
        }],
    );
    graph.nodes.push(config);
    let masked = rng.random_bool(0.5);
    let (anchor_line, anchor_extra) =
        sampled_alu(target, source, 13, masked, cfg, elen, groups, rng)?;
    let anchor = graph.node(
        Family::VectorCompute,
        vec![anchor_line],
        [
            vec![
                StateInstance::vector(source),
                StateInstance {
                    kind: StateKind::VCtrl,
                    selector: "vl/vtype/vstart".into(),
                },
            ],
            if masked {
                vec![
                    StateInstance {
                        kind: StateKind::Mask,
                        selector: "v0".into(),
                    },
                    StateInstance::vector(target),
                ]
            } else {
                vec![]
            },
            anchor_extra.map_or_else(Vec::new, |extra| vec![StateInstance::vector(extra)]),
        ]
        .concat(),
        vec![StateInstance::vector(target)],
    );
    let anchor_id = anchor.id;
    graph.nodes.push(anchor);
    if masked {
        let initialization = graph.node(
            Family::Move,
            vec![format!("vmv.v.i v{target}, 0")],
            vec![],
            vec![StateInstance::vector(target)],
        );
        graph.before(anchor_id, initialization, StateInstance::vector(target));
    }
    let mut unresolved = vec![(anchor_id, source)];
    if let Some(extra) = anchor_extra {
        unresolved.push((anchor_id, extra));
    }
    let mut frontier = Some((anchor_id, target));
    let budget = rng.random_range(3..=6);
    for _ in 0..budget {
        let backward = !unresolved.is_empty() && (frontier.is_none() || rng.random_bool(0.5));
        if backward {
            let (consumer, output) = unresolved.pop().expect("backward edge exists");
            let families = index.producers(StateKind::VecReg);
            let extend = groups.len() > 2
                && rng.random_bool(0.35)
                && !(masked && consumer == anchor_id && output == source)
                && families.contains(&Family::VectorCompute);
            if extend {
                let input = groups.pop().expect("spare source group exists");
                let (line, extra) = sampled_alu(output, input, 13, false, cfg, elen, groups, rng)?;
                let node = graph.node(
                    Family::VectorCompute,
                    vec![line],
                    [
                        vec![StateInstance::vector(input)],
                        extra.map_or_else(Vec::new, |base| vec![StateInstance::vector(base)]),
                    ]
                    .concat(),
                    vec![StateInstance::vector(output)],
                );
                let id = graph.before(consumer, node, StateInstance::vector(output));
                unresolved.push((id, input));
                if let Some(extra) = extra {
                    unresolved.push((id, extra));
                }
            } else {
                if !families.contains(&Family::Load) {
                    return Err(error("load producer missing from reverse index"));
                }
                insert_load_before(&mut graph, consumer, output, cfg, source_addr, rng);
            }
        } else if let Some((producer, input)) = frontier {
            let families = index.consumers(StateKind::VecReg);
            let reduction =
                groups.len() >= 2 && rng.random_bool(0.15) && families.contains(&Family::Reduction);
            let compute = !groups.is_empty()
                && rng.random_bool(0.55)
                && families.contains(&Family::VectorCompute);
            if reduction {
                let accumulator = groups.pop().expect("reduction accumulator group exists");
                let output = groups.pop().expect("reduction output group exists");
                let node = graph.node(
                    Family::Reduction,
                    vec![format!("vredsum.vs v{output}, v{input}, v{accumulator}")],
                    vec![
                        StateInstance::vector(input),
                        StateInstance::vector(accumulator),
                        StateInstance::vector(output),
                    ],
                    vec![StateInstance::vector(output)],
                );
                let reduction_id = graph.after(producer, node, StateInstance::vector(input));
                let zero = graph.node(
                    Family::Move,
                    vec![format!("vmv.v.i v{accumulator}, 0")],
                    vec![StateInstance {
                        kind: StateKind::VCtrl,
                        selector: "vl/vtype/vstart".into(),
                    }],
                    vec![StateInstance::vector(accumulator)],
                );
                graph.before(reduction_id, zero, StateInstance::vector(accumulator));
                let output_zero = graph.node(
                    Family::Move,
                    vec![format!("vmv.v.i v{output}, 0")],
                    vec![],
                    vec![StateInstance::vector(output)],
                );
                graph.before(reduction_id, output_zero, StateInstance::vector(output));
                frontier = Some((reduction_id, output));
            } else if compute {
                let output = groups.pop().expect("spare destination group exists");
                let compress = masked && rng.random_bool(0.25);
                let (line, extra) = if compress {
                    (format!("vcompress.vm v{output}, v{input}, v0"), None)
                } else {
                    sampled_alu(output, input, 13, false, cfg, elen, groups, rng)?
                };
                let mut consumed = vec![StateInstance::vector(input)];
                if let Some(extra) = extra {
                    consumed.push(StateInstance::vector(extra));
                }
                if compress {
                    consumed.push(StateInstance {
                        kind: StateKind::Mask,
                        selector: "v0".into(),
                    });
                    consumed.push(StateInstance::vector(output));
                }
                let node = graph.node(
                    Family::VectorCompute,
                    vec![line],
                    consumed,
                    vec![StateInstance::vector(output)],
                );
                let id = graph.after(producer, node, StateInstance::vector(input));
                if compress {
                    let initialization = graph.node(
                        Family::Move,
                        vec![format!("vmv.v.i v{output}, 0")],
                        vec![],
                        vec![StateInstance::vector(output)],
                    );
                    graph.before(id, initialization, StateInstance::vector(output));
                }
                if let Some(extra) = extra {
                    unresolved.push((id, extra));
                }
                frontier = Some((id, output));
            } else {
                if fp_enabled
                    && matches!(cfg.sew, 32 | 64)
                    && index.consumers(StateKind::Fpr).contains(&Family::Move)
                    && rng.random_bool(0.25)
                {
                    let fp = graph.node(
                        Family::Move,
                        vec![format!("vfmv.f.s f16, v{input}")],
                        vec![StateInstance::vector(input)],
                        vec![StateInstance {
                            kind: StateKind::Fpr,
                            selector: "f16".into(),
                        }],
                    );
                    let fp_id = graph.after(producer, fp, StateInstance::vector(input));
                    let transfer = if cfg.sew == 32 { "fmv.x.w" } else { "fmv.x.d" };
                    let observer = graph.node(
                        Family::Move,
                        vec![
                            format!("{transfer} x18, f16"),
                            "sd x18, 64(x12)".into(),
                            "sd x18, 72(x12)".into(),
                        ],
                        vec![StateInstance {
                            kind: StateKind::Fpr,
                            selector: "f16".into(),
                        }],
                        vec![StateInstance {
                            kind: StateKind::Gpr,
                            selector: "x18".into(),
                        }],
                    );
                    graph.after(
                        fp_id,
                        observer,
                        StateInstance {
                            kind: StateKind::Fpr,
                            selector: "f16".into(),
                        },
                    );
                    frontier = None;
                    continue;
                }
                let store = rng.random_bool(0.6) && families.contains(&Family::Store);
                if !store && !families.contains(&Family::Move) {
                    return Err(error("move consumer missing from forward index"));
                }
                insert_terminal_observer(&mut graph, producer, input, cfg, output_addr, store, rng);
                frontier = None;
            }
        }
    }
    while let Some((consumer, output)) = unresolved.pop() {
        insert_load_before(&mut graph, consumer, output, cfg, source_addr, rng);
    }
    if masked {
        if !index
            .producers(StateKind::Mask)
            .contains(&Family::VectorCompute)
        {
            return Err(error("mask producer missing from reverse index"));
        }
        let producer = graph
            .nodes
            .iter()
            .rev()
            .find(|node| {
                node.produces.iter().any(|state| {
                    state.kind == StateKind::VecReg && state.selector == format!("v{source}")
                })
            })
            .map(|node| node.id)
            .ok_or_else(|| error("mask source producer is missing"))?;
        let mask = graph.node(
            Family::VectorCompute,
            vec![format!("vmseq.vi v0, v{source}, 0")],
            vec![StateInstance::vector(source)],
            vec![StateInstance {
                kind: StateKind::Mask,
                selector: "v0".into(),
            }],
        );
        let mask_id = graph.before(
            anchor_id,
            mask,
            StateInstance {
                kind: StateKind::Mask,
                selector: "v0".into(),
            },
        );
        graph.edges.push(GraphEdge {
            producer,
            consumer: mask_id,
            state: StateInstance::vector(source),
        });
        let other_mask_consumers = graph
            .nodes
            .iter()
            .filter(|node| {
                node.id != anchor_id
                    && node.id != mask_id
                    && node
                        .consumes
                        .iter()
                        .any(|state| state.kind == StateKind::Mask && state.selector == "v0")
            })
            .map(|node| node.id)
            .collect::<Vec<_>>();
        for consumer in other_mask_consumers {
            graph.edges.push(GraphEdge {
                producer: mask_id,
                consumer,
                state: StateInstance {
                    kind: StateKind::Mask,
                    selector: "v0".into(),
                },
            });
        }
    }
    if let Some((producer, input)) = frontier {
        insert_terminal_observer(&mut graph, producer, input, cfg, output_addr, true, rng);
    }
    if rng.random_bool(0.25) {
        append_vstart_flow(&mut graph, target, cfg, output_addr);
    }
    let mut current_control_producer = graph.nodes[0].id;
    let control = StateInstance {
        kind: StateKind::VCtrl,
        selector: "vl/vtype/vstart".into(),
    };
    let mut control_edges = Vec::new();
    for node in graph.nodes.iter_mut().skip(1) {
        if node.lines.iter().any(|line| line.starts_with('v')) {
            if !node
                .consumes
                .iter()
                .any(|state| state.kind == StateKind::VCtrl)
            {
                node.consumes.push(control.clone());
            }
            control_edges.push((current_control_producer, node.id));
        }
        if node
            .produces
            .iter()
            .any(|state| state.kind == StateKind::VCtrl)
        {
            current_control_producer = node.id;
        }
    }
    for (producer, consumer) in control_edges {
        graph.edges.push(GraphEdge {
            producer,
            consumer,
            state: control.clone(),
        });
    }
    if !graph.valid_edges() {
        return Err(error("state-flow edge lost during bidirectional insertion"));
    }
    Ok(graph)
}

pub fn generate(
    core_ids: &[String],
    mut config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    if config.isa_base != ISABase::Rv64 || !extensions.contains_rv64(RV64Extensions::Zvl32b) {
        return Err(error("artifact graph needs RV64 base-V"));
    }
    if config.hart_count() != 1 {
        return Err(error("artifact graph supports one hart"));
    }
    let subject = core_ids.first().ok_or_else(|| error("missing subject"))?;
    let profile = RvvSamplingProfile::from_subject_and_env(subject).map_err(error)?;
    config.mem_size = config.mem_size.max(minimum_memory_size(profile));
    let fp_enabled = extensions.contains_rv64(RV64Extensions::D);
    let mut testcase = crate::impls::generate_model_skeleton_with_compact_initialization(
        core_ids, config, extensions,
    )?;
    crate::rvv_stateflow::clear_ambient_test_instructions(&mut testcase);
    let setup_prefix_len = if subject.eq_ignore_ascii_case("ara") {
        crate::rvv_stateflow::inject_vector_state_enable(&mut testcase)?
    } else {
        0
    };
    let mut rng = StdRng::seed_from_u64(testcase.config.seed.unwrap_or(0) ^ 0x5443_4144_5346_5643);
    let vector_range = testcase
        .config
        .random_config
        .register_operand_config
        .vector_register_number_range;
    let (cfg, mut groups) = config_and_groups(profile, vector_range, &mut rng)?;
    let (start, end) = testcase
        .mem_range
        .get(subject)
        .copied()
        .ok_or_else(|| error("missing subject memory range"))?;
    if end.saturating_sub(start) < 0x180 {
        return Err(error("artifact graph needs 384 writable bytes"));
    }
    let source_addr = start.div_ceil(8) * 8;
    let output_addr = source_addr + 0x80;
    if output_addr + 0x80 > end {
        return Err(error("artifact graph output exceeds memory range"));
    }
    let mut lines = vec![
        format!("li x11, {source_addr}"),
        format!("li x12, {output_addr}"),
        format!("li x9, {}", cfg.sew / 4),
        format!("li x13, {}", rng.random_range(1..=15)),
    ];
    for offset in [0, 8, 16, 24, 32, 40, 48, 56] {
        let value = rng.random::<u64>();
        lines.push(format!(
            "li x15, {}",
            i64::from_ne_bytes(value.to_ne_bytes())
        ));
        lines.push(format!("sd x15, {offset}(x11)"));
    }
    let element_bytes = cfg.sew / 8;
    let element_store = match cfg.sew {
        8 => "sb",
        16 => "sh",
        32 => "sw",
        64 => "sd",
        _ => unreachable!(),
    };
    lines.push("li x15, 0".into());
    lines.push(format!("{element_store} x15, 0(x11)"));
    lines.push("li x15, 1".into());
    for offset in [element_bytes, element_bytes * 2] {
        lines.push(format!("{element_store} x15, {offset}(x11)"));
    }
    let graph = build_graph(
        cfg,
        profile.elen_bits,
        source_addr,
        output_addr,
        fp_enabled,
        &mut groups,
        &mut rng,
    )?;
    lines.extend(graph.rendered_lines());
    validate_vector_lines(&lines, cfg, profile.elen_bits)?;
    let instructions = lines
        .iter()
        .map(|line| decode(line))
        .collect::<Result<Vec<_>, _>>()?;
    testcase.harts[0]
        .sequences_mut(HartPhase::Test)
        .extend(instructions);
    crate::rvv_model_program::append_catalog_coverage_target(
        &mut testcase,
        setup_prefix_len,
        subject,
        crate::rvv_model_program::RvvModelProgramMode::RandomCatalog,
        false,
        Some(output_addr + 64),
    )?;
    let before_negative = testcase.harts[0].sequences(HartPhase::Test).len();
    crate::rvv_negative::append_expected_illegal_probe(&mut testcase)?;
    let negative_selected = testcase.harts[0].sequences(HartPhase::Test).len() > before_negative;
    if negative_selected {
        let test = testcase.harts[0].sequences_mut(HartPhase::Test);
        let negative = test.pop().expect("sampled negative probe was appended");
        if subject.eq_ignore_ascii_case("ara") {
            test.truncate(setup_prefix_len);
            test.push(negative);
            testcase.harts[0]
                .shared_sequence_indices_mut(HartPhase::Test)
                .retain(|index| *index < setup_prefix_len);
            for hart in &mut testcase.harts {
                hart.sequences_mut(HartPhase::Init).clear();
                hart.shared_sequence_indices_mut(HartPhase::Init).clear();
            }
        } else {
            test.insert(setup_prefix_len, negative);
        }
    }
    eprintln!(
        "RVV generated graph retained={} nodes={} edges={} families={:?} config=e{},{} vl={} seed={:?}",
        !negative_selected || !subject.eq_ignore_ascii_case("ara"),
        graph.nodes.len(),
        graph.edges.len(),
        graph
            .nodes
            .iter()
            .map(|node| node.family)
            .collect::<Vec<_>>(),
        cfg.sew,
        cfg.lmul,
        cfg.vl,
        testcase.config.seed
    );
    Ok(testcase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_window_contains_the_largest_whole_register_transfer() {
        assert_eq!(
            minimum_memory_size(RvvSamplingProfile::for_subject("ara")),
            4608
        );
        assert_eq!(
            minimum_memory_size(RvvSamplingProfile::for_subject("xiangshan")),
            1024
        );
    }

    #[test]
    fn state_edges_reject_intervening_vector_group_writes() {
        let mut graph = StateGraph {
            vector_group_width: 4,
            ..StateGraph::default()
        };
        let producer = graph.node(Family::Load, vec![], vec![], vec![StateInstance::vector(8)]);
        let consumer = graph.node(
            Family::Store,
            vec![],
            vec![StateInstance::vector(8)],
            vec![],
        );
        let producer_id = producer.id;
        let consumer_id = consumer.id;
        graph.nodes.extend([producer, consumer]);
        graph.edges.push(GraphEdge {
            producer: producer_id,
            consumer: consumer_id,
            state: StateInstance::vector(8),
        });
        assert!(graph.valid_edges());
        let clobber = graph.node(
            Family::VectorCompute,
            vec![],
            vec![],
            vec![StateInstance::vector(10)],
        );
        graph.nodes.insert(1, clobber);
        assert!(!graph.valid_edges());
    }

    #[test]
    fn state_edges_reject_mask_clobber_of_vector_zero() {
        let graph = StateGraph {
            vector_group_width: 1,
            ..StateGraph::default()
        };
        let mask = StateInstance {
            kind: StateKind::Mask,
            selector: "v0".into(),
        };
        assert!(graph.overwrites(&mask, &StateInstance::vector(0)));
        assert!(graph.overwrites(&StateInstance::vector(0), &mask));
        assert!(!graph.overwrites(&mask, &StateInstance::vector(1)));
    }

    #[test]
    fn family_index_looks_up_both_directions() {
        let index = FamilyIndex;
        assert!(index.producers(StateKind::VecReg).contains(&Family::Load));
        assert!(index.consumers(StateKind::VecReg).contains(&Family::Store));
        assert!(
            index
                .producers(StateKind::Mask)
                .contains(&Family::VectorCompute)
        );
        assert!(
            index
                .consumers(StateKind::Mask)
                .contains(&Family::VectorCompute)
        );
    }

    #[test]
    fn configuration_sampling_keeps_policy_range_and_reaches_lmul_eight() {
        let mut lmul_eight = false;
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (cfg, groups) =
                config_and_groups(RvvSamplingProfile::for_subject("ara"), (1, 31), &mut rng)
                    .expect("Ara profile has a graph VC domain");
            lmul_eight |= cfg.lmul == "m8";
            assert!((2..=4).contains(&cfg.vl));
            assert!(groups.len() >= 2);
            assert!(groups.iter().all(|base| (1..=31).contains(base)));
        }
        assert!(lmul_eight);
    }

    #[test]
    fn inserted_relations_preserve_concrete_state_edges() {
        let mut compressed = 0;
        let mut reductions = 0;
        let mut fp_moves = 0;
        let mut strided_loads = 0;
        let mut fof_loads = 0;
        let mut strided_stores = 0;
        let mut vstart_flows = 0;
        for (sew, lmul, vl) in [
            (8, "m1", 4),
            (16, "m2", 4),
            (32, "m1", 4),
            (32, "m4", 4),
            (64, "m1", 2),
            (64, "m8", 4),
        ] {
            let cfg = RvvConfigConstraint::new(ISABase::Rv64, sew, vl, lmul, "tu", "mu");
            let width = rvv_legal::register_width(sew, lmul, sew).unwrap();
            for seed in 0..64 {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut groups = rvv_legal::legal_group_bases((1, 31), width);
                groups.shuffle(&mut rng);
                let graph = build_graph(
                    cfg,
                    64,
                    0x8001_0000,
                    0x8001_0080,
                    true,
                    &mut groups,
                    &mut rng,
                )
                .expect("bidirectional relation binds valid state instances");
                assert!(graph.valid_edges());
                let lines = graph.rendered_lines();
                for node in &graph.nodes {
                    let Some(line) = node.lines.first() else {
                        continue;
                    };
                    if !(line.contains(", v0.t")
                        || line.starts_with("vcompress.vm")
                        || line.starts_with("vredsum.vs"))
                    {
                        continue;
                    }
                    let destination = line
                        .split_whitespace()
                        .nth(1)
                        .expect("vector destination")
                        .trim_end_matches(',');
                    let state = StateInstance {
                        kind: StateKind::VecReg,
                        selector: destination.to_string(),
                    };
                    assert!(node.consumes.contains(&state));
                    assert!(graph.edges.iter().any(|edge| {
                        edge.consumer == node.id
                            && edge.state == state
                            && graph.nodes.iter().any(|producer| {
                                producer.id == edge.producer
                                    && producer
                                        .lines
                                        .iter()
                                        .any(|line| line == &format!("vmv.v.i {destination}, 0"))
                            })
                    }));
                }
                compressed += lines
                    .iter()
                    .filter(|line| line.starts_with("vcompress.vm"))
                    .count();
                reductions += lines
                    .iter()
                    .filter(|line| line.starts_with("vredsum.vs"))
                    .count();
                fp_moves += lines
                    .iter()
                    .filter(|line| line.starts_with("vfmv.f.s"))
                    .count();
                strided_loads += lines.iter().filter(|line| line.starts_with("vlse")).count();
                fof_loads += lines
                    .iter()
                    .filter(|line| line.starts_with("vle") && line.contains("ff.v"))
                    .count();
                if lines
                    .iter()
                    .any(|line| line.starts_with("vle") && line.contains("ff.v"))
                {
                    assert!(lines.iter().any(|line| line == "csrr x19, vl"));
                }
                strided_stores += lines.iter().filter(|line| line.starts_with("vsse")).count();
                if lines.iter().any(|line| line == "csrrw x0, vstart, x20") {
                    vstart_flows += 1;
                    assert!(lines.iter().any(|line| line == "csrrs x21, vstart, x0"));
                    assert!(lines.iter().any(|line| line == "sd x22, 96(x12)"));
                }
                assert!(graph.nodes.iter().any(|node| node.family == Family::Load));
                assert!(
                    graph
                        .nodes
                        .iter()
                        .any(|node| node.family == Family::Store || node.family == Family::Move)
                );
                validate_vector_lines(&lines, cfg, 64)
                    .expect("VC rules accept bound vector instructions");
                for line in &lines {
                    decode(line).expect("generated graph instruction decodes");
                }
            }
        }
        assert!(compressed > 0, "mask-consuming compression must be sampled");
        assert!(reductions > 0, "reduction family must be sampled");
        assert!(fp_moves > 0, "FPR state must be sampled");
        assert!(
            strided_loads > 0,
            "strided memory producers must be sampled"
        );
        assert!(fof_loads > 0, "FOF memory producers must be sampled");
        assert!(
            strided_stores > 0,
            "strided memory observers must be sampled"
        );
        assert!(vstart_flows > 0, "vstart dataflow must be sampled");
    }
}
