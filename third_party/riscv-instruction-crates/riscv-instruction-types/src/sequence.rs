use serde::{Deserialize, Serialize};

use crate::{
    AsmDecode, AsmDecodeWithContext, AsmEncode, AsmParseError, IsaDecodeContext,
    InstructionSequenceItem, IsaProfile, PseudoInstruction, RandomGenerationError,
};

/// Instruction sequence containing zero or more pre instructions followed by a single actual instruction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionSequence<T> {
    /// Vector of pre instructions executed before the main instruction (can be empty)
    pub pre_instructions: Vec<T>,
    /// The final actual instruction
    pub instruction: T,
    /// Vector of post instructions executed after the main instruction
    pub post_instructions: Vec<T>,
}

impl<T> InstructionSequence<T> {
    pub fn new(instruction: T) -> Self {
        Self {
            pre_instructions: Vec::new(),
            instruction,
            post_instructions: Vec::new(),
        }
    }

    pub fn try_for_each_pseudo_instruction<E, F>(&self, mut f: F) -> Result<(), E>
    where
        T: InstructionSequenceItem,
        F: FnMut(&PseudoInstruction) -> Result<(), E>,
    {
        for instruction in &self.pre_instructions {
            if let Some(pseudo) = instruction.as_pseudo_instruction() {
                f(pseudo)?;
            }
        }
        if let Some(pseudo) = self.instruction.as_pseudo_instruction() {
            f(pseudo)?;
        }
        for instruction in &self.post_instructions {
            if let Some(pseudo) = instruction.as_pseudo_instruction() {
                f(pseudo)?;
            }
        }
        Ok(())
    }

    pub fn validate_for_profile(&self, profile: IsaProfile) -> Result<(), RandomGenerationError>
    where
        T: InstructionSequenceItem,
    {
        self.try_for_each_pseudo_instruction(|pseudo| pseudo.validate_for_profile(profile))
    }

    pub fn validate_asm_decode_context(&self, ctx: IsaDecodeContext) -> Result<(), RandomGenerationError>
    where
        T: InstructionSequenceItem,
    {
        self.try_for_each_pseudo_instruction(|pseudo| pseudo.validate_asm_decode_context(ctx))
    }
}

impl<T: AsmEncode> AsmEncode for InstructionSequence<T> {
    fn encode_asm(&self) -> String {
        let mut out = String::new();
        for instruction in &self.pre_instructions {
            out.push_str(&instruction.encode_asm());
            out.push('\n');
        }
        out.push_str(&self.instruction.encode_asm());
        if !self.post_instructions.is_empty() {
            for instruction in &self.post_instructions {
                out.push('\n');
                out.push_str(&instruction.encode_asm());
            }
        }
        out
    }
}

impl<T: AsmEncode> std::fmt::Display for InstructionSequence<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.encode_asm())
    }
}

impl<T> AsmDecode for InstructionSequence<T>
where
    T: AsmDecode + InstructionSequenceItem,
{
    fn decode_asm(input: &str) -> Result<Self, AsmParseError> {
        decode_sequence_with(input, |line| T::decode_asm(line))
    }
}

impl<T> std::str::FromStr for InstructionSequence<T>
where
    T: AsmDecode + InstructionSequenceItem,
{
    type Err = AsmParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        <Self as AsmDecode>::decode_asm(s)
    }
}

impl<T> AsmDecodeWithContext for InstructionSequence<T>
where
    T: AsmDecodeWithContext + InstructionSequenceItem,
{
    fn decode_asm_with_context(input: &str, ctx: IsaDecodeContext) -> Result<Self, AsmParseError> {
        let sequence = decode_sequence_with(input, |line| T::decode_asm_with_context(line, ctx))?;
        sequence
            .validate_asm_decode_context(ctx)
            .map_err(|err| AsmParseError::invalid_instruction("InstructionSequence", err.to_string()))?;
        Ok(sequence)
    }
}

fn decode_sequence_with<T, F>(input: &str, mut decode: F) -> Result<InstructionSequence<T>, AsmParseError>
where
    T: InstructionSequenceItem,
    F: FnMut(&str) -> Result<T, AsmParseError>,
{
    let lines: Vec<_> = input
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();

    if lines.is_empty() {
        return Err(AsmParseError::empty_input("InstructionSequence"));
    }

    if lines.len() == 1 {
        return Ok(InstructionSequence {
            pre_instructions: Vec::new(),
            instruction: decode(lines[0])?,
            post_instructions: Vec::new(),
        });
    }

    // Decode all lines upfront using the same decode function
    let decoded: Vec<T> = lines
        .iter()
        .map(|line| decode(line))
        .collect::<Result<Vec<_>, _>>()?;

    let mut matches = Vec::new();

    for main_idx in 0..decoded.len() {
        if decoded[main_idx].is_pseudo_instruction() {
            continue;
        }

        let pre_all_pseudo = decoded[..main_idx]
            .iter()
            .all(InstructionSequenceItem::is_pseudo_instruction);
        if !pre_all_pseudo {
            continue;
        }

        let post_all_pseudo = decoded[main_idx + 1..]
            .iter()
            .all(InstructionSequenceItem::is_pseudo_instruction);
        if !post_all_pseudo {
            continue;
        }

        matches.push(main_idx);
    }

    match matches.len() {
        1 => {
            let main_idx = matches[0];
            // We need to consume decoded — rebuild by splitting
            let mut decoded = decoded;
            let post_instructions: Vec<T> = decoded.drain(main_idx + 1..).collect();
            let instruction = decoded.remove(main_idx);
            let pre_instructions = decoded;
            Ok(InstructionSequence {
                pre_instructions,
                instruction,
                post_instructions,
            })
        }
        0 => Err(AsmParseError::invalid_instruction(
            "InstructionSequence",
            "could not identify a unique main instruction line in the provided multi-line asm",
        )),
        count => Err(AsmParseError::invalid_instruction(
            "InstructionSequence",
            format!(
                "found {count} possible main instruction lines; multi-line asm is ambiguous"
            ),
        )),
    }
}
