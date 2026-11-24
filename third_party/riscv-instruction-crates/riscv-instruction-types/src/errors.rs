use std::fmt;

#[derive(Debug, Clone)]
pub struct RandomGenerationError {
    ty: &'static str,
    message: String,
}

impl RandomGenerationError {
    /// Create a new error with the given type name and message.
    pub fn new(ty: &'static str, message: impl Into<String>) -> Self {
        Self {
            ty,
            message: message.into(),
        }
    }

    /// Convenience helper for errors caused by exhausting retries while respecting constraints.
    pub fn exhausted(ty: &'static str, attempts: usize, last_error: Option<String>) -> Self {
        let mut msg = format!("exhausted {attempts} attempts while generating a valid value");
        if let Some(err) = last_error {
            if !err.is_empty() {
                msg.push_str(": ");
                msg.push_str(&err);
            }
        }
        Self::new(ty, msg)
    }

    /// Attach additional context to the error message.
    pub fn with_context(self, context: impl Into<String>) -> Self {
        let mut msg = self.message;
        let context: String = context.into();
        if !context.is_empty() {
            if !msg.is_empty() {
                msg.push_str("; ");
            }
            msg.push_str(&context);
        }
        Self {
            message: msg,
            ..self
        }
    }

    /// Returns the type name associated with this error.
    pub fn ty(&self) -> &'static str {
        self.ty
    }
}

impl fmt::Display for RandomGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            write!(f, "failed to generate random {}", self.ty)
        } else {
            write!(f, "failed to generate random {}: {}", self.ty, self.message)
        }
    }
}

impl std::error::Error for RandomGenerationError {}

/// Error returned when parsing canonical assembly text back into a typed model fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsmParseErrorKind {
    EmptyInput,
    MnemonicMismatch,
    SyntaxMismatch,
    InvalidOperand,
    InvalidInstruction,
}

/// Structured assembly parsing error with lightweight category information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsmParseError {
    ty: &'static str,
    kind: AsmParseErrorKind,
    message: String,
}

impl AsmParseError {
    pub fn new(ty: &'static str, kind: AsmParseErrorKind, message: impl Into<String>) -> Self {
        Self {
            ty,
            kind,
            message: message.into(),
        }
    }

    pub fn empty_input(ty: &'static str) -> Self {
        Self::new(ty, AsmParseErrorKind::EmptyInput, "assembly text is empty")
    }

    pub fn mnemonic_mismatch(ty: &'static str, expected: &str, actual: &str) -> Self {
        Self::new(
            ty,
            AsmParseErrorKind::MnemonicMismatch,
            format!("expected mnemonic `{expected}`, got `{actual}`"),
        )
    }

    pub fn syntax_mismatch(ty: &'static str, message: impl Into<String>) -> Self {
        Self::new(ty, AsmParseErrorKind::SyntaxMismatch, message)
    }

    pub fn invalid_operand(ty: &'static str, message: impl Into<String>) -> Self {
        Self::new(ty, AsmParseErrorKind::InvalidOperand, message)
    }

    pub fn invalid_instruction(ty: &'static str, message: impl Into<String>) -> Self {
        Self::new(ty, AsmParseErrorKind::InvalidInstruction, message)
    }

    pub fn with_context(self, context: impl Into<String>) -> Self {
        let context: String = context.into();
        if context.is_empty() {
            return self;
        }

        let mut message = self.message;
        if !message.is_empty() {
            message.push_str("; ");
        }
        message.push_str(&context);

        Self { message, ..self }
    }

    pub fn ty(&self) -> &'static str {
        self.ty
    }

    pub fn kind(&self) -> AsmParseErrorKind {
        self.kind
    }

    pub fn is_mnemonic_mismatch(&self) -> bool {
        matches!(self.kind, AsmParseErrorKind::MnemonicMismatch)
    }
}

impl fmt::Display for AsmParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            write!(f, "failed to parse asm {}", self.ty)
        } else {
            write!(f, "failed to parse asm {}: {}", self.ty, self.message)
        }
    }
}

impl std::error::Error for AsmParseError {}
