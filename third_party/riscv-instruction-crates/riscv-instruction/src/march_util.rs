/// Metadata interface that lets generated extension enums expose dependency hints to the `march` builder.
pub trait MarchExtension {
    /// Returns unified-db-derived companion / required extension names.
    fn required_extension_names(&self) -> &'static [&'static str] {
        &[]
    }
}

#[macro_export]
macro_rules! generate_assemble_march {
    () => {
        use riscv_instruction_types::CsrSelectionConfig as TypesCsrSelectionConfig;
        use std::collections::BTreeSet;

        #[derive(Default)]
        struct CsrDomainFlags {
            supervisor: bool,
            hypervisor: bool,
            virtual_supervisor: bool,
            rnmi: bool,
            floating: bool,
            vector: bool,
            debug: bool,
        }

        impl CsrDomainFlags {
            fn finalize(mut self) -> TypesCsrSelectionConfig {
                if self.hypervisor {
                    self.supervisor = true;
                    self.virtual_supervisor = true;
                }
                TypesCsrSelectionConfig {
                    machine_csr_operands_enabled: true,
                    supervisor_csr_operands_enabled: self.supervisor,
                    hypervisor_csr_operands_enabled: self.hypervisor,
                    virtual_supervisor_csr_operands_enabled: self.virtual_supervisor,
                    rnmi_csr_operands_enabled: self.rnmi,
                    floating_csr_operands_enabled: self.floating,
                    vector_csr_operands_enabled: self.vector,
                    debug_csr_operands_enabled: self.debug,
                }
            }
        }

        #[derive(Clone, Copy)]
        enum MarchCompatAction {
            InsertStd(char),
            InsertOther(&'static str),
        }

        /// Primary representation only models how the extension itself should appear in `-march`.
        #[derive(Clone, Copy)]
        enum PrimaryMarchAction {
            SelfLowercase,
            IntegerBase,
            FloatingF,
            FloatingD,
            FloatingQ,
            BaseCompressed,
            VectorBase,
            InsertStd(char),
            InsertOther(&'static str),
            Ignore,
            CompressedFamilyLowercase,
        }

        /// CSR hints describe side effects that do not necessarily appear in the textual `-march`.
        #[derive(Clone, Copy)]
        struct CsrDomainHint {
            supervisor: bool,
            hypervisor: bool,
            rnmi: bool,
            debug: bool,
        }

        impl CsrDomainHint {
            const NONE: Self = Self {
                supervisor: false,
                hypervisor: false,
                rnmi: false,
                debug: false,
            };

            const SUPERVISOR: Self = Self {
                supervisor: true,
                hypervisor: false,
                rnmi: false,
                debug: false,
            };

            const HYPERVISOR: Self = Self {
                supervisor: false,
                hypervisor: true,
                rnmi: false,
                debug: false,
            };

            const RNMI: Self = Self {
                supervisor: false,
                hypervisor: false,
                rnmi: true,
                debug: false,
            };

            const SUPERVISOR_DEBUG: Self = Self {
                supervisor: true,
                hypervisor: false,
                rnmi: false,
                debug: true,
            };
        }

        /// A named rule keeps textual representation, CSR hints, and GNU compatibility shims together.
        struct NamedExtensionRule {
            extension: &'static str,
            primary_action: PrimaryMarchAction,
            csr_hint: CsrDomainHint,
            compat_common: &'static [MarchCompatAction],
            compat_rv32_only: &'static [MarchCompatAction],
            compat_rv64_only: &'static [MarchCompatAction],
        }

        const GNU_COMPAT_B_COMMON: &[MarchCompatAction] = &[
            MarchCompatAction::InsertOther("zba"),
            MarchCompatAction::InsertOther("zbb"),
            MarchCompatAction::InsertOther("zbc"),
            MarchCompatAction::InsertOther("zbs"),
        ];

        const GNU_COMPAT_A_FAMILY_COMMON: &[MarchCompatAction] =
            &[MarchCompatAction::InsertStd('a')];

        const GNU_COMPAT_ZABHA_COMMON: &[MarchCompatAction] = &[
            MarchCompatAction::InsertStd('a'),
            MarchCompatAction::InsertOther("zacas"),
        ];

        const GNU_COMPAT_ZCB_COMMON: &[MarchCompatAction] = &[
            MarchCompatAction::InsertStd('m'),
            MarchCompatAction::InsertOther("zbb"),
        ];
        const GNU_COMPAT_ZCB_RV64: &[MarchCompatAction] = &[MarchCompatAction::InsertOther("zba")];

        const GNU_COMPAT_ZCMOP_RV32: &[MarchCompatAction] = &[
            MarchCompatAction::InsertStd('a'),
            MarchCompatAction::InsertOther("zacas"),
        ];

        const GNU_COMPAT_SMDBLTRP_COMMON: &[MarchCompatAction] =
            &[MarchCompatAction::InsertOther("smctr")];

        const NAMED_EXTENSION_RULES: &[NamedExtensionRule] = &[
            NamedExtensionRule {
                extension: "I",
                primary_action: PrimaryMarchAction::IntegerBase,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "M",
                primary_action: PrimaryMarchAction::InsertStd('m'),
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "A",
                primary_action: PrimaryMarchAction::InsertStd('a'),
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "F",
                primary_action: PrimaryMarchAction::FloatingF,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "D",
                primary_action: PrimaryMarchAction::FloatingD,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Q",
                primary_action: PrimaryMarchAction::FloatingQ,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "C",
                primary_action: PrimaryMarchAction::BaseCompressed,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "V",
                primary_action: PrimaryMarchAction::VectorBase,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "U",
                primary_action: PrimaryMarchAction::Ignore,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "S",
                primary_action: PrimaryMarchAction::Ignore,
                csr_hint: CsrDomainHint::SUPERVISOR,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Sm",
                primary_action: PrimaryMarchAction::Ignore,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "B",
                primary_action: PrimaryMarchAction::Ignore,
                csr_hint: CsrDomainHint::NONE,
                compat_common: GNU_COMPAT_B_COMMON,
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Sdext",
                primary_action: PrimaryMarchAction::Ignore,
                csr_hint: CsrDomainHint::SUPERVISOR_DEBUG,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zalrsc",
                primary_action: PrimaryMarchAction::Ignore,
                csr_hint: CsrDomainHint::NONE,
                compat_common: GNU_COMPAT_A_FAMILY_COMMON,
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "H",
                primary_action: PrimaryMarchAction::InsertOther("h"),
                csr_hint: CsrDomainHint::HYPERVISOR,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zaamo",
                primary_action: PrimaryMarchAction::InsertOther("zaamo"),
                csr_hint: CsrDomainHint::NONE,
                compat_common: GNU_COMPAT_A_FAMILY_COMMON,
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zacas",
                primary_action: PrimaryMarchAction::InsertOther("zacas"),
                csr_hint: CsrDomainHint::NONE,
                compat_common: GNU_COMPAT_A_FAMILY_COMMON,
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zalasr",
                primary_action: PrimaryMarchAction::InsertOther("zalasr"),
                csr_hint: CsrDomainHint::NONE,
                compat_common: GNU_COMPAT_A_FAMILY_COMMON,
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zabha",
                primary_action: PrimaryMarchAction::InsertOther("zabha"),
                csr_hint: CsrDomainHint::NONE,
                compat_common: GNU_COMPAT_ZABHA_COMMON,
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zcb",
                primary_action: PrimaryMarchAction::CompressedFamilyLowercase,
                csr_hint: CsrDomainHint::NONE,
                compat_common: GNU_COMPAT_ZCB_COMMON,
                compat_rv32_only: &[],
                compat_rv64_only: GNU_COMPAT_ZCB_RV64,
            },
            NamedExtensionRule {
                extension: "Zcmp",
                primary_action: PrimaryMarchAction::CompressedFamilyLowercase,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zcmt",
                primary_action: PrimaryMarchAction::CompressedFamilyLowercase,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zcmop",
                primary_action: PrimaryMarchAction::CompressedFamilyLowercase,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: GNU_COMPAT_ZCMOP_RV32,
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zcd",
                primary_action: PrimaryMarchAction::CompressedFamilyLowercase,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zcf",
                primary_action: PrimaryMarchAction::CompressedFamilyLowercase,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Zihintntl",
                primary_action: PrimaryMarchAction::CompressedFamilyLowercase,
                csr_hint: CsrDomainHint::NONE,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Svinval",
                primary_action: PrimaryMarchAction::SelfLowercase,
                csr_hint: CsrDomainHint::SUPERVISOR,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Smdbltrp",
                primary_action: PrimaryMarchAction::SelfLowercase,
                csr_hint: CsrDomainHint::SUPERVISOR,
                compat_common: GNU_COMPAT_SMDBLTRP_COMMON,
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
            NamedExtensionRule {
                extension: "Smrnmi",
                primary_action: PrimaryMarchAction::SelfLowercase,
                csr_hint: CsrDomainHint::RNMI,
                compat_common: &[],
                compat_rv32_only: &[],
                compat_rv64_only: &[],
            },
        ];

        fn lookup_named_extension_rule(ext_name: &str) -> Option<&'static NamedExtensionRule> {
            NAMED_EXTENSION_RULES
                .iter()
                .find(|rule| rule.extension == ext_name)
        }

        /// Shared accumulator that derives both `-march` pieces and CSR domains in one pass.
        struct ExtensionBuildState {
            xlen: u32,
            std_exts: BTreeSet<char>,
            other_exts: BTreeSet<String>,
            csr_flags: CsrDomainFlags,
        }

        impl ExtensionBuildState {
            fn new(xlen: u32) -> Self {
                Self {
                    xlen,
                    std_exts: BTreeSet::new(),
                    other_exts: BTreeSet::new(),
                    csr_flags: CsrDomainFlags::default(),
                }
            }

            fn insert_std(&mut self, ext: char) {
                self.std_exts.insert(ext);
            }

            fn insert_other(&mut self, extension: &str) {
                self.other_exts.insert(extension.to_string());
            }

            fn insert_other_lowercase(&mut self, extension: &str) {
                self.other_exts.insert(extension.to_ascii_lowercase());
            }

            fn insert_other_many(&mut self, extensions: &[&str]) {
                for extension in extensions {
                    self.insert_other(extension);
                }
            }

            /// Only the base `I` extension should inject `zilsd`.
            fn enable_integer_base(&mut self) {
                self.insert_std('i');
                if self.xlen == 32 {
                    self.insert_other_many(&["zilsd"]);
                }
            }

            /// Base `C` itself only contributes the compressed ISA marker; companion subsets come from data-driven metadata.
            fn enable_base_compressed(&mut self) {
                self.insert_std('c');
            }

            /// Requires compressed encoding support without inheriting base-`C` extras.
            fn enable_compressed_family(&mut self) {
                self.insert_std('c');
            }

            fn enable_floating_f(&mut self) {
                self.insert_std('f');
                self.csr_flags.floating = true;
            }

            fn enable_floating_d(&mut self) {
                self.insert_std('d');
                self.enable_floating_f();
            }

            fn enable_floating_q(&mut self) {
                self.insert_std('q');
                self.enable_floating_d();
            }

            fn enable_vector(&mut self) {
                self.insert_std('v');
                self.csr_flags.vector = true;
            }

            fn supervisor_csr_operands_enabled(&mut self) {
                self.csr_flags.supervisor = true;
            }

            fn hypervisor_csr_operands_enabled(&mut self) {
                self.csr_flags.hypervisor = true;
            }

            fn rnmi_csr_operands_enabled(&mut self) {
                self.csr_flags.rnmi = true;
            }

            fn debug_csr_operands_enabled(&mut self) {
                self.csr_flags.debug = true;
            }

            fn apply_primary_action(&mut self, action: PrimaryMarchAction, ext_name: &str) {
                match action {
                    PrimaryMarchAction::SelfLowercase => self.insert_other_lowercase(ext_name),
                    PrimaryMarchAction::IntegerBase => self.enable_integer_base(),
                    PrimaryMarchAction::FloatingF => self.enable_floating_f(),
                    PrimaryMarchAction::FloatingD => self.enable_floating_d(),
                    PrimaryMarchAction::FloatingQ => self.enable_floating_q(),
                    PrimaryMarchAction::BaseCompressed => self.enable_base_compressed(),
                    PrimaryMarchAction::VectorBase => self.enable_vector(),
                    PrimaryMarchAction::InsertStd(std_ext) => self.insert_std(std_ext),
                    PrimaryMarchAction::InsertOther(other_ext) => self.insert_other(other_ext),
                    PrimaryMarchAction::Ignore => {}
                    PrimaryMarchAction::CompressedFamilyLowercase => {
                        self.enable_compressed_family();
                        self.insert_other_lowercase(ext_name);
                    }
                }
            }

            /// Applies the extension's direct `-march` representation before any GNU compatibility overlays.
            fn apply_primary_extension_representation(&mut self, ext_name: &str) {
                if let Some(rule) = lookup_named_extension_rule(ext_name) {
                    self.apply_primary_action(rule.primary_action, ext_name);
                    return;
                }

                if ext_name.starts_with("Zv") || ext_name.starts_with("Zve") {
                    self.enable_vector();
                    self.insert_other_lowercase(ext_name);
                    return;
                }

                self.insert_other_lowercase(ext_name);
            }

            /// Applies CSR-domain side effects that are orthogonal to the textual `-march` pieces.
            fn apply_csr_domain_hints(&mut self, ext_name: &str) {
                let Some(rule) = lookup_named_extension_rule(ext_name) else {
                    return;
                };

                if rule.csr_hint.supervisor {
                    self.supervisor_csr_operands_enabled();
                }
                if rule.csr_hint.hypervisor {
                    self.hypervisor_csr_operands_enabled();
                }
                if rule.csr_hint.rnmi {
                    self.rnmi_csr_operands_enabled();
                }
                if rule.csr_hint.debug {
                    self.debug_csr_operands_enabled();
                }
            }

            fn apply_compat_actions(&mut self, actions: &[MarchCompatAction]) {
                for action in actions {
                    self.apply_compat_action(*action);
                }
            }

            /// These overlays are the remaining GNU assembler compatibility shims that still need explicit handling.
            fn apply_gnu_compatibility_overlay(&mut self, ext_name: &str) {
                let Some(rule) = lookup_named_extension_rule(ext_name) else {
                    return;
                };

                self.apply_compat_actions(rule.compat_common);

                let arch_specific_actions = match self.xlen {
                    32 => rule.compat_rv32_only,
                    64 => rule.compat_rv64_only,
                    _ => &[],
                };

                self.apply_compat_actions(arch_specific_actions);
            }

            fn apply_compat_action(&mut self, action: MarchCompatAction) {
                match action {
                    MarchCompatAction::InsertStd(std_ext) => self.insert_std(std_ext),
                    MarchCompatAction::InsertOther(other_ext) => self.insert_other(other_ext),
                }
            }

            fn apply_extension(&mut self, ext_name: &str) {
                self.apply_primary_extension_representation(ext_name);
                self.apply_csr_domain_hints(ext_name);
                self.apply_gnu_compatibility_overlay(ext_name);
            }
        }

        /// Apply each extension reduction step at most once so an extension and its
        /// dependencies cannot inject the same `-march` fragment repeatedly.
        fn apply_extension_once(
            state: &mut ExtensionBuildState,
            applied_extensions: &mut BTreeSet<String>,
            extension_name: &str,
        ) {
            if applied_extensions.insert(extension_name.to_string()) {
                state.apply_extension(extension_name);
            }
        }

        impl ExtensionBuildState {
            fn into_march(self, base: &str) -> String {
                assemble_march(base, self.std_exts, self.other_exts)
            }

            fn into_csr_config(self) -> TypesCsrSelectionConfig {
                self.csr_flags.finalize()
            }
        }

        /// Assemble the collected extension components into the final `-march` string.
        fn assemble_march(
            base: &str,
            mut std_exts: BTreeSet<char>,
            other_exts: BTreeSet<String>,
        ) -> String {
            // If no extensions were requested, return only the base integer ISA marker `i`.
            if std_exts.is_empty() && other_exts.is_empty() {
                return format!("{}i", base);
            }

            // Any non-empty extension set still requires the base `i` extension.
            std_exts.insert('i');

            // Build the standard-extension segment in canonical order (IMAFDQCV).
            let mut std_str = String::new();
            let canonical_order = "imafdqcv";

            for ext_char in canonical_order.chars() {
                if std_exts.remove(&ext_char) {
                    std_str.push(ext_char);
                }
            }
            // Append any remaining standard extensions outside canonical order in sorted order.
            for ext_char in std_exts {
                std_str.push(ext_char);
            }

            let mut result = format!("{}{}", base, std_str);

            // Append all other extensions separated by underscores. `BTreeSet` keeps them sorted.
            if !other_exts.is_empty() {
                let other_str = other_exts.into_iter().collect::<Vec<String>>().join("_");
                result.push('_');
                result.push_str(&other_str);
            }

            result
        }

        fn collect_extension_state<T: std::fmt::Display + crate::march_util::MarchExtension>(
            xlen: u32,
            extensions: &[T],
        ) -> ExtensionBuildState {
            let mut state = ExtensionBuildState::new(xlen);
            let mut applied_extensions = BTreeSet::new();

            for ext in extensions {
                for dependency in ext.required_extension_names() {
                    apply_extension_once(&mut state, &mut applied_extensions, dependency);
                }

                let extension_name = ext.to_string();
                apply_extension_once(&mut state, &mut applied_extensions, &extension_name);
            }

            state
        }

        fn build_march_from_extensions<T: std::fmt::Display + crate::march_util::MarchExtension>(
            base: &str,
            xlen: u32,
            extensions: &[T],
        ) -> String {
            collect_extension_state(xlen, extensions).into_march(base)
        }

        fn csr_config_from_extensions<T: std::fmt::Display + crate::march_util::MarchExtension>(
            xlen: u32,
            extensions: &[T],
        ) -> TypesCsrSelectionConfig {
            collect_extension_state(xlen, extensions).into_csr_config()
        }

        /// Build a RISC-V `-march` string from the given `RV32Extensions` slice.
        pub fn build_rv32_march(extensions: &[RV32Extensions]) -> String {
            build_march_from_extensions("rv32", 32, extensions)
        }

        /// Derive the CSR-domain configuration implied by an `RV32Extensions` set.
        pub fn csr_config_from_rv32_extensions(
            extensions: &[RV32Extensions],
        ) -> TypesCsrSelectionConfig {
            csr_config_from_extensions(32, extensions)
        }

        /// Build a RISC-V `-march` string from the given `RV64Extensions` slice.
        pub fn build_rv64_march(extensions: &[RV64Extensions]) -> String {
            build_march_from_extensions("rv64", 64, extensions)
        }

        /// Derive the CSR-domain configuration implied by an `RV64Extensions` set.
        pub fn csr_config_from_rv64_extensions(
            extensions: &[RV64Extensions],
        ) -> TypesCsrSelectionConfig {
            csr_config_from_extensions(64, extensions)
        }
    };
}

#[cfg(test)]
mod tests {
    use std::fmt;

    #[derive(Clone, Copy)]
    pub(crate) enum RV32Extensions {
        Dummy,
    }

    impl fmt::Display for RV32Extensions {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Dummy => write!(f, "I"),
            }
        }
    }

    impl super::MarchExtension for RV32Extensions {}

    #[derive(Clone, Copy)]
    pub(crate) enum RV64Extensions {
        Dummy,
    }

    impl fmt::Display for RV64Extensions {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Dummy => write!(f, "I"),
            }
        }
    }

    impl super::MarchExtension for RV64Extensions {}

    crate::generate_assemble_march!();

    #[derive(Clone, Copy)]
    enum FakeMarchExtension {
        WeirdVector,
    }

    impl fmt::Display for FakeMarchExtension {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::WeirdVector => write!(f, "Zve32x-test"),
            }
        }
    }

    impl fmt::Debug for FakeMarchExtension {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::WeirdVector => write!(f, "Zve32x_test"),
            }
        }
    }

    impl super::MarchExtension for FakeMarchExtension {}

    #[derive(Clone, Copy)]
    enum FakeCsrExtension {
        DebugSupervisor,
    }

    impl fmt::Display for FakeCsrExtension {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::DebugSupervisor => write!(f, "Sdext"),
            }
        }
    }

    impl fmt::Debug for FakeCsrExtension {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::DebugSupervisor => write!(f, "Sdext_sanitized"),
            }
        }
    }

    impl super::MarchExtension for FakeCsrExtension {}

    fn build_fake_march(extensions: &[FakeMarchExtension]) -> String {
        build_march_from_extensions("rv32", 32, extensions)
    }

    fn build_fake_csr_march(extensions: &[FakeCsrExtension]) -> String {
        build_march_from_extensions("rv64", 64, extensions)
    }

    fn build_fake_csr_config(
        extensions: &[FakeCsrExtension],
    ) -> riscv_instruction_types::CsrSelectionConfig {
        csr_config_from_extensions(64, extensions)
    }

    #[derive(Clone, Copy)]
    enum FakeExtension {
        B,
        F,
        Q,
        Zaamo,
        Zabha,
        Zcb,
        Zcmop,
        Zalasr,
        H,
        Sm,
        Smdbltrp,
        Zvfbfmin,
    }

    impl fmt::Display for FakeExtension {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::B => write!(f, "B"),
                Self::F => write!(f, "F"),
                Self::Q => write!(f, "Q"),
                Self::Zaamo => write!(f, "Zaamo"),
                Self::Zabha => write!(f, "Zabha"),
                Self::Zcb => write!(f, "Zcb"),
                Self::Zcmop => write!(f, "Zcmop"),
                Self::Zalasr => write!(f, "Zalasr"),
                Self::H => write!(f, "H"),
                Self::Sm => write!(f, "Sm"),
                Self::Smdbltrp => write!(f, "Smdbltrp"),
                Self::Zvfbfmin => write!(f, "Zvfbfmin"),
            }
        }
    }

    impl fmt::Debug for FakeExtension {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::B => write!(f, "B_sanitized"),
                Self::F => write!(f, "F_sanitized"),
                Self::Q => write!(f, "Q_sanitized"),
                Self::Zaamo => write!(f, "Zaamo_sanitized"),
                Self::Zabha => write!(f, "Zabha_sanitized"),
                Self::Zcb => write!(f, "Zcb_sanitized"),
                Self::Zcmop => write!(f, "Zcmop_sanitized"),
                Self::Zalasr => write!(f, "Zalasr_sanitized"),
                Self::H => write!(f, "H_sanitized"),
                Self::Sm => write!(f, "Sm_sanitized"),
                Self::Smdbltrp => write!(f, "Smdbltrp_sanitized"),
                Self::Zvfbfmin => write!(f, "Zvfbfmin_sanitized"),
            }
        }
    }

    impl super::MarchExtension for FakeExtension {
        fn required_extension_names(&self) -> &'static [&'static str] {
            match self {
                Self::Zvfbfmin => &["F"],
                _ => &[],
            }
        }
    }

    #[derive(Clone, Copy)]
    enum FakeDataDrivenExtension {
        Demo,
    }

    impl fmt::Display for FakeDataDrivenExtension {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Demo => write!(f, "Zdemo"),
            }
        }
    }

    impl super::MarchExtension for FakeDataDrivenExtension {
        fn required_extension_names(&self) -> &'static [&'static str] {
            match self {
                Self::Demo => &["M", "Zbb"],
            }
        }
    }

    fn build_fake_rv32_march(extensions: &[FakeExtension]) -> String {
        build_march_from_extensions("rv32", 32, extensions)
    }

    fn build_fake_rv64_march(extensions: &[FakeExtension]) -> String {
        build_march_from_extensions("rv64", 64, extensions)
    }

    fn build_fake_rv64_csr_config(
        extensions: &[FakeExtension],
    ) -> riscv_instruction_types::CsrSelectionConfig {
        csr_config_from_extensions(64, extensions)
    }

    fn build_fake_data_driven_march(extensions: &[FakeDataDrivenExtension]) -> String {
        build_march_from_extensions("rv64", 64, extensions)
    }

    #[test]
    fn build_march_from_extensions_uses_display_names() {
        assert_eq!(
            build_fake_march(&[FakeMarchExtension::WeirdVector]),
            "rv32iv_zve32x-test"
        );
    }

    #[test]
    fn csr_config_from_extensions_uses_display_names() {
        let config = build_fake_csr_config(&[FakeCsrExtension::DebugSupervisor]);

        assert!(config.machine_csr_operands_enabled);
        assert!(config.supervisor_csr_operands_enabled);
        assert!(config.debug_csr_operands_enabled);
        assert_eq!(
            build_fake_csr_march(&[FakeCsrExtension::DebugSupervisor]),
            "rv64i"
        );
    }

    #[test]
    fn generated_wrapper_functions_smoke_test() {
        assert_eq!(build_rv32_march(&[RV32Extensions::Dummy]), "rv32i_zilsd");
        assert_eq!(build_rv64_march(&[RV64Extensions::Dummy]), "rv64i");

        let rv32_csr = csr_config_from_rv32_extensions(&[RV32Extensions::Dummy]);
        let rv64_csr = csr_config_from_rv64_extensions(&[RV64Extensions::Dummy]);

        assert!(rv32_csr.machine_csr_operands_enabled);
        assert!(rv64_csr.machine_csr_operands_enabled);
        assert!(!rv32_csr.supervisor_csr_operands_enabled);
        assert!(!rv64_csr.supervisor_csr_operands_enabled);
    }

    #[test]
    fn build_march_from_extensions_consumes_data_driven_dependencies() {
        assert_eq!(
            build_fake_data_driven_march(&[FakeDataDrivenExtension::Demo]),
            "rv64im_zbb_zdemo"
        );
    }

    #[test]
    fn zcb_keeps_base_c_extras_separate() {
        assert_eq!(
            build_fake_rv32_march(&[FakeExtension::Zcb]),
            "rv32imc_zbb_zcb"
        );
        assert_eq!(
            build_fake_rv64_march(&[FakeExtension::Zcb]),
            "rv64imc_zba_zbb_zcb"
        );
    }

    #[test]
    fn gnu_compatibility_overlays_stay_explicit() {
        assert_eq!(
            build_fake_rv64_march(&[FakeExtension::B]),
            "rv64i_zba_zbb_zbc_zbs"
        );
        assert_eq!(
            build_fake_rv64_march(&[FakeExtension::Zaamo]),
            "rv64ia_zaamo"
        );
        assert_eq!(
            build_fake_rv64_march(&[FakeExtension::Zabha]),
            "rv64ia_zabha_zacas"
        );
        assert_eq!(
            build_fake_rv64_march(&[FakeExtension::Zalasr]),
            "rv64ia_zalasr"
        );
        assert_eq!(
            build_fake_rv64_march(&[FakeExtension::Smdbltrp]),
            "rv64i_smctr_smdbltrp"
        );
    }

    #[test]
    fn arch_specific_compatibility_overlays_remain_arch_specific() {
        assert_eq!(
            build_fake_rv32_march(&[FakeExtension::Zcmop]),
            "rv32iac_zacas_zcmop"
        );
        assert_eq!(
            build_fake_rv64_march(&[FakeExtension::Zcmop]),
            "rv64ic_zcmop"
        );
    }

    #[test]
    fn hypervisor_and_vector_floating_extensions_drive_csr_domains() {
        let hypervisor = build_fake_rv64_csr_config(&[FakeExtension::H]);
        assert!(hypervisor.machine_csr_operands_enabled);
        assert!(hypervisor.supervisor_csr_operands_enabled);
        assert!(hypervisor.hypervisor_csr_operands_enabled);
        assert!(hypervisor.virtual_supervisor_csr_operands_enabled);

        let vector_floating = build_fake_rv64_csr_config(&[FakeExtension::Zvfbfmin]);
        assert!(vector_floating.machine_csr_operands_enabled);
        assert!(vector_floating.vector_csr_operands_enabled);
        assert!(vector_floating.floating_csr_operands_enabled);
    }

    #[test]
    fn floating_standard_extensions_no_longer_inject_unrelated_prefixed_extensions() {
        assert_eq!(build_fake_rv64_march(&[FakeExtension::F]), "rv64if");
        assert_eq!(build_fake_rv64_march(&[FakeExtension::Q]), "rv64ifdq");
    }

    #[test]
    fn sm_does_not_emit_unsupported_prefixed_march_fragment() {
        assert_eq!(build_fake_rv32_march(&[FakeExtension::Sm]), "rv32i");
        assert_eq!(build_fake_rv64_march(&[FakeExtension::Sm]), "rv64i");
    }
}
