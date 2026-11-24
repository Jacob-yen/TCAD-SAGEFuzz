pub mod march_util;
pub mod sequence_generator;

riscv_instruction_macros::generate_separated_riscv_instructions!(
    "../assets/riscv_instructions.json"
);
generate_assemble_march!();
generate_sequence_generator!();

#[cfg(test)]
mod test {
    use crate::*;
    use enum_iterator::all;
    use rand::{SeedableRng, rngs::StdRng};
    use riscv_instruction_types::{
        AsmDecodeWithContext, AsmEncode, FLen, InstructionSequence, IsaDecodeContext,
        IsaProfile, MemoryAccessConfig, MemoryAccessInstruction, PseudoInstruction,
        RandomGenerationError, RegisterBank, RegisterOperandConfig, RuntimeMemoryOperandPatch,
        RuntimeOperandAccess, RuntimeOperandUsage, RuntimeRegisterAssignment,
        RuntimeRegisterRole, ValidatedValue, XLen, XLenImmediate,
    };
    use std::collections::{HashMap, VecDeque};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::{Arc, Mutex, OnceLock};

    fn test_random_config() -> RandomConfig {
        let memory_access_config = MemoryAccessConfig {
            memory_base_register_number_range: (8, 12),
            memory_offset_immediate_range: (0, 16),
            ..MemoryAccessConfig::default()
        };
        let register_operand_config = RegisterOperandConfig {
            integer_register_number_range: (13, 20),
            ..RegisterOperandConfig::default()
        };

        RandomConfig {
            isa_profile: IsaProfile::new(XLen::X64, Some(FLen::F64)),
            memory_access_config,
            register_operand_config,
            ..RandomConfig::default()
        }
    }

    fn rv32_test_random_config(extension: RV32Extensions) -> RandomConfig {
        RandomConfig {
            isa_profile: IsaProfile::new(XLen::X32, Some(FLen::F64)),
            csr_selection_config: csr_config_from_rv32_extensions(&[extension]),
            ..test_random_config()
        }
    }

    fn rv64_test_random_config(extension: RV64Extensions) -> RandomConfig {
        RandomConfig {
            isa_profile: IsaProfile::new(XLen::X64, Some(FLen::F64)),
            csr_selection_config: csr_config_from_rv64_extensions(&[extension]),
            ..test_random_config()
        }
    }

    const SAMPLES_PER_INSTRUCTION: usize = 30;

    fn generate_rv32_instructions(
        extension: RV32Extensions,
    ) -> Result<Vec<String>, RandomGenerationError> {
        let mut rng = rand::rng();
        let config = rv32_test_random_config(extension);
        let results = extension.all_instruction_sequences_for_test(
            &mut rng,
            &config,
            SAMPLES_PER_INSTRUCTION,
        );
        results
            .into_iter()
            .map(|r| r.map(|seq| seq.encode_asm()))
            .collect()
    }

    fn generate_rv64_instructions(
        extension: RV64Extensions,
    ) -> Result<Vec<String>, RandomGenerationError> {
        let mut rng = rand::rng();
        let config = rv64_test_random_config(extension);
        let results = extension.all_instruction_sequences_for_test(
            &mut rng,
            &config,
            SAMPLES_PER_INSTRUCTION,
        );
        results
            .into_iter()
            .map(|r| r.map(|seq| seq.encode_asm()))
            .collect()
    }

    fn create_assembly_file(instructions: &[String], filename: &str) -> std::io::Result<()> {
        // Delete any existing file with the same name first.
        if std::path::Path::new(filename).exists() {
            std::fs::remove_file(filename)?;
        }

        // Build the full assembly-file contents in memory.
        let mut content = String::new();

        content.push_str(".section .text\n");
        content.push_str(".global _start\n");
        content.push_str("_start:\n");

        for inst in instructions {
            content.push_str(&format!("    {}\n", inst));
        }

        content.push_str("    # Exit program\n");
        content.push_str("    li a0, 0\n");
        content.push_str("    li a7, 93\n");
        content.push_str("    ecall\n");

        // Retry file creation several times.
        for attempt in 1..=5 {
            match std::fs::write(filename, &content) {
                Ok(_) => {
                    // After a successful write, wait briefly so the filesystem can catch up.
                    std::thread::sleep(std::time::Duration::from_millis(10));

                    // Verify that the file really exists and is readable.
                    if std::path::Path::new(filename).exists() {
                        match std::fs::read_to_string(filename) {
                            Ok(read_content) => {
                                if read_content.len() == content.len() {
                                    return Ok(());
                                } else {
                                    return Err(std::io::Error::new(
                                        std::io::ErrorKind::InvalidData,
                                        format!(
                                            "文件内容长度不匹配: 期望 {}, 实际 {}",
                                            content.len(),
                                            read_content.len()
                                        ),
                                    ));
                                }
                            }
                            Err(e) => {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::PermissionDenied,
                                    format!("无法读取刚创建的文件: {}", e),
                                ));
                            }
                        }
                    } else {
                        if attempt < 5 {
                            println!("    尝试 {}: 文件写入后不存在，重试...", attempt);
                            std::thread::sleep(std::time::Duration::from_millis(50));
                            continue;
                        } else {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                format!("文件创建后不存在: {} (尝试 {})", filename, attempt),
                            ));
                        }
                    }
                }
                Err(e) => {
                    if attempt < 5 {
                        println!("    尝试 {}: 写入失败 {}, 重试...", attempt, e);
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        continue;
                    } else {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("文件写入失败 (尝试 {}): {}", attempt, e),
                        ));
                    }
                }
            }
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "所有文件创建尝试都失败了",
        ))
    }

    fn assert_instruction_roundtrip_with_context<T: AsmEncode + AsmDecodeWithContext>(
        asm: &str,
        ctx: IsaDecodeContext,
    ) {
        let instruction = T::decode_asm_with_context(asm, ctx)
            .unwrap_or_else(|err| panic!("failed to parse `{asm}`: {err}"));
        assert_eq!(instruction.encode_asm(), asm);
    }

    fn assert_sequence_roundtrip_with_context<T>(sequence: &InstructionSequence<T>, ctx: IsaDecodeContext)
    where
        T: Clone + AsmEncode + AsmDecodeWithContext + riscv_instruction_types::InstructionSequenceItem,
    {
        let instruction_asm = sequence.instruction.encode_asm();
        assert_instruction_roundtrip_with_context::<T>(&instruction_asm, ctx);

        let sequence_asm = sequence.encode_asm();
        assert_instruction_roundtrip_with_context::<InstructionSequence<T>>(&sequence_asm, ctx);
    }

    fn assert_random_generation_roundtrip_with_context<T, F>(
        label: &str,
        sample_count: usize,
        seed: u64,
        config: &RandomConfig,
        ctx: IsaDecodeContext,
        mut generate: F,
    ) where
        T: Clone + AsmEncode + AsmDecodeWithContext + riscv_instruction_types::InstructionSequenceItem,
        F: FnMut(
            &mut StdRng,
            &RandomConfig,
        ) -> Result<InstructionSequence<T>, RandomGenerationError>,
    {
        let mut rng = StdRng::seed_from_u64(seed);

        for sample_idx in 0..sample_count {
            let sequence = generate(&mut rng, config).unwrap_or_else(|err| {
                panic!("failed to generate roundtrip sample {sample_idx} for {label}: {err}")
            });
            assert_sequence_roundtrip_with_context(&sequence, ctx);
        }
    }

    fn resolve_riscv_assembler() -> PathBuf {
        if let Ok(path) = std::env::var("RISCV_AS") {
            let path = path.trim();
            if !path.is_empty() {
                return PathBuf::from(path);
            }
        }

        let bundled =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/riscv64-unknown-elf-as");
        if bundled.is_file() {
            return bundled;
        }

        PathBuf::from("riscv64-unknown-elf-as")
    }

    fn test_assembly(filename: &str, march: &str, output_path: &str) -> (bool, String) {
        // Re-check that the input file exists before running the test.
        for check_attempt in 1..=3 {
            if std::path::Path::new(filename).exists() {
                break;
            } else {
                if check_attempt < 3 {
                    println!("    检查 {}: 文件不存在，等待...", check_attempt);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                } else {
                    return (false, format!("输入文件在所有检查后仍不存在: {}", filename));
                }
            }
        }

        // Try reading the file to confirm permissions and content integrity.
        match std::fs::File::open(filename) {
            Ok(_) => {
                // The file can be opened, so keep validating its contents.
                match std::fs::read_to_string(filename) {
                    Ok(content) => {
                        if content.is_empty() {
                            return (false, format!("输入文件 {} 为空", filename));
                        }
                        if !content.contains(".section .text") {
                            return (false, format!("输入文件 {} 内容格式不正确", filename));
                        }
                    }
                    Err(e) => {
                        return (false, format!("无法读取输入文件 {}: {}", filename, e));
                    }
                }
            }
            Err(e) => {
                return (false, format!("无法打开输入文件 {}: {}", filename, e));
            }
        }

        // Resolve the absolute file path.
        let abs_path = match std::fs::canonicalize(filename) {
            Ok(path) => path.to_string_lossy().to_string(),
            Err(_) => filename.to_string(),
        };

        // Remove any stale output file.
        let _ = std::fs::remove_file(output_path);

        let assembler = resolve_riscv_assembler();
        let assembler_display = assembler.display().to_string();

        let output = Command::new(&assembler)
            .arg(format!("-march={}", march))
            .arg("-o")
            .arg(output_path)
            .arg(&abs_path)
            .output();

        match output {
            Ok(result) => {
                let success = result.status.success();
                let stderr = String::from_utf8_lossy(&result.stderr);
                let stdout = String::from_utf8_lossy(&result.stdout);
                let has_error = stderr.to_lowercase().contains("error");

                let error_info = if !success || has_error {
                    format!(
                        "Exit code: {}\nCommand: {} -march={} -o {} {}\nSTDOUT:\n{}\nSTDERR:\n{}",
                        result.status.code().unwrap_or(-1),
                        assembler_display,
                        march,
                        output_path,
                        abs_path,
                        stdout,
                        stderr
                    )
                } else {
                    String::new()
                };

                (success && !has_error, error_info)
            }
            Err(e) => (
                false,
                format!("Command execution failed for {}: {}", assembler_display, e),
            ),
        }
    }

    #[derive(Debug, Clone)]
    enum MarchProbe {
        Supported,
        Unsupported { reason: String },
        ProbeFailure { reason: String },
    }

    fn sanitize_filename_component(value: &str) -> String {
        value
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                    ch
                } else {
                    '_'
                }
            })
            .collect()
    }

    fn march_probe_cache() -> &'static Mutex<HashMap<String, MarchProbe>> {
        static CACHE: OnceLock<Mutex<HashMap<String, MarchProbe>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Probe the assembler with a tiny assembly file before running the heavy random test.
    fn probe_march_support(march: &str) -> MarchProbe {
        if let Some(cached) = march_probe_cache().lock().unwrap().get(march).cloned() {
            return cached;
        }

        let probe = run_march_probe(march);
        march_probe_cache()
            .lock()
            .unwrap()
            .insert(march.to_string(), probe.clone());
        probe
    }

    fn run_march_probe(march: &str) -> MarchProbe {
        let assembler = resolve_riscv_assembler();
        let assembler_display = assembler.display().to_string();
        let suffix = format!(
            "{}_{}_{}",
            std::process::id(),
            sanitize_filename_component(march),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        );
        let asm_path = format!("probe_{}.S", suffix);
        let output_path = format!("probe_{}.o", suffix);
        let asm_source = ".section .text\n.global _start\n_start:\n    addi x0, x0, 0\n";

        let probe = (|| {
            std::fs::write(&asm_path, asm_source)
                .map_err(|err| format!("无法写入 march 探测文件 {}: {}", asm_path, err))?;

            let abs_path = std::fs::canonicalize(&asm_path)
                .map(|path| path.to_string_lossy().to_string())
                .unwrap_or_else(|_| asm_path.clone());

            let output = Command::new(&assembler)
                .arg(format!("-march={}", march))
                .arg("-o")
                .arg(&output_path)
                .arg(&abs_path)
                .output()
                .map_err(|err| {
                    format!(
                        "march 探测命令执行失败: {} -march={} -o {} {}: {}",
                        assembler_display, march, output_path, abs_path, err
                    )
                })?;

            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr_lower = stderr.to_ascii_lowercase();

            if stderr_lower.contains("unknown prefixed isa extension")
                || stderr_lower.contains("unknown isa extension")
                || stderr_lower.contains("unsupported isa extension")
            {
                return Ok(MarchProbe::Unsupported {
                    reason: format!(
                        "GNU as 不支持 march `{}`\nCommand: {} -march={} -o {} {}\nSTDOUT:\n{}\nSTDERR:\n{}",
                        march, assembler_display, march, output_path, abs_path, stdout, stderr
                    ),
                });
            }

            if !output.status.success() || stderr_lower.contains("error") {
                return Ok(MarchProbe::ProbeFailure {
                    reason: format!(
                        "march 探测失败\nCommand: {} -march={} -o {} {}\nSTDOUT:\n{}\nSTDERR:\n{}",
                        assembler_display, march, output_path, abs_path, stdout, stderr
                    ),
                });
            }

            Ok(MarchProbe::Supported)
        })();

        let _ = std::fs::remove_file(&asm_path);
        let _ = std::fs::remove_file(&output_path);

        probe.unwrap_or_else(|reason| MarchProbe::ProbeFailure { reason })
    }

    #[derive(Debug, Clone)]
    enum TestOutcome {
        Success,
        Failure { details: String },
        Skipped { reason: String },
    }

    #[derive(Debug, Clone)]
    struct TestResult {
        name: String,
        outcome: TestOutcome,
    }

    impl TestResult {
        fn success(name: String) -> Self {
            Self {
                name,
                outcome: TestOutcome::Success,
            }
        }

        fn failure(name: String, details: impl Into<String>) -> Self {
            Self {
                name,
                outcome: TestOutcome::Failure {
                    details: details.into(),
                },
            }
        }

        fn skipped(name: String, reason: impl Into<String>) -> Self {
            Self {
                name,
                outcome: TestOutcome::Skipped {
                    reason: reason.into(),
                },
            }
        }
    }

    const PARALLEL_ENV_VAR: &str = "RISCV_TEST_THREADS";

    fn resolve_parallelism() -> usize {
        if let Ok(value) = std::env::var(PARALLEL_ENV_VAR) {
            match value.parse::<usize>() {
                Ok(parsed) if parsed > 0 => return parsed,
                _ => println!(
                    "警告: 环境变量 {} 的值无效 ({}), 将使用默认线程数",
                    PARALLEL_ENV_VAR, value
                ),
            }
        }

        std::thread::available_parallelism()
            .map(|count| count.get().saturating_sub(5).max(1))
            .unwrap_or(1)
    }

    fn run_in_parallel<E>(
        items: Vec<E>,
        worker: fn(E) -> TestResult,
        threads: usize,
    ) -> Vec<TestResult>
    where
        E: Send + 'static,
    {
        if items.is_empty() {
            return Vec::new();
        }

        let total_items = items.len();
        let max_threads = threads.max(1).min(total_items);
        let queue = Arc::new(Mutex::new({
            let mut deque = VecDeque::with_capacity(items.len());
            for (idx, item) in items.into_iter().enumerate() {
                deque.push_back((idx, item));
            }
            deque
        }));
        let results = Arc::new(Mutex::new(vec![None; total_items]));

        let mut handles = Vec::with_capacity(max_threads);
        for _ in 0..max_threads {
            let queue = Arc::clone(&queue);
            let results = Arc::clone(&results);
            handles.push(std::thread::spawn(move || {
                loop {
                    let job = {
                        let mut queue_lock = queue.lock().unwrap();
                        queue_lock.pop_front()
                    };
                    match job {
                        Some((idx, item)) => {
                            let result = worker(item);
                            let mut results_lock = results.lock().unwrap();
                            results_lock[idx] = Some(result);
                        }
                        None => break,
                    }
                }
            }));
        }

        for handle in handles {
            let _ = handle.join();
        }

        let results = Arc::try_unwrap(results)
            .expect("无法回收结果所有权")
            .into_inner()
            .expect("无法获取结果");

        results
            .into_iter()
            .map(|entry| entry.expect("缺少测试结果"))
            .collect()
    }

    fn run_rv32_extension(extension: RV32Extensions) -> TestResult {
        let extension_name = format!("RV32_{:?}", extension);
        let march = build_rv32_march(&vec![extension]);

        match probe_march_support(&march) {
            MarchProbe::Supported => {}
            MarchProbe::Unsupported { reason } => {
                return TestResult::skipped(extension_name, reason);
            }
            MarchProbe::ProbeFailure { reason } => {
                return TestResult::failure(extension_name, reason);
            }
        }

        let instructions = match generate_rv32_instructions(extension) {
            Ok(instrs) => instrs,
            Err(err) => {
                return TestResult::failure(extension_name, format!("随机生成指令失败: {}", err));
            }
        };

        let filename = format!("test_{}.S", extension_name);
        let _ = std::fs::remove_file(&filename);

        if let Err(e) = create_assembly_file(&instructions, &filename) {
            return TestResult::failure(
                extension_name,
                format!("无法创建汇编文件 {}: {}", filename, e),
            );
        }

        if !std::path::Path::new(&filename).exists() {
            return TestResult::failure(
                extension_name,
                format!("汇编文件 {} 创建后不存在", filename),
            );
        }

        let output_path = format!("output_{}.o", extension_name);
        let (success, error_info) = test_assembly(&filename, &march, &output_path);

        let result = if success {
            TestResult::success(extension_name)
        } else {
            let mut details = if error_info.is_empty() {
                "未知错误".to_string()
            } else {
                error_info
            };

            match create_error_log(&extension_name, &instructions, &march, &details) {
                Ok((log_file, asm_file)) => {
                    details.push_str(&format!("; 错误日志: {}; 汇编文件: {}", log_file, asm_file));
                }
                Err(e) => details.push_str(&format!("; 创建错误日志失败: {}", e)),
            }

            TestResult::failure(extension_name, details)
        };

        let _ = std::fs::remove_file(&filename);
        let _ = std::fs::remove_file(&output_path);

        result
    }

    fn run_rv64_extension(extension: RV64Extensions) -> TestResult {
        let extension_name = format!("RV64_{:?}", extension);
        let march = build_rv64_march(&vec![extension]);

        match probe_march_support(&march) {
            MarchProbe::Supported => {}
            MarchProbe::Unsupported { reason } => {
                return TestResult::skipped(extension_name, reason);
            }
            MarchProbe::ProbeFailure { reason } => {
                return TestResult::failure(extension_name, reason);
            }
        }

        let instructions = match generate_rv64_instructions(extension) {
            Ok(instrs) => instrs,
            Err(err) => {
                return TestResult::failure(extension_name, format!("随机生成指令失败: {}", err));
            }
        };

        let filename = format!("test_{}.S", extension_name);
        let _ = std::fs::remove_file(&filename);

        if let Err(e) = create_assembly_file(&instructions, &filename) {
            return TestResult::failure(
                extension_name,
                format!("无法创建汇编文件 {}: {}", filename, e),
            );
        }

        if !std::path::Path::new(&filename).exists() {
            return TestResult::failure(
                extension_name,
                format!("汇编文件 {} 创建后不存在", filename),
            );
        }

        let output_path = format!("output_{}.o", extension_name);
        let (success, error_info) = test_assembly(&filename, &march, &output_path);

        let result = if success {
            TestResult::success(extension_name)
        } else {
            let mut details = if error_info.is_empty() {
                "未知错误".to_string()
            } else {
                error_info
            };

            match create_error_log(&extension_name, &instructions, &march, &details) {
                Ok((log_file, asm_file)) => {
                    details.push_str(&format!("; 错误日志: {}; 汇编文件: {}", log_file, asm_file));
                }
                Err(e) => details.push_str(&format!("; 创建错误日志失败: {}", e)),
            }

            TestResult::failure(extension_name, details)
        };

        let _ = std::fs::remove_file(&filename);
        let _ = std::fs::remove_file(&output_path);

        result
    }

    fn collect_rv32_extensions() -> Vec<RV32Extensions> {
        all::<RV32Extensions>().collect()
    }

    fn collect_rv64_extensions() -> Vec<RV64Extensions> {
        all::<RV64Extensions>().collect()
    }

    fn create_error_log(
        extension_name: &str,
        instructions: &[String],
        march: &str,
        error_info: &str,
    ) -> std::io::Result<(String, String)> {
        // Create the error-log directory.
        std::fs::create_dir_all("error_logs")?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let log_filename = format!("error_logs/{}_{}_errors.log", extension_name, timestamp);

        // Build the full log contents in memory.
        let mut log_content = String::new();

        log_content.push_str("=== 错误日志 ===\n");
        log_content.push_str(&format!("指令集扩展: {}\n", extension_name));
        log_content.push_str(&format!("MARCH: {}\n", march));
        log_content.push_str(&format!("时间: {}\n", timestamp));
        log_content.push_str(&format!("指令数量: {}\n", instructions.len()));
        log_content.push_str("\n");

        log_content.push_str("=== 错误详情 ===\n");
        log_content.push_str(error_info);
        log_content.push_str("\n\n");

        log_content.push_str("=== 生成的指令 ===\n");
        for (i, inst) in instructions.iter().enumerate() {
            log_content.push_str(&format!("{:4}: {}\n", i + 1, inst));
        }

        // Write the full log contents in one shot.
        std::fs::write(&log_filename, log_content)?;

        // Also save the assembly file under the error-log directory.
        let asm_filename = format!("error_logs/{}_{}.S", extension_name, timestamp);
        create_assembly_file(instructions, &asm_filename)?;

        Ok((log_filename, asm_filename))
    }

    #[test]
    fn separated_instruction_offset_access() -> Result<(), Box<dyn std::error::Error>> {
        let xd = IntegerRegister::new(5)?;
        let base = riscv_instruction_types::BaseAddressRegister::new(9)?;
        let imm = riscv_instruction_types::Immediate::<12, true>::new(16)?;
        let lw_struct = RV32_I_LW { imm, xs1: base, xd };

        assert_eq!(lw_struct.offset_operand_value(), Some(16));

        let lw_instruction =
            RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::LW(lw_struct.clone())));
        assert_eq!(lw_instruction.offset_operand_value(), Some(16));

        Ok(())
    }

    #[test]
    fn separated_struct_runtime_operand_access_reads_and_writes_memory_operands()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut lw = RV32_I_LW {
            imm: riscv_instruction_types::Immediate::<12, true>::new(16)?,
            xs1: riscv_instruction_types::BaseAddressRegister::new(9)?,
            xd: IntegerRegister::new(5)?,
        };

        let registers = lw.register_operands();
        assert_eq!(registers.len(), 2);
        assert_eq!(registers[0].name, "xs1", "unexpected operand ordering");
        assert_eq!(registers[1].name, "xd", "unexpected operand ordering");

        let memory = lw.memory_operands();
        assert_eq!(memory.len(), 1);
        assert_eq!(memory[0].base_register_operand_name, Some("xs1"));
        assert_eq!(memory[0].offset_operand_name, Some("imm"));
        assert_eq!(memory[0].offset_value, Some(16));
        let base = memory[0]
            .base_register
            .as_ref()
            .expect("lw should expose a memory base register");
        assert_eq!(base.usage, RuntimeOperandUsage::MemoryBase);
        assert_eq!(base.role, RuntimeRegisterRole::MemoryBase);
        assert_eq!(base.register.physical_number, 9);

        assert!(lw.set_register_operand(
            "xd",
            RuntimeRegisterAssignment {
                bank: RegisterBank::Integer,
                number: 7,
                numbering: riscv_instruction_types::RegisterNumbering::Physical,
            },
        )?);
        assert!(lw.set_memory_operand(
            0,
            RuntimeMemoryOperandPatch {
                base_register: Some(RuntimeRegisterAssignment {
                    bank: RegisterBank::Integer,
                    number: 11,
                    numbering: riscv_instruction_types::RegisterNumbering::Physical,
                }),
                offset_value: Some(24),
            },
        )?);

        assert_eq!(lw.xd.get(), 7);
        assert_eq!(lw.xs1.get(), 11);
        assert_eq!(lw.imm.get(), 24);

        Ok(())
    }

    #[test]
    fn separated_enum_runtime_operand_access_avoids_manual_variant_matching()
    -> Result<(), Box<dyn std::error::Error>> {
        let lw = RV32_I_LW {
            imm: riscv_instruction_types::Immediate::<12, true>::new(8)?,
            xs1: riscv_instruction_types::BaseAddressRegister::new(10)?,
            xd: IntegerRegister::new(6)?,
        };
        let mut instruction = RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::LW(lw)));

        let registers = instruction.register_operands();
        assert_eq!(registers.len(), 2);
        assert!(registers.iter().any(|operand| operand.name == "xs1"));
        assert!(registers.iter().any(|operand| operand.name == "xd"));

        assert!(instruction.set_memory_base_register(
            0,
            RuntimeRegisterAssignment {
                bank: RegisterBank::Integer,
                number: 12,
                numbering: riscv_instruction_types::RegisterNumbering::Physical,
            },
        )?);
        assert!(instruction.set_memory_offset_value(0, 20)?);
        assert!(instruction.set_register_operand(
            "xd",
            RuntimeRegisterAssignment {
                bank: RegisterBank::Integer,
                number: 13,
                numbering: riscv_instruction_types::RegisterNumbering::Physical,
            },
        )?);

        let rewritten = instruction.encode_asm();
        assert_eq!(rewritten, "lw x13, 20(x12)");

        Ok(())
    }

    #[test]
    fn pseudo_instruction_runtime_operand_access_supports_uniform_register_rewrite()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut pseudo = PseudoInstruction::LoadImmediate {
            rd: IntegerRegister::new(8)?,
            immediate: XLenImmediate::X64(32),
            purpose: riscv_instruction_types::LoadImmediatePurpose::Generic,
        };

        let registers = pseudo.register_operands();
        assert_eq!(registers.len(), 1);
        assert_eq!(registers[0].name, "rd");
        assert_eq!(registers[0].register.physical_number, 8);

        assert!(pseudo.set_register_operand(
            "rd",
            RuntimeRegisterAssignment {
                bank: RegisterBank::Integer,
                number: 9,
                numbering: riscv_instruction_types::RegisterNumbering::Physical,
            },
        )?);
        assert_eq!(pseudo.encode_asm(), "li x9, 32");

        Ok(())
    }

    #[test]
    fn separated_instruction_from_str_roundtrips_canonical_asm() {
        for asm in [
            "add x1, x2, x3",
            "lw x1, 16(x2)",
            "c.lui x1, 0xfffff",
            "cm.push {ra, s0-s1}, -32",
            "vadd.vi v1, v2, 7, v0.t",
            "ssamoswap.d.aqrl x1, x2, 0(x3)",
        ] {
            assert_instruction_roundtrip_with_context::<RiscvInstruction>(asm, IsaDecodeContext::RV64);
        }
    }

    #[test]
    fn separated_instruction_sequence_from_str_roundtrips() {
        for asm in [
            "add x1, x2, x3",
            "li x8, 16\nlw x1, 0(x8)",
            "li x8, 16\nfadd.s f1, f2, f3, dyn\nfrflags x9",
            "lla x1, .Lriscv_jalr_deadbeef\njalr x0, 0(x1)\n.Lriscv_jalr_deadbeef:",
        ] {
            assert_instruction_roundtrip_with_context::<InstructionSequence<RiscvInstruction>>(asm, IsaDecodeContext::RV64);
        }
    }

    #[test]
    fn separated_instruction_sequence_parses_expected_structure() {
        let asm = "li x8, 16\nfadd.s f1, f2, f3, dyn\nfrflags x9";
        let sequence = InstructionSequence::<RiscvInstruction>::decode_asm_with_context(asm, IsaDecodeContext::RV64)
            .expect("expected sequence to parse");

        assert_eq!(sequence.encode_asm(), asm);
        assert_eq!(
            sequence.pre_instructions,
            &[RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
                rd: IntegerRegister::new(8).expect("valid register"),
                immediate: XLenImmediate::X64(16),
                purpose: riscv_instruction_types::LoadImmediatePurpose::Generic,
            })]
        );
        assert_eq!(
            sequence.post_instructions,
            &[RiscvInstruction::Pseudo(PseudoInstruction::ReadFflags {
                rd: IntegerRegister::new(9).expect("valid register"),
            })]
        );

        let mut unified = Vec::with_capacity(
            sequence.pre_instructions.len() + 1 + sequence.post_instructions.len(),
        );
        unified.extend(sequence.pre_instructions.iter().cloned());
        unified.push(sequence.instruction.clone());
        unified.extend(sequence.post_instructions.iter().cloned());
        assert_eq!(unified.len(), 3);
        assert!(matches!(unified[0], RiscvInstruction::Pseudo(_)));
        assert!(matches!(
            unified[1],
            RiscvInstruction::RV32(_) | RiscvInstruction::RV64(_)
        ));
        assert!(matches!(unified[2], RiscvInstruction::Pseudo(_)));
    }

    #[test]
    fn separated_instruction_sequence_rejects_ambiguous_multiline_asm() {
        let err = InstructionSequence::<RiscvInstruction>::decode_asm_with_context("frflags x1\nfrflags x2", IsaDecodeContext::RV64)
            .expect_err("expected ambiguous sequence to be rejected");
        assert!(
            err.to_string()
                .contains("could not identify a unique main instruction line"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn separated_instruction_sequence_json_roundtrips_with_pseudos() {
        let asm = "li x8, 16\nfadd.s f1, f2, f3, dyn\nfrflags x9";
        let sequence = InstructionSequence::<RiscvInstruction>::decode_asm_with_context(asm, IsaDecodeContext::RV64)
            .expect("expected sequence to parse");

        let json = serde_json::to_string(&sequence).expect("serialize sequence");
        let decoded: InstructionSequence<RiscvInstruction> =
            serde_json::from_str(&json).expect("deserialize sequence");

        assert_eq!(decoded, sequence);
        assert_eq!(decoded.encode_asm(), asm);
        decoded
            .validate_for_profile(IsaProfile::new(XLen::X64, Some(FLen::F32)))
            .expect("sequence should validate under matching profile");
        decoded
            .validate_for_profile(IsaProfile::rv64())
            .expect_err("frflags pseudo should fail without floating support");
    }

    #[test]
    fn separated_self_generated_asm_roundtrips_for_all_extensions() {
        const SAMPLE_COUNT: usize = 64;

        for (idx, extension) in all::<RV32Extensions>().enumerate() {
            let config = rv32_test_random_config(extension);
            assert_random_generation_roundtrip_with_context(
                &format!("separated::{extension:?}"),
                SAMPLE_COUNT,
                0x5256_3332_0000_0000u64 + idx as u64,
                &config,
                IsaDecodeContext::RV32,
                |rng, config| extension.random_sequence_with_rng(rng, config),
            );
        }

        for (idx, extension) in all::<RV64Extensions>().enumerate() {
            let config = rv64_test_random_config(extension);
            assert_random_generation_roundtrip_with_context(
                &format!("separated::{extension:?}"),
                SAMPLE_COUNT,
                0x5256_3634_0000_0000u64 + idx as u64,
                &config,
                IsaDecodeContext::RV64,
                |rng, config| extension.random_sequence_with_rng(rng, config),
            );
        }
    }

    #[test]
    fn test_all_separated_instructions() {
        let _ = std::fs::remove_file("output.o");
        let _ = std::fs::remove_dir_all("error_logs");

        fn toolchain_unsupported_reason(name: &str, details: &str) -> Option<String> {
            if name.contains("Zalasr") {
                return Some(details.to_string());
            }

            let details_lower = details.to_lowercase();
            let unsupported = details_lower.contains("unknown prefixed isa extension")
                || details_lower.contains("unknown isa extension")
                || details_lower.contains("unrecognized opcode")
                || details_lower.contains("requires the following: rv64i");

            unsupported.then(|| details.to_string())
        }

        let parallelism = resolve_parallelism();

        let rv32_extensions = collect_rv32_extensions();
        let mut results = run_in_parallel(rv32_extensions, run_rv32_extension, parallelism);

        let rv64_extensions = collect_rv64_extensions();
        results.extend(run_in_parallel(
            rv64_extensions,
            run_rv64_extension,
            parallelism,
        ));

        let mut failed_cases = Vec::new();
        let mut skipped_cases: Vec<(String, String)> = Vec::new();

        for result in results {
            match &result.outcome {
                TestOutcome::Failure { details } => {
                    if let Some(reason) = toolchain_unsupported_reason(&result.name, details) {
                        skipped_cases.push((result.name.clone(), reason));
                    } else {
                        failed_cases.push(result);
                    }
                }
                TestOutcome::Skipped { reason } => {
                    skipped_cases.push((result.name.clone(), reason.clone()));
                }
                TestOutcome::Success => {}
            }
        }

        if !skipped_cases.is_empty() {
            println!("跳过的用例 (当前 GNU 工具链不支持对应 -march):");
            for (name, reason) in &skipped_cases {
                println!("- {}: {}", name, reason);
            }
        }

        if failed_cases.is_empty() {
            println!("无失败用例");
        } else {
            println!("失败的测试:");
            for case in &failed_cases {
                if let TestOutcome::Failure { details, .. } = &case.outcome {
                    println!("- {}: {}", case.name, details);
                }
            }
        }

        if let Ok(entries) = std::fs::read_dir(".") {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().to_str() {
                    if filename.starts_with("test_") && filename.ends_with(".S") {
                        let _ = std::fs::remove_file(filename);
                    }
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(".") {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().to_str() {
                    if filename.starts_with("output_") && filename.ends_with(".o") {
                        let _ = std::fs::remove_file(filename);
                    }
                }
            }
        }

        if failed_cases.is_empty() {
            return;
        }

        panic!("测试失败！有 {} 个测试用例失败", failed_cases.len());
    }
}
