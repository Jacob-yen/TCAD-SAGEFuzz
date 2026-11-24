use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
    str,
};

#[derive(Debug, Default, Clone)]
pub struct CoverageSummary {
    pub type_counts: HashMap<String, (u64, u64)>,
}

impl CoverageSummary {
    pub fn render_lines(&self) -> Vec<String> {
        let mut keys: Vec<_> = self.type_counts.keys().cloned().collect();
        keys.sort();
        keys.into_iter()
            .map(|key| {
                let (total, hit) = self.type_counts.get(&key).copied().unwrap_or((0, 0));
                let pct = if total > 0 {
                    hit as f64 * 100.0 / total as f64
                } else {
                    0.0
                };
                format!("{key}: {hit} / {total} ({pct:.2}%)")
            })
            .collect()
    }
}

pub fn parse_verilator_dat<P: AsRef<Path>>(path: P) -> io::Result<CoverageSummary> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    let mut counts: HashMap<String, (u64, u64)> = HashMap::new();

    while reader.read_until(b'\n', &mut buffer)? != 0 {
        if buffer.starts_with(b"C '") {
            let count = buffer
                .iter()
                .rposition(|&byte| byte == b' ')
                .and_then(|pos| {
                    str::from_utf8(&buffer[pos + 1..buffer.len().saturating_sub(1)]).ok()
                })
                .and_then(|text| text.trim().parse::<u64>().ok());

            let typ = parse_verilator_type_from_record(&buffer).unwrap_or_else(|| "unknown".into());
            if let Some(count) = count {
                let entry = counts.entry(typ).or_insert((0, 0));
                entry.0 += 1;
                if count > 0 {
                    entry.1 += 1;
                }
            }
        }
        buffer.clear();
    }

    Ok(CoverageSummary {
        type_counts: counts,
    })
}

pub fn parse_verilator_hit_hashes<P: AsRef<Path>>(
    path: P,
    type_filter: &[&str],
) -> io::Result<HashSet<u64>> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    let mut hits = HashSet::new();

    while reader.read_until(b'\n', &mut buffer)? != 0 {
        if !buffer.starts_with(b"C '") {
            buffer.clear();
            continue;
        }

        let count = buffer
            .iter()
            .rposition(|&byte| byte == b' ')
            .and_then(|pos| str::from_utf8(&buffer[pos + 1..buffer.len().saturating_sub(1)]).ok())
            .and_then(|text| text.trim().parse::<u64>().ok())
            .unwrap_or(0);

        if count == 0 {
            buffer.clear();
            continue;
        }

        let (first_q, last_q) = match (
            buffer.iter().position(|&byte| byte == b'\''),
            buffer.iter().rposition(|&byte| byte == b'\''),
        ) {
            (Some(first_q), Some(last_q)) if last_q > first_q + 1 => (first_q, last_q),
            _ => {
                buffer.clear();
                continue;
            }
        };

        let key = &buffer[first_q + 1..last_q];
        if !type_filter.is_empty() {
            let typ = parse_verilator_type_from_key(key);
            if !typ.as_deref().is_some_and(|typ| type_filter.contains(&typ)) {
                buffer.clear();
                continue;
            }
        }

        hits.insert(fnv1a64(key));
        buffer.clear();
    }

    Ok(hits)
}

pub fn parse_spike_hit_hashes<P: AsRef<Path>>(path: P) -> io::Result<HashSet<u64>> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut hits = HashSet::new();

    for line in reader.lines() {
        let line = line?;
        let mut tokens = line.split_whitespace();

        if tokens.next() != Some("core") {
            continue;
        }

        let Some(_core_id) = tokens.next() else {
            continue;
        };

        let Some(token0) = tokens.next() else {
            continue;
        };

        let pc_token = if token0.starts_with("0x") {
            token0
        } else if token0.chars().all(|ch| ch.is_ascii_digit()) {
            match tokens.next() {
                Some(pc) => pc,
                None => continue,
            }
        } else {
            continue;
        };

        if !pc_token.starts_with("0x") {
            continue;
        }
        let pc_hex = pc_token.trim_start_matches("0x");

        let Some(instr_token) = tokens.next() else {
            continue;
        };
        if !instr_token.starts_with("(0x") {
            continue;
        }
        let instr_hex = instr_token.trim_start_matches("(0x").trim_end_matches(')');

        let mut key = Vec::with_capacity(pc_hex.len() + 1 + instr_hex.len());
        key.extend_from_slice(pc_hex.as_bytes());
        key.push(b':');
        key.extend_from_slice(instr_hex.as_bytes());
        hits.insert(fnv1a64(&key));
    }

    Ok(hits)
}

pub fn parse_rocket_hit_hashes<P: AsRef<Path>>(path: P) -> io::Result<HashSet<u64>> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut hits = HashSet::new();

    for line in reader.lines() {
        let line = line?;
        let mut tokens = line.split_whitespace();

        let Some(token0) = tokens.next() else {
            continue;
        };
        if !token0.chars().all(|ch| ch.is_ascii_digit()) {
            continue;
        }

        let Some(pc_token) = tokens.next() else {
            continue;
        };
        if !pc_token.starts_with("0x") {
            continue;
        }
        let pc_hex = pc_token.trim_start_matches("0x");

        let Some(instr_token) = tokens.next() else {
            continue;
        };
        if !instr_token.starts_with("(0x") {
            continue;
        }
        let instr_hex = instr_token.trim_start_matches("(0x").trim_end_matches(')');

        let mut key = Vec::with_capacity(pc_hex.len() + 1 + instr_hex.len());
        key.extend_from_slice(pc_hex.as_bytes());
        key.push(b':');
        key.extend_from_slice(instr_hex.as_bytes());
        hits.insert(fnv1a64(&key));
    }

    Ok(hits)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn parse_verilator_type_from_record(record: &[u8]) -> Option<String> {
    let (first_q, last_q) = match (
        record.iter().position(|&byte| byte == b'\''),
        record.iter().rposition(|&byte| byte == b'\''),
    ) {
        (Some(first_q), Some(last_q)) if last_q > first_q + 1 => (first_q, last_q),
        _ => return None,
    };

    parse_verilator_type_from_key(&record[first_q + 1..last_q])
}

fn parse_verilator_type_from_key(key: &[u8]) -> Option<String> {
    for segment in key.split(|&byte| byte == 0x01) {
        if let Some(pos) = segment.iter().position(|&byte| byte == 0x02) {
            let (name, value) = (&segment[..pos], &segment[pos + 1..]);
            if name == b"t" {
                return Some(str::from_utf8(value).unwrap_or("unknown").to_string());
            }
        }
    }

    None
}
