//! 用于“首次定位耗时”指标的 best-effort 计时探针。
//! Best-effort timing probe for "time to first localization" metrics.
//!
//! 这份全局状态只服务于 CLI 或 benchmark 场景下的可观测性。
//! This global state exists only for observability in CLI/benchmark flows.
//! diff-analysis 的正确性绝不能依赖探针是否开启。
//! Correctness of diff-analysis must never depend on whether the probe is enabled.

use std::{sync::Mutex, time::Instant};

use once_cell::sync::Lazy;

#[derive(Debug, Clone, Copy, Default)]
pub struct LocalizationTiming {
    pub first_exception_ms: Option<f64>,
    pub first_write_ms: Option<f64>,
}

static LOCALIZATION_PROBE: Lazy<Mutex<Option<(Instant, LocalizationTiming)>>> =
    Lazy::new(|| Mutex::new(None));

pub fn localization_probe_reset(start: Instant) {
    let mut guard = LOCALIZATION_PROBE
        .lock()
        .expect("localization probe mutex poisoned");
    *guard = Some((start, LocalizationTiming::default()));
}

pub fn localization_probe_get() -> Option<LocalizationTiming> {
    let guard = LOCALIZATION_PROBE
        .lock()
        .expect("localization probe mutex poisoned");
    guard.as_ref().map(|(_, timing)| *timing)
}

pub(super) fn record_first_exception_localization() {
    // 只保留首个时间戳；比起后续轮次，第一次成功定位异常触发指令更有观察价值。
    // Keep the first timestamp only; later rounds are less interesting than the first point
    // where the reducer managed to localize an exception-triggering instruction.
    let now = Instant::now();
    let mut guard = LOCALIZATION_PROBE
        .lock()
        .expect("localization probe mutex poisoned");
    if let Some((start, ref mut timing)) = *guard
        && timing.first_exception_ms.is_none()
    {
        timing.first_exception_ms = Some((now - start).as_secs_f64() * 1e3);
    }
}

pub(super) fn record_first_write_localization() {
    // 写差异计时沿用同样的策略：指标只记录第一次成功定位。
    // Same policy as exception timing: the first confirmed write localization is the metric.
    let now = Instant::now();
    let mut guard = LOCALIZATION_PROBE
        .lock()
        .expect("localization probe mutex poisoned");
    if let Some((start, ref mut timing)) = *guard
        && timing.first_write_ms.is_none()
    {
        timing.first_write_ms = Some((now - start).as_secs_f64() * 1e3);
    }
}
