use crate::{PluginDoctorCheck, PluginDoctorReport, doctor::PluginDoctorStatus};

use super::common::{decode_message, pb};

pub(crate) fn decode_plugin_doctor_report(bytes: &[u8]) -> Result<PluginDoctorReport, String> {
    let message: pb::PluginDoctorReport = decode_message(bytes, "plugin doctor report")?;
    plugin_doctor_report_from_proto(message)
}

pub(crate) fn plugin_doctor_report_to_proto(report: &PluginDoctorReport) -> pb::PluginDoctorReport {
    pb::PluginDoctorReport {
        plugin_id: report.plugin_id.clone(),
        checks: report
            .checks
            .iter()
            .map(plugin_doctor_check_to_proto)
            .collect(),
    }
}

pub(crate) fn plugin_doctor_report_from_proto(
    message: pb::PluginDoctorReport,
) -> Result<PluginDoctorReport, String> {
    Ok(PluginDoctorReport {
        plugin_id: message.plugin_id,
        checks: message
            .checks
            .into_iter()
            .map(plugin_doctor_check_from_proto)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn plugin_doctor_check_to_proto(check: &PluginDoctorCheck) -> pb::PluginDoctorCheck {
    pb::PluginDoctorCheck {
        label: check.label.clone(),
        env_var: check.env_var.clone(),
        status: doctor_status_to_proto(check.status),
        value: check.value.clone(),
        note: check.note.clone(),
    }
}

fn plugin_doctor_check_from_proto(
    message: pb::PluginDoctorCheck,
) -> Result<PluginDoctorCheck, String> {
    Ok(PluginDoctorCheck {
        label: message.label,
        env_var: message.env_var,
        status: doctor_status_from_proto(message.status, "plugin doctor check.status")?,
        value: message.value,
        note: message.note,
    })
}

fn doctor_status_to_proto(status: PluginDoctorStatus) -> i32 {
    match status {
        PluginDoctorStatus::Pass => pb::PluginDoctorStatus::Pass as i32,
        PluginDoctorStatus::Warn => pb::PluginDoctorStatus::Warn as i32,
        PluginDoctorStatus::Fail => pb::PluginDoctorStatus::Fail as i32,
    }
}

fn doctor_status_from_proto(value: i32, field: &str) -> Result<PluginDoctorStatus, String> {
    match pb::PluginDoctorStatus::try_from(value)
        .map_err(|_| format!("invalid {field} value: {value}"))?
    {
        pb::PluginDoctorStatus::Pass => Ok(PluginDoctorStatus::Pass),
        pb::PluginDoctorStatus::Warn => Ok(PluginDoctorStatus::Warn),
        pb::PluginDoctorStatus::Fail => Ok(PluginDoctorStatus::Fail),
        pb::PluginDoctorStatus::Unspecified => Err(format!("{field} must not be unspecified")),
    }
}
