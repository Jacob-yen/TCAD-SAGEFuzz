use std::io;
use std::process::{Child, Command};

#[cfg(unix)]
pub fn configure_command_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
pub fn configure_command_process_group(_command: &mut Command) {}

#[cfg(unix)]
pub fn kill_child_process_group(child: &mut Child) -> io::Result<()> {
    let pid = child.id() as i32;
    if pid <= 0 {
        return Ok(());
    }

    let result = unsafe { libc::kill(-pid, libc::SIGKILL) };
    if result == 0 {
        Ok(())
    } else {
        let error = io::Error::last_os_error();
        if matches!(error.raw_os_error(), Some(libc::ESRCH)) {
            Ok(())
        } else {
            Err(error)
        }
    }
}

#[cfg(not(unix))]
pub fn kill_child_process_group(child: &mut Child) -> io::Result<()> {
    child.kill()
}
