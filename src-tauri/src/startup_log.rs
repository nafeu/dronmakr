use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct StartupState {
    pub port: u16,
    pub spawn_error: Mutex<Option<String>>,
    pub backend_ready: Mutex<bool>,
    pub lines: Mutex<Vec<String>>,
}

impl StartupState {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            spawn_error: Mutex::new(None),
            backend_ready: Mutex::new(false),
            lines: Mutex::new(Vec::new()),
        }
    }

    pub fn push_line(&self, line: impl Into<String>) {
        let text = line.into();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        append_both_logs(trimmed);
        self.lines.lock().unwrap().push(trimmed.to_string());
    }

    pub fn set_spawn_error(&self, err: impl Into<String>) {
        let err = err.into();
        self.push_line(format!("sidecar spawn failed: {err}"));
        *self.spawn_error.lock().unwrap() = Some(err);
    }

    pub fn set_backend_ready(&self, ready: bool) {
        *self.backend_ready.lock().unwrap() = ready;
    }
}

pub fn logs_dir() -> PathBuf {
    user_data_root().join("logs")
}

pub fn user_data_root() -> PathBuf {
    if cfg!(target_os = "macos") {
        home_dir().join("Library/Application Support/dronmakr")
    } else if cfg!(target_os = "windows") {
        std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home_dir().join("AppData/Roaming"))
            .join("dronmakr")
    } else {
        home_dir().join(".local/share/dronmakr")
    }
}

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

pub fn init_log_files() -> (PathBuf, PathBuf) {
    let dir = logs_dir();
    let _ = create_dir_all(&dir);
    let startup = dir.join("startup.log");
    let errors = dir.join("errors.log");
    let stamp = unix_stamp();
    append_file(
        &startup,
        &format!("--- dronmakr app launch {stamp} ---"),
    );
    append_file(
        &errors,
        &format!("--- dronmakr app launch {stamp} ---"),
    );
    (startup, errors)
}

fn unix_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("({secs})")
}

fn append_both_logs(line: &str) {
    let dir = logs_dir();
    append_file(&dir.join("startup.log"), line);
    append_file(&dir.join("errors.log"), line);
}

fn append_file(path: &Path, line: &str) {
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartupDiagnostics {
    pub port: u16,
    pub backend_ready: bool,
    pub spawn_error: Option<String>,
    pub startup_log_path: String,
    pub errors_log_path: String,
    pub lines: Vec<String>,
}

#[tauri::command]
pub fn get_startup_diagnostics(state: tauri::State<'_, StartupState>) -> StartupDiagnostics {
    let dir = logs_dir();
    StartupDiagnostics {
        port: state.port,
        backend_ready: *state.backend_ready.lock().unwrap(),
        spawn_error: state.spawn_error.lock().unwrap().clone(),
        startup_log_path: dir.join("startup.log").to_string_lossy().into_owned(),
        errors_log_path: dir.join("errors.log").to_string_lossy().into_owned(),
        lines: state.lines.lock().unwrap().clone(),
    }
}
