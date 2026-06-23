use std::net::TcpListener;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::{AppHandle, Emitter, Manager, RunEvent, Url};
use tauri_plugin_opener::OpenerExt;
#[cfg(not(debug_assertions))]
use tauri_plugin_shell::process::CommandEvent;
#[cfg(not(debug_assertions))]
use tauri_plugin_shell::ShellExt;

const PREFERRED_PORTS: [u16; 4] = [3766, 3767, 3768, 3769];
const DEV_PORT: u16 = 3766;

struct BackendState {
    port: u16,
    child: Mutex<Option<tauri_plugin_shell::process::CommandChild>>,
}

fn can_bind_port(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}

fn select_listen_port() -> u16 {
    for port in PREFERRED_PORTS {
        if can_bind_port(port) {
            return port;
        }
    }
    TcpListener::bind(("127.0.0.1", 0))
        .ok()
        .and_then(|listener| listener.local_addr().ok().map(|addr| addr.port()))
        .unwrap_or(DEV_PORT)
}

fn wait_for_health(port: u16, timeout: Duration) -> bool {
    let url = format!("http://127.0.0.1:{port}/api/health");
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(resp) = ureq::get(&url).timeout(Duration::from_secs(2)).call() {
            if resp.status() / 100 == 2 {
                return true;
            }
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}

fn navigate_main(app: &AppHandle, port: u16) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    let url = format!("http://127.0.0.1:{port}/");
    if let Ok(parsed) = Url::parse(&url) {
        let _ = window.navigate(parsed);
    }
}

#[cfg(not(debug_assertions))]
fn spawn_backend_sidecar(
    app: &AppHandle,
    port: u16,
) -> Result<tauri_plugin_shell::process::CommandChild, String> {
    let sidecar = app
        .shell()
        .sidecar("dronmakr-backend")
        .map_err(|e| format!("sidecar missing: {e}"))?;
    let (mut rx, child) = sidecar
        .args(["--port", &port.to_string()])
        .spawn()
        .map_err(|e| format!("failed to spawn backend: {e}"))?;

    let app_handle = app.clone();
    thread::spawn(move || {
        while let Some(event) = rx.blocking_recv() {
            if let CommandEvent::Stdout(line) = event {
                let _ = app_handle.emit("backend-log", String::from_utf8_lossy(&line).to_string());
            }
        }
    });

    Ok(child)
}

fn open_files_root(app: &AppHandle) {
    let app = app.clone();
    thread::spawn(move || {
        let port = app.state::<BackendState>().port;
        let url = format!("http://127.0.0.1:{port}/api/settings");
        let Ok(resp) = ureq::get(&url).call() else {
            return;
        };
        let Ok(settings) = resp.into_json::<serde_json::Value>() else {
            return;
        };
        let Some(root) = settings.get("FILES_ROOT").and_then(|v| v.as_str()) else {
            return;
        };
        if root.is_empty() {
            return;
        }
        let _ = app.opener().open_path(root, None::<&str>);
    });
}

fn navigate_to(app: &AppHandle, path: &str) {
    let port = app.state::<BackendState>().port;
    let url = format!("http://127.0.0.1:{port}{path}");
    if let Some(window) = app.get_webview_window("main") {
        if let Ok(parsed) = Url::parse(&url) {
            let _ = window.navigate(parsed);
        }
    }
}

fn build_menu(app: &AppHandle) -> tauri::Result<Menu<tauri::Wry>> {
    let app_menu = Submenu::with_items(
        app,
        "dronmakr",
        true,
        &[
            &MenuItem::with_id(app, "open_files", "Open generated files folder", true, None::<&str>)?,
            &PredefinedMenuItem::separator(app)?,
            &MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?,
            &MenuItem::with_id(app, "about", "About", true, None::<&str>)?,
            &MenuItem::with_id(app, "report_issue", "Report issue", true, None::<&str>)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::quit(app, Some("Quit dronmakr"))?,
        ],
    )?;
    Menu::with_items(app, &[&app_menu])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let backend_port = if cfg!(debug_assertions) {
        DEV_PORT
    } else {
        select_listen_port()
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_macos_fps::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_drag::init())
        .manage(BackendState {
            port: backend_port,
            child: Mutex::new(None),
        })
        .setup(move |app| {
            let menu = build_menu(app.handle())?;
            app.set_menu(menu)?;

            app.on_menu_event(|app, event| match event.id().as_ref() {
                "open_files" => open_files_root(app),
                "settings" => navigate_to(app, "/settings"),
                "about" => navigate_to(app, "/about"),
                "report_issue" => navigate_to(app, "/about"),
                _ => {}
            });

            let handle = app.handle().clone();
            let startup_port = backend_port;

            thread::spawn(move || {
                #[cfg(not(debug_assertions))]
                {
                    match spawn_backend_sidecar(&handle, startup_port) {
                        Ok(child) => {
                            if let Some(state) = handle.try_state::<BackendState>() {
                                *state.child.lock().unwrap() = Some(child);
                            }
                        }
                        Err(err) => {
                            let _ = handle.emit("backend-error", err);
                            return;
                        }
                    }
                }

                if wait_for_health(startup_port, Duration::from_secs(45)) {
                    let _ = handle.run_on_main_thread({
                        let handle = handle.clone();
                        move || navigate_main(&handle, startup_port)
                    });
                } else {
                    let _ = handle.emit(
                        "backend-error",
                        "Backend did not become ready in time",
                    );
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(state) = app.try_state::<BackendState>() {
                    if let Some(child) = state.child.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
            }
        });
}
