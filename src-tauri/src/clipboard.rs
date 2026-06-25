#[cfg(target_os = "macos")]
fn copy_files_impl(paths: &[String]) -> Result<(), String> {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_app_kit::{NSPasteboard, NSPasteboardWriting};
    use objc2_foundation::{NSArray, NSString, NSURL};

    if paths.is_empty() {
        return Err("No file paths provided.".into());
    }

    for path in paths {
        if !std::path::Path::new(path).is_file() {
            return Err(format!("File not found: {path}"));
        }
    }

    let pasteboard = NSPasteboard::generalPasteboard();
    let _ = pasteboard.clearContents();

    let urls: Vec<Retained<NSURL>> = paths
        .iter()
        .map(|path| NSURL::fileURLWithPath_isDirectory(&NSString::from_str(path), false))
        .collect();

    let writers: Vec<Retained<ProtocolObject<dyn NSPasteboardWriting>>> = urls
        .iter()
        .map(|url| ProtocolObject::from_retained(url.clone()))
        .collect();

    let array = NSArray::from_retained_slice(&writers);
    if pasteboard.writeObjects(&array) {
        Ok(())
    } else {
        Err("Failed to write files to the clipboard.".into())
    }
}

#[cfg(not(target_os = "macos"))]
fn copy_files_impl(_paths: &[String]) -> Result<(), String> {
    Err("Copying files to the clipboard is only supported on macOS.".into())
}

#[tauri::command]
pub async fn copy_files_to_clipboard(
    app: tauri::AppHandle,
    paths: Vec<String>,
) -> Result<(), String> {
    let (tx, rx) = std::sync::mpsc::channel();
    app.run_on_main_thread(move || {
        let result = copy_files_impl(&paths);
        let _ = tx.send(result);
    })
    .map_err(|err| err.to_string())?;

    rx.recv()
        .map_err(|_| "Clipboard operation was interrupted.".to_string())?
}
