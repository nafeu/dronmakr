fn main() {
    let icons_dir = std::path::Path::new("icons");
    if icons_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(icons_dir) {
            for entry in entries.flatten() {
                println!("cargo:rerun-if-changed={}", entry.path().display());
            }
        }
    }
    tauri_build::build()
}
