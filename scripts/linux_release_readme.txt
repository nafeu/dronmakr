dronmakr for Linux (experimental)
=================================

Linux desktop builds are published for testing and early adopters. They are not
fully supported yet. Expect rough edges, distro-specific issues, and missing
polish compared to macOS and Windows.

Release builds are **x86_64 (AMD64) only** for now.

We still welcome bug reports (especially with errors.log). Reports help us
prioritize Linux stability work.

Install
-------
Choose one format from the release:

- .tar.gz: extract anywhere, then run the dronmakr binary from the bundle
  (see deb/rpm folders inside the archive for package layouts).
- .deb: Debian/Ubuntu-derived distros: sudo dpkg -i dronmakr_*_amd64.deb
- .rpm: Fedora/RHEL/Arch (with rpm tools): sudo rpm -i dronmakr-*-1.x86_64.rpm

On first launch, choose where to store your dronmakr-files folder (generated
audio, presets, config). You can type the path manually if the folder picker
does not open.

Folder picker (optional)
------------------------
Native folder buttons need one of:

  zenity   (common on GNOME-based distros)
  kdialog  (KDE)
  yad

Example (Arch): sudo pacman -S zenity

If none are installed, type paths into the text fields instead.

Requirements
------------
- x86_64 Linux with WebKitGTK-based desktop (standard on most distros)
- VST3 plug-ins for preset-based drone generation (Linux VST hosting is less
  mature than on macOS/Windows)
- libsndfile (usually pulled in by the package or distro deps)

Logs and support
----------------
Error log: ~/.local/share/dronmakr/logs/errors.log

Report issues: https://github.com/nafeu/dronmakr/issues
Include your distro, install method (deb/rpm/tar.gz), and relevant log lines.

Updates
-------
Download the latest release from GitHub and reinstall (deb/rpm) or replace the
extracted bundle (tar.gz).
