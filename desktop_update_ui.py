"""Tk update window runs in its own process (started with desktop_app.py --update-ui)."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from desktop_update_install import apply_download_and_launch
from updater import UpdateInfo, check_for_update
from version import __version__


def _truncate_notes(notes: str, limit: int = 1200) -> str:
    n = notes.strip()
    if len(n) <= limit:
        return n
    return n[: limit - 40] + "\n\n…(release notes truncated)…"


def main() -> None:
    root = tk.Tk()
    root.title("dronmakr updater")
    root.geometry("520x420")

    info_var: dict[str, UpdateInfo | None] = {"v": None}
    pending: dict[str, bool] = {"busy": False}

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frm, text=f"Installed version: {__version__}", font=("TkDefaultFont", 11)).pack(
        anchor=tk.W, pady=(0, 6)
    )
    status_var = tk.StringVar(value='Click "Check for updates".')
    ttk.Label(frm, textvariable=status_var, wraplength=480).pack(anchor=tk.W)

    gap = ttk.Frame(frm, height=10)
    gap.pack(fill=tk.X)

    notes_frm = ttk.Frame(frm)
    notes_frm.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
    ttk.Label(notes_frm, text="Latest release notes:").pack(anchor=tk.W)
    txt = tk.Text(notes_frm, height=12, wrap=tk.WORD, state=tk.DISABLED, background="#222", foreground="#f0dcc8")
    sb = ttk.Scrollbar(notes_frm, orient=tk.VERTICAL, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)

    txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _set_notes(s: str) -> None:
        txt.configure(state=tk.NORMAL)
        txt.delete("1.0", tk.END)
        txt.insert(tk.END, s or "")
        txt.configure(state=tk.DISABLED)

    _set_notes("(Run a check.)")

    pro = ttk.Progressbar(frm, mode="indeterminate")
    pro.pack(fill=tk.X, pady=6)

    def set_busy(busy: bool) -> None:
        pending["busy"] = busy
        if busy:
            pro.start(14)
            btn_chk.configure(state=tk.DISABLED)
            btn_install.configure(state=tk.DISABLED)
            btn_close.configure(state=tk.DISABLED)
        else:
            pro.stop()
            btn_chk.configure(state=tk.NORMAL)
            btn_close.configure(state=tk.NORMAL)
            btn_install.configure(state=tk.NORMAL if info_var["v"] is not None else tk.DISABLED)

    def run_check_worker() -> None:
        try:
            u = check_for_update(timeout=12)
        except Exception as e:  # noqa: BLE001
            root.after(0, lambda: _after_check(None, error=str(e)))
            return

        root.after(0, lambda: _after_check(u, error=None))

    def _after_check(u: UpdateInfo | None, error: str | None) -> None:
        set_busy(False)
        if error:
            messagebox.showerror("Update check failed", error, parent=root)
            status_var.set("Could not reach GitHub.")
            btn_install.configure(state=tk.DISABLED)
            return
        info_var["v"] = u
        if not u:
            status_var.set("You're on the latest public release.")
            _set_notes("")
            btn_install.configure(state=tk.DISABLED)
            return
        status_var.set(f"Update available: {u.tag} ({u.asset_name})")
        _set_notes(_truncate_notes(u.notes))
        btn_install.configure(state=tk.NORMAL)

    def on_check() -> None:
        status_var.set("Checking GitHub…")
        btn_install.configure(state=tk.DISABLED)
        set_busy(True)
        threading.Thread(target=run_check_worker, daemon=True).start()

    def run_install_worker() -> None:
        u = info_var["v"]
        if not u:

            def _noop() -> None:
                messagebox.showinfo("Update", "No update selected — run Check first.", parent=root)
                set_busy(False)

            root.after(0, _noop)
            return
        msg, launched = apply_download_and_launch(u)

        def _done() -> None:
            messagebox.showinfo("Update", msg, parent=root)
            set_busy(False)
            if launched:
                root.quit()

        root.after(0, _done)

    def on_install() -> None:
        if info_var["v"] is None:
            messagebox.showinfo("Update", "Check for updates first.", parent=root)
            return
        set_busy(True)
        threading.Thread(target=run_install_worker, daemon=True).start()

    bf = ttk.Frame(frm)
    bf.pack(fill=tk.X, pady=(6, 0))
    btn_chk = ttk.Button(bf, text="Check for updates", command=on_check)
    btn_chk.pack(side=tk.LEFT)

    btn_install = ttk.Button(bf, text="Download + install …", command=on_install, state=tk.DISABLED)
    btn_install.pack(side=tk.LEFT, padx=(8, 0))

    btn_close = ttk.Button(bf, text="Close", command=root.quit)
    btn_close.pack(side=tk.RIGHT)

    txt.bind("<1>", lambda e: txt.focus_set())
    root.mainloop()


if __name__ == "__main__":
    main()
