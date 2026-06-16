"""
app
===
A minimal desktop launcher for non-technical operators:

    1. Launch (double-click a launcher, or `python -m model.app`).
    2. "Select image…"  -> pick a .czi file.
       The counter runs and the interactive viewer opens in the browser.
    3. Review; add / remove neurons by eye; click "✓ Accept & save count".
       The app saves a one-line count CSV next to the image and closes itself.

Outputs for ``/path/foo.czi`` go to ``/path/foo_results/`` (neurons.csv,
overlay.png, zviewer.html, foo_count.csv, …).

How "Accept" gets back to the app: the app runs a tiny localhost HTTP listener
and opens the viewer with ``?accept=http://127.0.0.1:<port>/accept``. Clicking
Accept pings that URL (an image request, no CORS needed); the app writes the
count CSV and closes.

Only Python's standard library is needed for the GUI + listener.
"""
from __future__ import annotations

import csv
import sys
import threading
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    _TK_OK = True
except Exception:  # pragma: no cover - headless / no Tk
    tk = None
    _TK_OK = False

from .viz_zslice import _load_config

# 1x1 transparent GIF returned to the viewer's Accept ping.
_GIF = (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!"
        b"\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00"
        b"\x00\x02\x01D\x00;")


# --------------------------------------------------------------------------- #
# Non-GUI helpers (unit-testable)
# --------------------------------------------------------------------------- #
def results_dir_for(image_path: Path) -> Path:
    return image_path.with_name(image_path.stem + "_results")


def write_count_csv(out_path: Path, image_name: str, count: int) -> Path:
    """Write a simplified CSV containing just the neuron count."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "neuron_count"])
        w.writerow([image_name, int(count)])
    return out_path


# --------------------------------------------------------------------------- #
# Localhost listener for the viewer's "Accept" ping
# --------------------------------------------------------------------------- #
def start_accept_server(on_accept) -> "ThreadingHTTPServer":
    """Start a 127.0.0.1 HTTP server; call ``on_accept(count:int)`` on /accept."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence console noise
            pass

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path.rstrip("/") == "/accept":
                q = urllib.parse.parse_qs(parsed.query)
                try:
                    n = int(float(q.get("count", ["0"])[0]))
                except Exception:
                    n = 0
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "image/gif")
                self.end_headers()
                try:
                    self.wfile.write(_GIF)
                except Exception:
                    pass
                try:
                    on_accept(n)
                except Exception:
                    pass
            else:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class CounterApp:
    BG = "#0f1012"
    FG = "#e7e7ea"
    SUBFG = "#9a9aa2"
    ACCENT = "#34c759"

    def __init__(self, root: "tk.Tk"):
        self.root = root
        self.cfg = _load_config(None)
        self.image_path: Path | None = None
        self.result = None
        self.httpd = None
        self._accepted = False
        self._busy = False

        root.title("Neuron Counter")
        root.configure(bg=self.BG)
        root.geometry("520x300")
        root.minsize(460, 260)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        tk.Label(root, text="Neuron Counter", bg=self.BG, fg=self.FG,
                 font=("Helvetica", 22, "bold")).pack(anchor="w", padx=22, pady=(22, 2))
        tk.Label(root, text="Select a .czi image — it is counted and opened for review.",
                 bg=self.BG, fg=self.SUBFG, font=("Helvetica", 12)).pack(anchor="w", padx=22)

        self.select_btn = tk.Button(root, text="Select image…", command=self.on_select,
                                    font=("Helvetica", 15, "bold"))
        self.select_btn.pack(pady=22)

        self.file_lbl = tk.Label(root, text="", bg=self.BG, fg=self.SUBFG,
                                 font=("Helvetica", 11))
        self.file_lbl.pack(anchor="w", padx=22)

        self.progress = ttk.Progressbar(root, mode="determinate", maximum=100, length=440)
        self.status_lbl = tk.Label(root, text="", bg=self.BG, fg=self.SUBFG,
                                   font=("Helvetica", 11), wraplength=470, justify="left")
        self.status_lbl.pack(anchor="w", padx=22, pady=(14, 0))

    # ---- step 1: pick + run --------------------------------------------- #
    def on_select(self):
        if self._busy:
            return
        path = filedialog.askopenfilename(
            title="Choose a .czi image",
            filetypes=[("Carl Zeiss image", "*.czi"), ("All files", "*.*")])
        if not path:
            return
        self.image_path = Path(path)
        self.file_lbl.config(text=self.image_path.name, fg=self.FG)
        self._busy = True
        self.select_btn.config(state="disabled")
        self.progress.pack(anchor="w", padx=22, pady=(14, 6))
        self.progress["value"] = 0
        self.status_lbl.config(
            text="Counting… the algorithm is running (this can take ~20–30 s). "
                 "The viewer opens automatically when it finishes.",
            fg=self.FG)
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _progress_cb(self, frac, msg):
        self.root.after(0, self._set_progress, frac, msg)

    def _set_progress(self, frac, msg):
        try:
            self.progress["value"] = max(0, min(100, frac * 100))
        except Exception:
            pass
        self.status_lbl.config(text=f"{msg}   ({int(frac * 100)}%)")

    def _run_pipeline(self):
        try:
            from .pipeline import run_pipeline
            out_dir = results_dir_for(self.image_path)
            result = run_pipeline(str(self.image_path), self.cfg,
                                  output_dir=str(out_dir), progress=self._progress_cb)
            self.root.after(0, self._on_done, result, None)
        except Exception:
            self.root.after(0, self._on_done, None, traceback.format_exc())

    # ---- step 2: open the viewer ---------------------------------------- #
    def _on_done(self, result, error):
        if error:
            self.progress.pack_forget()
            self._busy = False
            self.select_btn.config(state="normal")
            self.status_lbl.config(text="failed — pick another image")
            messagebox.showerror("Counting failed",
                                 "Something went wrong:\n\n" + error.strip().splitlines()[-1])
            return
        self.result = result
        viewer = Path(result.output_dir) / "zviewer.html"
        if not viewer.exists():
            self.progress.pack_forget()
            self._busy = False
            self.select_btn.config(state="normal")
            messagebox.showwarning("Viewer not found",
                                   "The interactive viewer was not generated.")
            return
        # Keep the bar visible (full) while the browser is launched, so it is
        # clear the wait was real computation and not a hang.
        self.progress["value"] = 100
        self.status_lbl.config(text="Done — opening the viewer in your browser…",
                               fg=self.FG)
        # start the accept-listener and open the viewer pointed at it
        if self.httpd is None:
            self.httpd = start_accept_server(self._on_accept_ping)
        port = self.httpd.server_address[1]
        accept_url = f"http://127.0.0.1:{port}/accept"
        uri = (viewer.resolve().as_uri()
               + "?accept=" + urllib.parse.quote(accept_url, safe="")
               + "&v=" + str(int(viewer.stat().st_mtime)))
        webbrowser.open(uri, new=2)
        self.progress.pack_forget()
        self.status_lbl.config(
            text=(f"Counted {result.n_neurons} neurons.\n"
                  "Review in your browser — add / remove as needed, then click "
                  "“✓ Accept & save count”.\nThis window will close automatically."),
            fg=self.FG)

    # ---- step 3: accepted in the browser -> save + close ---------------- #
    def _on_accept_ping(self, count):
        # called from the HTTP server thread -> marshal onto the Tk thread
        self.root.after(0, self._finish, int(count))

    def _finish(self, count):
        if self._accepted:
            return
        self._accepted = True
        out = Path(self.result.output_dir) / (self.image_path.stem + "_count.csv")
        try:
            write_count_csv(out, self.image_path.name, count)
        except Exception:
            pass
        self.status_lbl.config(text=f"✓ Accepted — {count} neurons saved to {out.name}. Closing…",
                               fg=self.ACCENT)
        self.root.after(1300, self._shutdown)

    def _on_close(self):
        self._shutdown()

    def _shutdown(self):
        if self.httpd is not None:
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()
        try:
            self.root.destroy()
        except Exception:
            pass


def main(argv=None) -> int:
    if not _TK_OK:
        sys.stderr.write(
            "Tkinter (Python's GUI toolkit) is not available in this Python.\n"
            "On macOS install a python.org build or `brew install python-tk`; "
            "on Debian/Ubuntu `sudo apt install python3-tk`.\n")
        return 1
    root = tk.Tk()
    CounterApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
