from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .profiles import HEATMAP_PROFILE
from .recording import capture_and_save_recording, discover_recording_csv_files, load_recording_csv


class PressureMatrixApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Pressure Matrix Collector")
        self.geometry("980x620")
        self.minsize(900, 560)

        self.label_var = tk.StringVar(value="class_a")
        self.duration_var = tk.StringVar(value="5")
        self.frame_count_var = tk.StringVar(value="64")
        self.dataset_root_var = tk.StringVar(value=str(Path.cwd() / "dataset"))
        self.status_var = tk.StringVar(value="Ready to capture a labeled recording.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.frame_count_status_var = tk.StringVar(value="0 frames captured")
        self.preview_status_var = tk.StringVar(value="No saved recording selected.")

        self._capture_thread: threading.Thread | None = None
        self._heatmap_image = None
        self._heatmap_texts: list[list[object]] = []
        self._preview_files: list[Path] = []
        self._current_preview_index = 0

        self._build_layout()
        self._build_heatmap()
        self.refresh_dataset_browser()

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        control_frame = ttk.Frame(self, padding=16)
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(3, weight=1)

        ttk.Label(control_frame, text="Class label").grid(row=0, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.label_var, width=24).grid(row=0, column=1, sticky="ew", padx=(8, 24))

        ttk.Label(control_frame, text="Duration (seconds)").grid(row=0, column=2, sticky="w")
        ttk.Entry(control_frame, textvariable=self.duration_var, width=12).grid(row=0, column=3, sticky="ew", padx=(8, 24))

        ttk.Label(control_frame, text="Fixed frames").grid(row=0, column=4, sticky="w")
        ttk.Entry(control_frame, textvariable=self.frame_count_var, width=12).grid(row=0, column=5, sticky="ew", padx=(8, 24))

        ttk.Label(control_frame, text="Dataset root").grid(row=1, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(control_frame, textvariable=self.dataset_root_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=(8, 8), pady=(12, 0))
        ttk.Button(control_frame, text="Browse", command=self._browse_dataset_root).grid(row=1, column=4, sticky="ew", pady=(12, 0))

        self.start_button = ttk.Button(control_frame, text="Start Capture", command=self._start_capture)
        self.start_button.grid(row=1, column=5, sticky="ew", padx=(24, 0), pady=(12, 0))

        progress_frame = ttk.Frame(self, padding=(16, 0, 16, 16))
        progress_frame.grid(row=1, column=0, sticky="nsew")
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(3, weight=1)

        ttk.Label(progress_frame, textvariable=self.status_var, wraplength=420).grid(row=0, column=0, sticky="w")
        ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=1.0).grid(row=1, column=0, sticky="ew", pady=(10, 8))
        ttk.Label(progress_frame, textvariable=self.frame_count_status_var).grid(row=2, column=0, sticky="nw")

        browser_frame = ttk.LabelFrame(progress_frame, text="Saved recordings", padding=12)
        browser_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        browser_frame.columnconfigure(0, weight=1)
        browser_frame.rowconfigure(1, weight=1)

        ttk.Label(browser_frame, textvariable=self.preview_status_var, wraplength=380).grid(row=0, column=0, sticky="w")
        self.recording_list = tk.Listbox(browser_frame, height=12)
        self.recording_list.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        self.recording_list.bind("<<ListboxSelect>>", self._on_recording_selected)

        browser_buttons = ttk.Frame(browser_frame)
        browser_buttons.grid(row=2, column=0, sticky="ew")
        browser_buttons.columnconfigure(0, weight=1)
        ttk.Button(browser_buttons, text="Refresh", command=self.refresh_dataset_browser).grid(row=0, column=0, sticky="w")
        ttk.Button(browser_buttons, text="Preview selected", command=self.preview_selected_recording).grid(row=0, column=1, sticky="e")

        canvas_frame = ttk.Frame(self, padding=(0, 0, 16, 16))
        canvas_frame.grid(row=1, column=1, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas_frame = canvas_frame

    def _build_heatmap(self) -> None:
        figure = Figure(figsize=(4.8, 4.8), dpi=100)
        self._axes = figure.add_subplot(111)
        self._axes.set_title("Live Heatmap")
        self._axes.set_xlabel("Column")
        self._axes.set_ylabel("Row")

        blank_frame = [[0] * (HEATMAP_PROFILE.cols or 4) for _ in range(HEATMAP_PROFILE.rows or 4)]
        self._heatmap_image = self._axes.imshow(
            blank_frame,
            cmap="Blues",
            aspect="equal",
            vmin=HEATMAP_PROFILE.vmin,
            vmax=HEATMAP_PROFILE.vmax,
        )
        figure.colorbar(self._heatmap_image, ax=self._axes, fraction=0.046, pad=0.04)

        for row in range(HEATMAP_PROFILE.rows or 4):
            row_texts = []
            for col in range(HEATMAP_PROFILE.cols or 4):
                text = self._axes.text(
                    col,
                    row,
                    "0",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                )
                row_texts.append(text)
            self._heatmap_texts.append(row_texts)

        canvas = FigureCanvasTkAgg(figure, master=self.canvas_frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        canvas.draw()
        self._canvas = canvas

    def _browse_dataset_root(self) -> None:
        selected_path = filedialog.askdirectory(initialdir=self.dataset_root_var.get() or str(Path.cwd()))
        if selected_path:
            self.dataset_root_var.set(selected_path)
            self.refresh_dataset_browser()

    def refresh_dataset_browser(self) -> None:
        dataset_root = Path(self.dataset_root_var.get()).expanduser()
        self._preview_files = discover_recording_csv_files(dataset_root)

        self.recording_list.delete(0, tk.END)
        for path in self._preview_files:
            try:
                relative_path = path.relative_to(dataset_root)
            except ValueError:
                relative_path = path
            self.recording_list.insert(tk.END, str(relative_path))

        if self._preview_files:
            self.preview_status_var.set(f"Found {len(self._preview_files)} saved recordings.")
        else:
            self.preview_status_var.set("No saved recordings found in the selected dataset folder.")

    def _on_recording_selected(self, _event) -> None:
        selection = self.recording_list.curselection()
        if selection:
            self._current_preview_index = selection[0]

    def preview_selected_recording(self) -> None:
        if not self._preview_files:
            messagebox.showinfo("Preview", "No saved recordings are available yet.")
            return

        selection = self.recording_list.curselection()
        index = selection[0] if selection else self._current_preview_index
        if index < 0 or index >= len(self._preview_files):
            index = 0

        path = self._preview_files[index]
        try:
            loaded = load_recording_csv(path)
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc))
            return

        self._show_preview_frame(loaded.frames[0])
        self.preview_status_var.set(f"Previewing {loaded.label}: {path.name}")

    def _show_preview_frame(self, frame) -> None:
        self._heatmap_image.set_data(frame)
        for row_index, row in enumerate(frame):
            for col_index, value in enumerate(row):
                self._heatmap_texts[row_index][col_index].set_text(str(int(value)))
        self._canvas.draw_idle()

    def _start_capture(self) -> None:
        if self._capture_thread and self._capture_thread.is_alive():
            return

        try:
            duration_seconds = float(self.duration_var.get())
            target_frame_count = int(self.frame_count_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Duration and fixed frames must be numeric values.")
            return

        label = self.label_var.get().strip()
        if not label:
            messagebox.showerror("Missing label", "Please enter a class label before starting capture.")
            return

        dataset_root = Path(self.dataset_root_var.get()).expanduser()
        self._set_capture_ui_enabled(False)
        self.progress_var.set(0.0)
        self.status_var.set("Capturing live sensor data...")
        self.frame_count_status_var.set("0 frames captured")

        def frame_callback(frame, frame_count: int) -> None:
            self.after(0, lambda frame=frame, frame_count=frame_count: self._update_heatmap(frame, frame_count))

        def progress_callback(progress: float, elapsed_seconds: float, frame_count: int) -> None:
            self.after(
                0,
                lambda progress=progress, elapsed_seconds=elapsed_seconds, frame_count=frame_count: self._update_progress(
                    progress,
                    elapsed_seconds,
                    frame_count,
                ),
            )

        def worker() -> None:
            try:
                result = capture_and_save_recording(
                    profile=HEATMAP_PROFILE,
                    dataset_root=dataset_root,
                    label=label,
                    duration_seconds=duration_seconds,
                    target_frame_count=target_frame_count,
                    frame_callback=frame_callback,
                    progress_callback=progress_callback,
                )
            except Exception as exc:  # pragma: no cover - surfaced in the UI
                self.after(0, lambda exc=exc: self._capture_failed(exc))
                return

            self.after(0, lambda result=result: self._capture_succeeded(result))

        self._capture_thread = threading.Thread(target=worker, daemon=True)
        self._capture_thread.start()

    def _update_heatmap(self, frame, frame_count: int) -> None:
        self._heatmap_image.set_data(frame)
        for row_index, row in enumerate(frame):
            for col_index, value in enumerate(row):
                self._heatmap_texts[row_index][col_index].set_text(str(int(value)))
        self._canvas.draw_idle()
        self.frame_count_status_var.set(f"{frame_count} frames captured")

    def _update_progress(self, progress: float, elapsed_seconds: float, frame_count: int) -> None:
        self.progress_var.set(progress)
        self.status_var.set(
            f"Recording in progress: {elapsed_seconds:.1f}s elapsed, {frame_count} valid frames collected."
        )

    def _capture_succeeded(self, result) -> None:
        self._set_capture_ui_enabled(True)
        self.progress_var.set(1.0)
        self.status_var.set(
            f"Saved {result.normalized_frame_count} frames to {result.csv_path}"
        )
        self.frame_count_status_var.set(
            f"{result.raw_frame_count} raw frames -> {result.normalized_frame_count} normalized frames"
        )
        self.refresh_dataset_browser()

    def _capture_failed(self, exc: Exception) -> None:
        self._set_capture_ui_enabled(True)
        self.status_var.set("Capture failed.")
        messagebox.showerror("Capture failed", str(exc))

    def _set_capture_ui_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (self.start_button,):
            widget.configure(state=state)


def main() -> None:
    app = PressureMatrixApp()
    app.mainloop()
