from __future__ import annotations

import threading
import time
import tkinter as tk
import serial
import serial.tools.list_ports
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .profiles import HEATMAP_PROFILE
from .recording import capture_and_save_recording, discover_recording_csv_files, load_recording_csv
from .config import SerialProfile
from .serial_io import open_serial, packet_to_grid, read_one_packet

class PressureMatrixApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Pressure Matrix Collector")
        self.geometry("980x620")
        self.minsize(900, 560)
        self.runtime_profile = HEATMAP_PROFILE
        self._live_thread: threading.Thread | None = None
        self._live_stop_event = threading.Event()
        self._live_serial: serial.Serial | None = None
        self._live_serial_lock = threading.Lock()
        self._live_frame_lock = threading.Lock()
        self._live_latest_frame = None
        self._live_latest_frame_count = 0
        self._live_drawn_frame_count = 0

        self.com_var = tk.StringVar(value=self.runtime_profile.port)
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
        self._refresh_com_ports()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(0, self._start_live_preview)


    

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

        ttk.Label(control_frame, text="COM Port").grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.com_box = ttk.Combobox(control_frame, textvariable=self.com_var, state="readonly")
        self.com_box.grid(row=2, column=1, sticky="ew", padx=(8, 24), pady=(12, 0))

        ttk.Button(control_frame, text="Refresh COM", command=self._refresh_com_ports)\
            .grid(row=2, column=2, sticky="ew", pady=(12, 0))

        ttk.Button(control_frame, text="Connect", command=self._connect_com)\
            .grid(row=2, column=3, sticky="ew", pady=(12, 0))

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

        blank_frame = [[0] * (self.runtime_profile.cols or 4) for _ in range(self.runtime_profile.rows or 4)]
        self._heatmap_image = self._axes.imshow(
            blank_frame,
            cmap="Blues",
            aspect="equal",
            vmin=self.runtime_profile.vmin,
            vmax=self.runtime_profile.vmax,
        )
        figure.colorbar(self._heatmap_image, ax=self._axes, fraction=0.046, pad=0.04)

        for row in range(self.runtime_profile.rows or 4):
            row_texts = []
            for col in range(self.runtime_profile.cols or 4):
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
        self.status_var.set("Preparing capture...")
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

        def launch_capture() -> None:
            self.status_var.set("Capturing live sensor data...")

            def worker() -> None:
                try:
                    result = capture_and_save_recording(
                        profile=self.runtime_profile,
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

        self._stop_live_preview(launch_capture)

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
        self._start_live_preview()

    def _capture_failed(self, exc: Exception) -> None:
        self._set_capture_ui_enabled(True)
        self.status_var.set("Capture failed.")
        messagebox.showerror("Capture failed", str(exc))
        self._start_live_preview()

    def _set_capture_ui_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (self.start_button,):
            widget.configure(state=state)

    def _refresh_com_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.com_box["values"] = ports

        if ports and self.com_var.get() not in ports:
            self.com_var.set(ports[0])

        selected_port = self.com_var.get().strip()
        if selected_port:
            self._set_runtime_port(selected_port)

    def _connect_com(self):
        port = self.com_var.get()

        if not port:
            messagebox.showerror("Error", "No COM selected")
            return

        self._set_runtime_port(port)
        self.status_var.set(f"Connected to {port}")

    def _set_runtime_port(self, port: str) -> None:
        self.runtime_profile = SerialProfile(
            port=port,
            baudrate=self.runtime_profile.baudrate,
            timeout=self.runtime_profile.timeout,
            header=self.runtime_profile.header,
            packet_size=self.runtime_profile.packet_size,
            rows=self.runtime_profile.rows,
            cols=self.runtime_profile.cols,
            value_dtype=self.runtime_profile.value_dtype,
            vmin=self.runtime_profile.vmin,
            vmax=self.runtime_profile.vmax,
        )

    def _start_live_preview(self) -> None:
        if self._live_thread and self._live_thread.is_alive():
            return

        self._live_stop_event.clear()
        with self._live_frame_lock:
            self._live_latest_frame = None
            self._live_latest_frame_count = 0
            self._live_drawn_frame_count = 0
        self._live_thread = threading.Thread(target=self._live_preview_worker, daemon=True)
        self._live_thread.start()
        self.after(0, self._refresh_live_preview_frame)

    def _stop_live_preview(self, on_stopped=None) -> None:
        self._live_stop_event.set()
        self._close_live_serial()
        if on_stopped is not None:
            self._wait_for_live_preview_stop(on_stopped)

    def _wait_for_live_preview_stop(self, on_stopped) -> None:
        if self._live_thread and self._live_thread.is_alive():
            self.after(50, lambda: self._wait_for_live_preview_stop(on_stopped))
            return

        self._live_thread = None
        on_stopped()

    def _close_live_serial(self) -> None:
        with self._live_serial_lock:
            live_serial = self._live_serial
            self._live_serial = None

        if live_serial is not None:
            try:
                live_serial.close()
            except Exception:
                pass

    def _publish_status(self, message: str) -> None:
        try:
            self.after(0, lambda message=message: self.status_var.set(message))
        except tk.TclError:
            pass

    def _live_preview_worker(self) -> None:
        live_serial = None
        connected_port = None
        frame_count = 0
        last_message = None

        try:
            while not self._live_stop_event.is_set():
                profile = self.runtime_profile
                port = profile.port.strip()

                if not port:
                    message = "No COM port selected for live preview."
                    if last_message != message:
                        self._publish_status(message)
                        last_message = message
                    self._close_live_serial()
                    live_serial = None
                    connected_port = None
                    time.sleep(0.5)
                    continue

                if live_serial is None or connected_port != port:
                    self._close_live_serial()
                    try:
                        live_serial = open_serial(profile)
                    except Exception as exc:
                        message = f"Live preview waiting for {port}: {exc}"
                        if last_message != message:
                            self._publish_status(message)
                            last_message = message
                        time.sleep(1.0)
                        continue

                    with self._live_serial_lock:
                        self._live_serial = live_serial

                    connected_port = port
                    message = f"Live preview connected to {port}."
                    if last_message != message:
                        self._publish_status(message)
                        last_message = message

                try:
                    packet = read_one_packet(live_serial, profile)
                except Exception as exc:
                    message = f"Live preview disconnected from {connected_port}: {exc}"
                    if last_message != message:
                        self._publish_status(message)
                        last_message = message
                    self._close_live_serial()
                    live_serial = None
                    connected_port = None
                    time.sleep(0.5)
                    continue

                if packet is None:
                    continue

                try:
                    frame = packet_to_grid(packet, profile)
                except Exception as exc:
                    message = f"Live preview data error: {exc}"
                    if last_message != message:
                        self._publish_status(message)
                        last_message = message
                    continue

                frame_count += 1
                with self._live_frame_lock:
                    self._live_latest_frame = frame
                    self._live_latest_frame_count = frame_count
        finally:
            self._close_live_serial()
            self._live_thread = None

    def _on_close(self) -> None:
        self._stop_live_preview()
        self.destroy()

    def _refresh_live_preview_frame(self) -> None:
        if self._live_stop_event.is_set():
            return

        frame_to_draw = None
        frame_count = 0
        with self._live_frame_lock:
            if self._live_latest_frame is not None and self._live_latest_frame_count != self._live_drawn_frame_count:
                frame_to_draw = self._live_latest_frame
                frame_count = self._live_latest_frame_count
                self._live_drawn_frame_count = frame_count

        if frame_to_draw is not None:
            try:
                self._update_heatmap(frame_to_draw, frame_count)
            except tk.TclError:
                return

        self.after(50, self._refresh_live_preview_frame)
    
        
def main() -> None:
    app = PressureMatrixApp()
    app.mainloop()
