"""Browse the recording CSVs under a dataset folder and plot their pressure curves.

Selecting a file in the tree redraws the plot immediately. Ctrl/Shift+click selects
several recordings to compare, and pinned recordings stay on the plot while you browse.
"""

from __future__ import annotations

import argparse
import math
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .recording import discover_recording_csv_files, load_recording_csv

VALUE_SCALE = 256.0  # raw values are multiples of 256 (8-bit sensor), so raw / 256 reads as 0-255
BASELINE_FRAMES = 5  # nobody has pressed yet in these frames; their mean is the static baseline
MAX_SEPARATE_PLOTS = 16
MAX_LEGEND_ENTRIES = 15


def recording_title(path: Path) -> str:
    """'Amber 10-29 15:45:14' for timestamped recordings, otherwise '<folder> <file stem>'."""
    try:
        stamp = datetime.strptime(path.stem, "%Y%m%d_%H%M%S_%f")
    except ValueError:
        return f"{path.parent.name} {path.stem}"
    return f"{path.parent.name} {stamp:%m-%d %H:%M:%S}"


class CurveViewerApp(tk.Tk):
    def __init__(self, dataset_root: Path) -> None:
        super().__init__()
        self.title("Pressure Curve Viewer")
        self.geometry("1280x760")
        self.minsize(960, 560)

        self.dataset_root_var = tk.StringVar(value=str(dataset_root))
        self.filter_var = tk.StringVar()
        self.view_var = tk.StringVar(value="overlay")
        self.curve_var = tk.StringVar(value="total")
        self.baseline_var = tk.BooleanVar(value=False)
        self.color_by_label_var = tk.BooleanVar(value=False)
        self.share_y_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar()

        self._files: list[Path] = []
        self._tree_paths: dict[str, Path] = {}
        self._cache: dict[Path, np.ndarray] = {}
        self._pinned: list[Path] = []
        self._redraw_job: str | None = None
        self._filter_job: str | None = None

        self._build_layout()
        self.filter_var.trace_add("write", lambda *_: self._schedule_filter())
        self.reload_files()

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=(12, 12, 12, 6))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Dataset root").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.dataset_root_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Browse", command=self._browse_dataset_root).grid(row=0, column=2)
        ttk.Button(top, text="Reload", command=self.reload_files).grid(row=0, column=3, padx=(8, 0))

        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.grid(row=1, column=0, sticky="nsew", padx=12)

        left = ttk.Frame(panes, padding=(0, 0, 8, 0))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        panes.add(left, weight=1)

        ttk.Label(left, text="Filter (folder or file name, e.g. Jay or 20261029)").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(left, textvariable=self.filter_var).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 8))

        self.tree = ttk.Treeview(left, show="tree", selectmode="extended")
        self.tree.grid(row=2, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        tree_scroll.grid(row=2, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._schedule_redraw())

        tree_buttons = ttk.Frame(left)
        tree_buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(tree_buttons, text="Pin selected", command=self._pin_selected).pack(side="left")
        ttk.Button(tree_buttons, text="Clear selection", command=self._clear_selection).pack(side="left", padx=(8, 0))

        pinned_frame = ttk.LabelFrame(left, text="Pinned (stay on the plot)", padding=8)
        pinned_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        pinned_frame.columnconfigure(0, weight=1)
        self.pinned_list = tk.Listbox(pinned_frame, height=6, selectmode="extended", exportselection=False)
        self.pinned_list.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(pinned_frame, text="Unpin", command=self._unpin_selected).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(pinned_frame, text="Clear pins", command=self._clear_pins).grid(row=1, column=1, sticky="e", pady=(8, 0))

        right = ttk.Frame(panes)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        panes.add(right, weight=4)

        options = ttk.Frame(right)
        options.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(options, text="View").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(options, text="Overlay", value="overlay", variable=self.view_var, command=self._schedule_redraw)\
            .grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Radiobutton(options, text="One plot each", value="separate", variable=self.view_var, command=self._schedule_redraw)\
            .grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(options, text="Curve").grid(row=0, column=3, sticky="w", padx=(24, 0))
        ttk.Radiobutton(options, text="Total (sum of cells)", value="total", variable=self.curve_var, command=self._schedule_redraw)\
            .grid(row=0, column=4, sticky="w", padx=(8, 0))
        ttk.Radiobutton(options, text="Each cell", value="cells", variable=self.curve_var, command=self._schedule_redraw)\
            .grid(row=0, column=5, sticky="w", padx=(8, 0))
        ttk.Checkbutton(options, text=f"Subtract baseline (mean of first {BASELINE_FRAMES} frames)",
                        variable=self.baseline_var, command=self._schedule_redraw)\
            .grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(options, text="Color by label", variable=self.color_by_label_var, command=self._schedule_redraw)\
            .grid(row=1, column=3, columnspan=2, sticky="w", padx=(24, 0), pady=(6, 0))
        ttk.Checkbutton(options, text="Share Y axis", variable=self.share_y_var, command=self._schedule_redraw)\
            .grid(row=1, column=5, sticky="w", padx=(8, 0), pady=(6, 0))

        self.figure = Figure(figsize=(8, 6), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        toolbar_frame = ttk.Frame(right)
        toolbar_frame.grid(row=2, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)  # zoom, pan and save-as-image

        ttk.Label(self, textvariable=self.status_var, padding=(12, 6)).grid(row=2, column=0, sticky="ew")

    def _browse_dataset_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.dataset_root_var.get() or str(Path.cwd()))
        if selected:
            self.dataset_root_var.set(selected)
            self.reload_files()

    def reload_files(self) -> None:
        self._files = discover_recording_csv_files(self.dataset_root_var.get())
        self._cache.clear()
        existing = set(self._files)
        self._pinned = [path for path in self._pinned if path in existing]
        self._refresh_pinned_list()
        self._populate_tree()

    def _schedule_filter(self) -> None:
        if self._filter_job is not None:
            self.after_cancel(self._filter_job)
        self._filter_job = self.after(200, self._populate_tree)

    def _populate_tree(self) -> None:
        self._filter_job = None
        selected = set(self._selected_paths())
        open_groups = {iid for iid in self.tree.get_children() if self.tree.item(iid, "open")}
        self.tree.delete(*self.tree.get_children())
        self._tree_paths.clear()

        root = Path(self.dataset_root_var.get())
        needle = self.filter_var.get().strip().lower()
        groups: dict[str, list[Path]] = {}
        for path in self._files:
            group = path.parent.relative_to(root).as_posix()
            if needle and needle not in f"{group}/{path.name}".lower():
                continue
            groups.setdefault(group, []).append(path)

        reselect = []
        for group, paths in groups.items():
            group_iid = f"group:{group}"
            self.tree.insert("", "end", iid=group_iid, text=f"{group}  ({len(paths)})",
                             open=bool(needle) or group_iid in open_groups)
            for path in paths:
                iid = f"file:{len(self._tree_paths)}"
                self._tree_paths[iid] = path
                self.tree.insert(group_iid, "end", iid=iid, text=path.stem)
                if path in selected:
                    reselect.append(iid)
        if reselect:
            self.tree.selection_set(reselect)
        self._schedule_redraw()

    def _selected_paths(self) -> list[Path]:
        # Folder rows are not recordings, so they are skipped.
        return [self._tree_paths[iid] for iid in self.tree.selection() if iid in self._tree_paths]

    def _clear_selection(self) -> None:
        self.tree.selection_set(())

    def _pin_selected(self) -> None:
        for path in self._selected_paths():
            if path not in self._pinned:
                self._pinned.append(path)
        self._refresh_pinned_list()
        self._schedule_redraw()

    def _unpin_selected(self) -> None:
        dropped = {self._pinned[index] for index in self.pinned_list.curselection()}
        self._pinned = [path for path in self._pinned if path not in dropped]
        self._refresh_pinned_list()
        self._schedule_redraw()

    def _clear_pins(self) -> None:
        self._pinned.clear()
        self._refresh_pinned_list()
        self._schedule_redraw()

    def _refresh_pinned_list(self) -> None:
        self.pinned_list.delete(0, "end")
        for path in self._pinned:
            self.pinned_list.insert("end", recording_title(path))

    def _schedule_redraw(self) -> None:
        # Holding an arrow key in the tree fires many selection events; draw once they settle.
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(30, self._redraw)

    def _load(self, path: Path) -> np.ndarray:
        """Frames of one recording as a (frames, cells) array in raw / VALUE_SCALE units."""
        if path not in self._cache:
            frames = load_recording_csv(path).frames
            self._cache[path] = np.array([frame.reshape(-1) for frame in frames], dtype=float) / VALUE_SCALE
        return self._cache[path]

    def _curves(self, values: np.ndarray) -> np.ndarray:
        """(frames, 1) total pressure, or (frames, cells) per-cell pressure."""
        if self.baseline_var.get():
            values = values - values[:BASELINE_FRAMES].mean(axis=0)
        if self.curve_var.get() == "total":
            return values.sum(axis=1, keepdims=True)
        return values

    def _redraw(self) -> None:
        self._redraw_job = None
        paths = self._pinned + [path for path in self._selected_paths() if path not in self._pinned]
        loaded: list[tuple[Path, np.ndarray]] = []
        failed: list[str] = []
        for path in paths:
            try:
                loaded.append((path, self._curves(self._load(path))))
            except (OSError, ValueError) as exc:
                failed.append(f"{path.name}: {exc}")

        notes = []
        self.figure.clear()
        if not loaded:
            ax = self.figure.add_subplot()
            ax.text(0.5, 0.5, "Select recordings on the left\n(Ctrl/Shift+click to compare several)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        elif self.view_var.get() == "overlay":
            self._draw_overlay(loaded)
        else:
            if len(loaded) > MAX_SEPARATE_PLOTS:
                notes.append(f"showing the first {MAX_SEPARATE_PLOTS} of {len(loaded)} (one plot each)")
                loaded = loaded[:MAX_SEPARATE_PLOTS]
            self._draw_separate(loaded)
        self.canvas.draw_idle()

        root = self.dataset_root_var.get()
        if not self._files:
            status = f"No CSV files found under {root}"
        else:
            status = f"{len(self._files)} recordings in {root}  |  plotted {len(loaded)}  |  pinned {len(self._pinned)}"
        if failed:
            notes.append(f"could not load {len(failed)} file(s), first: {failed[0]}")
        self.status_var.set("  |  ".join([status, *notes]))

    def _draw_overlay(self, loaded: list[tuple[Path, np.ndarray]]) -> None:
        ax = self.figure.add_subplot()
        by_label = self.color_by_label_var.get()
        labelled: set[str] = set()
        for (path, curves), color in zip(loaded, self._assign_colors([path for path, _ in loaded])):
            single = curves.shape[1] == 1
            lines = ax.plot(curves, color=color, linewidth=1.6 if single else 0.8, alpha=0.9 if single else 0.5)
            if not by_label:
                lines[0].set_label(self._line_label(path))
            elif path.parent.name not in labelled:
                labelled.add(path.parent.name)
                lines[0].set_label(path.parent.name)
        self._decorate(ax)
        ax.set_xlabel("Frame")
        ax.set_ylabel(self._y_label())
        handles, _ = ax.get_legend_handles_labels()
        if len(handles) <= MAX_LEGEND_ENTRIES:
            ax.legend(fontsize=8, loc="upper right")

    def _draw_separate(self, loaded: list[tuple[Path, np.ndarray]]) -> None:
        count = len(loaded)
        cols = min(4, math.ceil(math.sqrt(count)))
        rows = math.ceil(count / cols)
        axes = self.figure.subplots(rows, cols, sharex=True, sharey=self.share_y_var.get(), squeeze=False).ravel()
        for ax, (path, curves), color in zip(axes, loaded, self._assign_colors([path for path, _ in loaded])):
            if curves.shape[1] == 1:
                ax.plot(curves[:, 0], color=color, linewidth=1.4)
            else:
                for cell in range(curves.shape[1]):
                    ax.plot(curves[:, cell], color=colormaps["tab20"](cell % 20), linewidth=0.9, label=f"cell {cell}")
            ax.set_title(self._line_label(path), fontsize=9)
            self._decorate(ax)
        for index in range(count, len(axes)):
            axes[index].set_visible(False)
            # The plot above the empty slot is now the bottom of its column, so it needs the frame ticks.
            axes[index - cols].xaxis.set_tick_params(labelbottom=True)
        self.figure.supxlabel("Frame")
        self.figure.supylabel(self._y_label())
        if loaded[0][1].shape[1] > 1:
            legend = self.figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=8, fontsize=7)
            for line in legend.get_lines():
                line.set_linewidth(2.5)

    def _assign_colors(self, paths: list[Path]) -> list:
        if not self.color_by_label_var.get():
            return self._palette(len(paths))
        labels = list(dict.fromkeys(path.parent.name for path in paths))
        by_label = dict(zip(labels, self._palette(len(labels))))
        return [by_label[path.parent.name] for path in paths]

    @staticmethod
    def _palette(count: int) -> list:
        if count <= 10:
            return list(colormaps["tab10"].colors[:count])
        return [colormaps["turbo"](index / (count - 1)) for index in range(count)]

    def _line_label(self, path: Path) -> str:
        return ("[pin] " if path in self._pinned else "") + recording_title(path)

    def _y_label(self) -> str:
        name = "Total pressure" if self.curve_var.get() == "total" else "Cell pressure"
        if self.baseline_var.get():
            name += " - baseline"
        return f"{name} (raw / {VALUE_SCALE:g})"

    def _decorate(self, ax: Axes) -> None:
        ax.grid(True, alpha=0.3)
        if self.baseline_var.get():
            ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset", help="Folder holding <label>/<recording>.csv files.")
    args = parser.parse_args(argv)
    CurveViewerApp(Path(args.dataset_root)).mainloop()
