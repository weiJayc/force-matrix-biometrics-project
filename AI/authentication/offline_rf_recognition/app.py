from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from .recognizer import RecognitionError, append_log, load_model, recognize, THRESHOLD
except ImportError:
    from recognizer import RecognitionError, append_log, load_model, recognize, THRESHOLD


class OfflineRFApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Random Forest Offline CSV Recognizer")
        self.geometry("760x620")
        self.minsize(700, 560)
        self.selected_csv: Path | None = None
        self.model = None

        self.file_name = tk.StringVar(value="No CSV selected")
        self.predicted = tk.StringVar(value="-")
        self.confidence = tk.StringVar(value="-")
        self.threshold = tk.StringVar(value=f"{THRESHOLD:.6f}")
        self.decision = tk.StringVar(value="-")
        self.error = tk.StringVar(value="")

        self._build_ui()
        self._load_model()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Button(frame, text="Select CSV", command=self._select_csv).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(frame, textvariable=self.file_name).grid(
            row=0, column=1, columnspan=2, sticky="w", padx=(12, 0)
        )

        ttk.Button(frame, text="Run Recognition", command=self._run).grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )

        results = ttk.LabelFrame(frame, text="Recognition Result", padding=12)
        results.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        results.columnconfigure(1, weight=1)

        for row, (label, variable) in enumerate(
            [
                ("Predicted Identity", self.predicted),
                ("Confidence", self.confidence),
                ("Threshold", self.threshold),
                ("Threshold Decision", self.decision),
            ]
        ):
            ttk.Label(results, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Label(results, textvariable=variable).grid(
                row=row, column=1, sticky="w", padx=(12, 0), pady=3
            )

        probability_frame = ttk.LabelFrame(frame, text="Class Probabilities", padding=12)
        probability_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        frame.rowconfigure(3, weight=1)
        probability_frame.rowconfigure(0, weight=1)
        probability_frame.columnconfigure(0, weight=1)

        self.probability_list = tk.Text(probability_frame, height=10, state="disabled")
        self.probability_list.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, textvariable=self.error, foreground="red", wraplength=680).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(12, 0)
        )

        ttk.Label(
            frame,
            text="Offline experimental threshold only; not an independently calibrated security gate.",
            wraplength=680,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def _load_model(self) -> None:
        try:
            self.model = load_model()
        except RecognitionError as error:
            self.error.set(str(error))

    def _select_csv(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select a 50x16 CSV recording",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filename:
            self.selected_csv = Path(filename)
            self.file_name.set(self.selected_csv.name)
            self.error.set("")

    def _run(self) -> None:
        if self.model is None:
            messagebox.showerror("Model Error", self.error.get() or "Model is not loaded.")
            return
        if self.selected_csv is None:
            messagebox.showerror("No CSV Selected", "Please select a CSV file first.")
            return
        try:
            result = recognize(self.selected_csv, self.model)
            append_log(result)
            self.predicted.set(result["predicted_identity"])
            self.confidence.set(f"{result['confidence']:.6f}")
            self.threshold.set(f"{result['threshold']:.6f}")
            self.decision.set("ACCEPT - 辨識通過" if result["acceptance"] == "Accept" else "拒絕辨識")
            self.error.set("")

            lines = [
                f"{class_name}: {probability:.6f}"
                for class_name, probability in result["probabilities"]
            ]
            self.probability_list.configure(state="normal")
            self.probability_list.delete("1.0", "end")
            self.probability_list.insert("1.0", "\n".join(lines))
            self.probability_list.configure(state="disabled")
        except RecognitionError as error:
            self.error.set(str(error))
            messagebox.showerror("Recognition Error", str(error))


def main() -> None:
    OfflineRFApp().mainloop()


if __name__ == "__main__":
    main()
