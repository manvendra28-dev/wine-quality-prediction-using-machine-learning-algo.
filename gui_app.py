"""
Wine Quality Prediction Graphical User Interface (GUI).
Built with Python Tkinter & TTK with a modern, elegant theme.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.predict import WineQualityPredictor


class WineQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wine Quality AI Predictor - Machine Learning System")
        self.root.geometry("1020x760")
        self.root.minsize(960, 720)
        self.root.configure(bg="#F4F5F7")

        # Load trained predictor
        try:
            self.predictor = WineQualityPredictor("models/best_wine_model_combined.joblib")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load trained model:\n{e}")
            sys.exit(1)

        self.entries = {}
        self.wine_type_var = tk.IntVar(value=0)  # 0: White, 1: Red

        self.setup_styles()
        self.build_ui()
        self.load_preset_premium()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Color palette
        self.c_primary = "#6B1D2F"      # Bordeaux Burgundy
        self.c_primary_hover = "#86223A"
        self.c_bg = "#F4F5F7"
        self.c_card = "#FFFFFF"
        self.c_text = "#1E293B"
        self.c_muted = "#64748B"
        self.c_border = "#E2E8F0"
        self.c_success = "#16A34A"
        self.c_warning = "#D97706"

        self.style.configure(".", background=self.c_bg, foreground=self.c_text, font=("Segoe UI", 10))
        
        # Frames
        self.style.configure("Card.TFrame", background=self.c_card, relief="flat")
        self.style.configure("Header.TFrame", background=self.c_primary)

        # Labels
        self.style.configure("HeaderTitle.TLabel", background=self.c_primary, foreground="#FFFFFF", font=("Segoe UI", 18, "bold"))
        self.style.configure("HeaderSub.TLabel", background=self.c_primary, foreground="#FBD5D5", font=("Segoe UI", 10))
        self.style.configure("CardTitle.TLabel", background=self.c_card, foreground=self.c_primary, font=("Segoe UI", 13, "bold"))
        self.style.configure("FieldLabel.TLabel", background=self.c_card, foreground=self.c_text, font=("Segoe UI", 9, "bold"))
        self.style.configure("FieldUnit.TLabel", background=self.c_card, foreground=self.c_muted, font=("Segoe UI", 8))
        self.style.configure("Muted.TLabel", background=self.c_card, foreground=self.c_muted, font=("Segoe UI", 9))

        # Buttons
        self.style.configure(
            "Primary.TButton",
            background=self.c_primary,
            foreground="#FFFFFF",
            font=("Segoe UI", 11, "bold"),
            padding=(16, 8),
            borderwidth=0
        )
        self.style.map(
            "Primary.TButton",
            background=[("active", self.c_primary_hover), ("pressed", "#4E1220")]
        )

        self.style.configure(
            "Secondary.TButton",
            background="#E2E8F0",
            foreground=self.c_text,
            font=("Segoe UI", 9, "bold"),
            padding=(10, 5),
            borderwidth=0
        )
        self.style.map(
            "Secondary.TButton",
            background=[("active", "#CBD5E1")]
        )

        # Progressbar
        self.style.configure(
            "Confidence.Horizontal.TProgressbar",
            troughcolor="#E2E8F0",
            background=self.c_primary,
            thickness=14
        )

    def build_ui(self):
        # 1. Top Header Banner
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(25, 18))
        header.pack(fill="x", side="top")

        title_box = ttk.Frame(header, style="Header.TFrame")
        title_box.pack(side="left")
        ttk.Label(title_box, text="🍷 Wine Quality AI Predictor", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(title_box, text="Machine Learning Physicochemical Quality Assessment System", style="HeaderSub.TLabel").pack(anchor="w", pady=(2, 0))

        badge_box = ttk.Frame(header, style="Header.TFrame")
        badge_box.pack(side="right")
        ttk.Label(
            badge_box, 
            text="Model: HistGradientBoosting (88.3% Accuracy)", 
            background="#4E1220", 
            foreground="#FFFFFF", 
            font=("Segoe UI", 9, "bold"), 
            padding=(10, 6)
        ).pack()

        # 2. Main Content Split (Left: Input Form, Right: Result Dashboard)
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill="both", expand=True)

        # Left Column: Inputs Card
        left_card = ttk.Frame(main_container, style="Card.TFrame", padding=20)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 12))

        # Preset Buttons Row
        preset_frame = ttk.Frame(left_card, style="Card.TFrame")
        preset_frame.pack(fill="x", pady=(0, 15))
        ttk.Label(preset_frame, text="Quick Presets:", style="FieldLabel.TLabel").pack(side="left", padx=(0, 10))
        ttk.Button(preset_frame, text="⭐ Premium Wine", style="Secondary.TButton", command=self.load_preset_premium).pack(side="left", padx=4)
        ttk.Button(preset_frame, text="🍷 Table Red", style="Secondary.TButton", command=self.load_preset_table_red).pack(side="left", padx=4)
        ttk.Button(preset_frame, text="🥂 Table White", style="Secondary.TButton", command=self.load_preset_table_white).pack(side="left", padx=4)
        ttk.Button(preset_frame, text="⚠️ Low Quality", style="Secondary.TButton", command=self.load_preset_poor).pack(side="left", padx=4)

        # Wine Type Toggle
        type_frame = ttk.Frame(left_card, style="Card.TFrame")
        type_frame.pack(fill="x", pady=(0, 15))
        ttk.Label(type_frame, text="Wine Variety:", style="FieldLabel.TLabel").pack(side="left", padx=(0, 15))
        ttk.Radiobutton(type_frame, text="White Wine", variable=self.wine_type_var, value=0, command=self.on_wine_type_change).pack(side="left", padx=8)
        ttk.Radiobutton(type_frame, text="Red Wine", variable=self.wine_type_var, value=1, command=self.on_wine_type_change).pack(side="left", padx=8)

        # 11 Physicochemical Attributes Grid
        grid_frame = ttk.Frame(left_card, style="Card.TFrame")
        grid_frame.pack(fill="both", expand=True)

        features_info = [
            ("alcohol", "Alcohol", "% vol", "8.0 - 14.9"),
            ("volatile acidity", "Volatile Acidity", "g/dm³ (acetic)", "0.08 - 1.58"),
            ("citric acid", "Citric Acid", "g/dm³", "0.00 - 1.66"),
            ("residual sugar", "Residual Sugar", "g/dm³", "0.6 - 65.8"),
            ("fixed acidity", "Fixed Acidity", "g/dm³ (tartaric)", "3.8 - 15.9"),
            ("chlorides", "Chlorides", "g/dm³ (sodium)", "0.009 - 0.61"),
            ("free sulfur dioxide", "Free SO₂", "mg/dm³", "1.0 - 289.0"),
            ("total sulfur dioxide", "Total SO₂", "mg/dm³", "6.0 - 440.0"),
            ("density", "Density", "g/cm³", "0.987 - 1.038"),
            ("pH", "pH Level", "scale", "2.72 - 4.01"),
            ("sulphates", "Sulphates", "g/dm³ (potassium)", "0.22 - 2.00")
        ]

        for i, (key, label, unit, rng) in enumerate(features_info):
            row = i // 2
            col = (i % 2) * 3

            cell = ttk.Frame(grid_frame, style="Card.TFrame", padding=(4, 4))
            cell.grid(row=row, column=col, columnspan=2, sticky="nsew", padx=6, pady=4)
            grid_frame.columnconfigure(col, weight=1)

            lbl_row = ttk.Frame(cell, style="Card.TFrame")
            lbl_row.pack(fill="x")
            ttk.Label(lbl_row, text=label, style="FieldLabel.TLabel").pack(side="left")
            ttk.Label(lbl_row, text=f"({unit})", style="FieldUnit.TLabel").pack(side="left", padx=(4, 0))

            entry = ttk.Entry(cell, font=("Segoe UI", 10))
            entry.pack(fill="x", pady=(2, 0))
            self.entries[key] = entry

        # Predict Button
        btn_frame = ttk.Frame(left_card, style="Card.TFrame", padding=(0, 15))
        btn_frame.pack(fill="x", side="bottom")
        predict_btn = ttk.Button(btn_frame, text="🔍 Evaluate & Predict Quality", style="Primary.TButton", command=self.make_prediction)
        predict_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        clear_btn = ttk.Button(btn_frame, text="Clear", style="Secondary.TButton", command=self.clear_fields)
        clear_btn.pack(side="right")

        # Right Column: Result Card
        right_card = ttk.Frame(main_container, style="Card.TFrame", padding=25)
        right_card.pack(side="right", fill="both", expand=True, padx=(12, 0))

        ttk.Label(right_card, text="Prediction Analysis", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 15))

        self.verdict_badge = tk.Label(
            right_card,
            text="READY FOR EVALUATION",
            font=("Segoe UI", 14, "bold"),
            bg="#E2E8F0",
            fg="#475569",
            padx=16,
            pady=12,
            relief="flat"
        )
        self.verdict_badge.pack(fill="x", pady=(0, 20))

        # Confidence Section
        conf_box = ttk.Frame(right_card, style="Card.TFrame")
        conf_box.pack(fill="x", pady=(0, 20))
        
        conf_label_row = ttk.Frame(conf_box, style="Card.TFrame")
        conf_label_row.pack(fill="x", pady=(0, 6))
        ttk.Label(conf_label_row, text="High Quality Confidence Score:", style="FieldLabel.TLabel").pack(side="left")
        self.conf_percent_lbl = ttk.Label(conf_label_row, text="--%", font=("Segoe UI", 11, "bold"), style="FieldLabel.TLabel")
        self.conf_percent_lbl.pack(side="right")

        self.progress_bar = ttk.Progressbar(conf_box, style="Confidence.Horizontal.TProgressbar", mode="determinate")
        self.progress_bar.pack(fill="x")

        # Physicochemical Drivers Box
        ttk.Label(right_card, text="Key Laboratory Drivers:", style="FieldLabel.TLabel").pack(anchor="w", pady=(10, 6))
        self.driver_box = tk.Text(
            right_card,
            height=9,
            font=("Segoe UI", 9),
            bg="#F8FAFC",
            fg=self.c_text,
            relief="flat",
            padx=10,
            pady=10
        )
        self.driver_box.pack(fill="both", expand=True)
        self.driver_box.insert("1.0", "Select a preset or enter chemical metrics, then click 'Evaluate & Predict Quality'.")
        self.driver_box.configure(state="disabled")

        # Model Info Footer in Card
        footer_info = ttk.Frame(right_card, style="Card.TFrame", padding=(0, 10))
        footer_info.pack(fill="x", side="bottom")
        ttk.Label(
            footer_info, 
            text="Target: High Quality (Rating ≥ 7) | Standard: UCI Physicochemical Dataset", 
            style="Muted.TLabel"
        ).pack(anchor="w")

    def on_wine_type_change(self):
        pass

    def fill_entries(self, data):
        for k, v in data.items():
            if k in self.entries:
                self.entries[k].delete(0, tk.END)
                self.entries[k].insert(0, str(v))
        if "wine_type" in data:
            self.wine_type_var.set(data["wine_type"])

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.verdict_badge.configure(text="READY FOR EVALUATION", bg="#E2E8F0", fg="#475569")
        self.conf_percent_lbl.configure(text="--%")
        self.progress_bar["value"] = 0
        self.update_driver_text("Fields cleared.")

    def load_preset_premium(self):
        self.fill_entries({
            "wine_type": 0,
            "alcohol": 12.8,
            "volatile acidity": 0.18,
            "citric acid": 0.36,
            "residual sugar": 1.4,
            "fixed acidity": 6.8,
            "chlorides": 0.034,
            "free sulfur dioxide": 38.0,
            "total sulfur dioxide": 105.0,
            "density": 0.9898,
            "pH": 3.25,
            "sulphates": 0.52
        })
        self.make_prediction()

    def load_preset_table_red(self):
        self.fill_entries({
            "wine_type": 1,
            "alcohol": 10.5,
            "volatile acidity": 0.42,
            "citric acid": 0.24,
            "residual sugar": 2.2,
            "fixed acidity": 7.9,
            "chlorides": 0.076,
            "free sulfur dioxide": 15.0,
            "total sulfur dioxide": 55.0,
            "density": 0.9964,
            "pH": 3.36,
            "sulphates": 0.60
        })
        self.make_prediction()

    def load_preset_table_white(self):
        self.fill_entries({
            "wine_type": 0,
            "alcohol": 10.2,
            "volatile acidity": 0.28,
            "citric acid": 0.30,
            "residual sugar": 6.4,
            "fixed acidity": 6.7,
            "chlorides": 0.048,
            "free sulfur dioxide": 32.0,
            "total sulfur dioxide": 135.0,
            "density": 0.9946,
            "pH": 3.18,
            "sulphates": 0.46
        })
        self.make_prediction()

    def load_preset_poor(self):
        self.fill_entries({
            "wine_type": 1,
            "alcohol": 9.2,
            "volatile acidity": 0.92,
            "citric acid": 0.02,
            "residual sugar": 2.8,
            "fixed acidity": 7.5,
            "chlorides": 0.110,
            "free sulfur dioxide": 8.0,
            "total sulfur dioxide": 35.0,
            "density": 0.9982,
            "pH": 3.55,
            "sulphates": 0.42
        })
        self.make_prediction()

    def update_driver_text(self, text):
        self.driver_box.configure(state="normal")
        self.driver_box.delete("1.0", tk.END)
        self.driver_box.insert("1.0", text)
        self.driver_box.configure(state="disabled")

    def make_prediction(self):
        sample = {"wine_type": self.wine_type_var.get()}
        try:
            for k, entry in self.entries.items():
                val_str = entry.get().strip()
                if not val_str:
                    messagebox.showwarning("Incomplete Input", f"Please provide a numeric value for '{k}'.")
                    return
                sample[k] = float(val_str)
        except ValueError as err:
            messagebox.showerror("Invalid Input", f"Please verify all entries are numbers.\nError: {err}")
            return

        result = self.predictor.predict_sample(sample)
        pred = result["prediction"]
        proba = result["high_quality_probability"]
        pct = proba * 100

        # Update Verdict Badge
        if pred == 1:
            self.verdict_badge.configure(
                text="⭐ HIGH QUALITY VINTAGE (Score ≥ 7)",
                bg=self.c_success,
                fg="#FFFFFF"
            )
            self.style.configure("Confidence.Horizontal.TProgressbar", background=self.c_success)
        else:
            self.verdict_badge.configure(
                text="🍷 STANDARD / TABLE QUALITY (Score < 7)",
                bg="#475569",
                fg="#FFFFFF"
            )
            self.style.configure("Confidence.Horizontal.TProgressbar", background="#D97706")

        self.conf_percent_lbl.configure(text=f"{pct:.1f}%")
        self.progress_bar["value"] = pct

        # Dynamic Explanatory Insights
        alcohol = sample.get("alcohol", 10.0)
        va = sample.get("volatile acidity", 0.3)
        so2 = sample.get("free sulfur dioxide", 20.0)
        wine_str = "Red Wine" if sample["wine_type"] == 1 else "White Wine"

        insights = [
            f"• Wine Classification: {wine_str}",
            f"• High Quality Probability: {pct:.1f}%",
            "",
            "Physicochemical Analysis:"
        ]

        if alcohol >= 11.5:
            insights.append(f"  ✔ Alcohol Content ({alcohol:.1f}%): Favorable high alcohol profile.")
        else:
            insights.append(f"  ▲ Alcohol Content ({alcohol:.1f}%): Lower alcohol, typical of table wines.")

        if va <= 0.35:
            insights.append(f"  ✔ Volatile Acidity ({va:.2f} g/dm³): Excellent low acetic acid, no vinegar flaw.")
        else:
            insights.append(f"  ▲ Volatile Acidity ({va:.2f} g/dm³): Elevated acetic acid may cause off-flavors.")

        if sample["wine_type"] == 0:
            if so2 >= 30:
                insights.append(f"  ✔ Free SO₂ ({so2:.0f} mg/dm³): Proper antioxidant preservation for white wine.")
            else:
                insights.append(f"  ▲ Free SO₂ ({so2:.0f} mg/dm³): Low antioxidant buffer, risk of oxidation.")
        else:
            if sample.get("sulphates", 0.5) >= 0.6:
                insights.append(f"  ✔ Sulphates ({sample.get('sulphates', 0.5):.2f} g/dm³): Strong microbial protection.")

        self.update_driver_text("\n".join(insights))


def main():
    root = tk.Tk()
    app = WineQualityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

