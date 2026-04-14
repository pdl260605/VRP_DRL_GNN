"""
VRP Model Comparison GUI
========================
Giao diện so sánh nhiều model VRP cùng một lúc.
Chạy: python run_gui.py  (từ thư mục PyTorch_GNN)
"""

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

# -- Matplotlib backend cho Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# -- PyTorch
import torch

# -- Project imports
from data import data_from_txt, generate_data
from model import AttentionModel
from baseline import load_model
from model_light import GNNLightModel

# ==============================================================================
# CONSTANTS
# ==============================================================================
WEIGHTS_DIR  = os.path.join(os.path.dirname(__file__), "Weights")
OPENDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "OpenData")
MAX_MODELS   = 6

COLOR_BG      = "#0f1117"
COLOR_PANEL   = "#1a1d27"
COLOR_CARD    = "#21253a"
COLOR_BORDER  = "#2e3350"
COLOR_ACCENT  = "#4f8ef7"
COLOR_ACCENT2 = "#7c5af0"
COLOR_SUCCESS = "#3ec97e"
COLOR_WARNING = "#f5a623"
COLOR_TEXT    = "#e8ecf5"
COLOR_MUTED   = "#6b7394"

ROUTE_COLORS = [
    "#4f8ef7", "#3ec97e", "#f5a623", "#e05c5c",
    "#c87af5", "#f76f8e", "#5af0c8", "#f7c94f",
    "#8ef76f", "#f78e4f",
]

FONT_HEADER = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)


# ==============================================================================
# SCROLLABLE FRAME
# ==============================================================================

class ScrollableFrame(tk.Frame):
    """Canvas-backed frame with optional vertical/horizontal scrollbars.
    
    Use .inner to place child widgets.
    orient: 'vertical' | 'horizontal' | 'both'
    """

    def __init__(self, parent, orient="vertical", bg=COLOR_BG, **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self._orient = orient

        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.inner   = tk.Frame(self._canvas, bg=bg)
        self._win_id = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")

        if orient in ("vertical", "both"):
            self._vsb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
            self._canvas.configure(yscrollcommand=self._vsb.set)
            self._vsb.pack(side="right", fill="y")

        if orient in ("horizontal", "both"):
            self._hsb = ttk.Scrollbar(self, orient="horizontal", command=self._canvas.xview)
            self._canvas.configure(xscrollcommand=self._hsb.set)
            self._hsb.pack(side="bottom", fill="x")

        self._canvas.pack(side="left", fill="both", expand=True)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel_on(self._canvas)
        self._bind_mousewheel_on(self.inner)

    def _on_inner_configure(self, event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        if self._orient == "vertical":
            self._canvas.itemconfig(self._win_id, width=event.width)

    def _scroll_y(self, event):
        if self._orient not in ("vertical", "both"):
            return
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _scroll_x(self, event):
        if self._orient not in ("horizontal", "both"):
            return
        if event.num == 4:
            self._canvas.xview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.xview_scroll(1, "units")
        else:
            self._canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_mousewheel_on(self, widget):
        widget.bind("<MouseWheel>",       self._scroll_y)
        widget.bind("<Button-4>",         self._scroll_y)
        widget.bind("<Button-5>",         self._scroll_y)
        widget.bind("<Shift-MouseWheel>", self._scroll_x)

    def bind_mousewheel_to(self, widget):
        """Propagate mousewheel events from a child widget to this scrollable frame."""
        self._bind_mousewheel_on(widget)


# ==============================================================================
# MODEL HELPERS
# ==============================================================================

def is_light_model(name):
    return "GNN_Light" in name

def _detect_light_params(state_dict):
    """Tự động đọc embed_dim và n_encode_layers từ shape của checkpoint.
    
    Tránh hard-code 64/2 cho mọi Light model vì VRP50 Light dùng 128/3.
    """
    embed_dim = state_dict["Encoder.init_W_depot.weight"].shape[0]
    n_layers  = len([k for k in state_dict.keys()
                     if "gnn_layers" in k and "W_msg.weight" in k])
    return embed_dim, n_layers

def load_vrp_model(weight_path, device):
    """Load model, tự động detect kiến trúc từ tên file và shape weights."""
    name = os.path.basename(weight_path)
    try:
        n_cust = int(name.split("VRP")[1].split("_")[0])
    except Exception:
        n_cust = 20

    state = torch.load(weight_path, map_location=device)

    if is_light_model(name):
        embed_dim, n_layers = _detect_light_params(state)
        model = GNNLightModel(embed_dim=embed_dim,
                              n_encode_layers=n_layers,
                              tanh_clipping=10.0)
        model.load_state_dict(state)
    else:
        model = load_model(weight_path, embed_dim=128,
                           n_customer=n_cust, n_encode_layers=3)

    model.to(device)
    model.eval()
    return model, n_cust

def run_model(model, data, decode_type="sampling"):
    with torch.no_grad():
        costs, _, pi = model(data, return_pi=True, decode_type=decode_type)
    return costs, pi

def prepare_data(txt_path, batch, device):
    raw = data_from_txt(txt_path)
    data = []
    for i in range(3):
        elem = [raw[i].squeeze(0) for _ in range(batch)]
        data.append(torch.stack(elem, 0).to(device))
    return data

def get_routes_from_pi(pi_row):
    routes, cur = [], []
    for node in pi_row:
        if node == 0:
            if cur:
                routes.append(cur)
                cur = []
        else:
            cur.append(int(node) - 1)
    if cur:
        routes.append(cur)
    return routes

def calc_tour_length(depot, customers, route):
    """Tinh do dai mot tour (depot -> nodes -> depot)."""
    coords = np.array([depot] + [customers[n] for n in route] + [depot])
    return float(np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))))

def plot_vrp_on_ax(ax, depot, customers, routes, title, cost):
    """Ve VRP solution. Hien thi tour length tung route trong legend."""
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors=COLOR_MUTED, labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor(COLOR_BORDER)

    tour_lengths = []
    for idx, route in enumerate(routes):
        color  = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
        coords = np.array([depot] + [customers[n] for n in route] + [depot])
        tlen   = calc_tour_length(depot, customers, route)
        tour_lengths.append(tlen)

        ax.plot(coords[:, 0], coords[:, 1], "-o",
                color=color, linewidth=1.5, markersize=3.5, alpha=0.85,
                label=f"R{idx+1}: {tlen:.2f}")
        for k in range(len(coords) - 1):
            ax.annotate("", xy=coords[k+1], xytext=coords[k],
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=0.8, mutation_scale=7))

    ax.scatter(customers[:, 0], customers[:, 1],
               c=COLOR_TEXT, s=20, zorder=5, linewidths=0)
    ax.scatter([depot[0]], [depot[1]],
               marker="*", c=COLOR_WARNING, s=180, zorder=6, linewidths=0)

    # Legend: hien tour length tung route
    if tour_lengths:
        ax.legend(fontsize=5.5, loc="lower right",
                  framealpha=0.6, facecolor=COLOR_CARD,
                  edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT,
                  handlelength=1.2, handletextpad=0.5,
                  borderpad=0.5, labelspacing=0.3)

    # Title: ten model + total cost + range [min - max]
    short = os.path.splitext(title)[0]
    if tour_lengths:
        t_min, t_max = min(tour_lengths), max(tour_lengths)
        range_str = f"Range: [{t_min:.2f} - {t_max:.2f}]"
    else:
        range_str = ""

    ax.set_title(f"{short}\nTotal: {cost:.3f}  |  {range_str}",
                 color=COLOR_TEXT, fontsize=7.5, fontweight="bold", pad=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

class VRPCompareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VRP-DRL-GNN · Model Comparison Dashboard")
        self.geometry("1440x880")
        self.minsize(900, 600)
        self.configure(bg=COLOR_BG)
        self.resizable(True, True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.selected_weights = []
        self.weight_files     = []
        self.txt_var    = tk.StringVar(value="")
        self.batch_var  = tk.IntVar(value=128)
        self.decode_var = tk.StringVar(value="sampling")
        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.layout_var = tk.StringVar(value="grid")
        self.results    = []

        self._apply_styles()
        self._build_ui()
        self._load_weight_list()
        self._load_data_list()

    # --------------------------------------------------------------------------
    # Styles
    # --------------------------------------------------------------------------

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Dark.TNotebook", background=COLOR_BG, borderwidth=0, tabmargins=[0,0,0,0])
        s.configure("Dark.TNotebook.Tab",
                    background=COLOR_CARD, foreground=COLOR_TEXT,
                    padding=[16, 7], font=FONT_BODY, borderwidth=0)
        s.map("Dark.TNotebook.Tab",
              background=[("selected", COLOR_ACCENT)],
              foreground=[("selected", "white")])
        s.configure("TCombobox",
                    fieldbackground=COLOR_CARD, background=COLOR_CARD,
                    foreground=COLOR_TEXT, arrowcolor=COLOR_TEXT,
                    bordercolor=COLOR_BORDER,
                    selectbackground=COLOR_ACCENT, selectforeground="white")
        s.configure("TScrollbar",
                    background=COLOR_BORDER, troughcolor=COLOR_CARD,
                    arrowcolor=COLOR_MUTED, bordercolor=COLOR_CARD)
        s.map("TScrollbar", background=[("active", COLOR_ACCENT)])
        s.configure("TProgressbar",
                    troughcolor=COLOR_CARD, background=COLOR_ACCENT,
                    bordercolor=COLOR_BORDER)

    # --------------------------------------------------------------------------
    # UI Build
    # --------------------------------------------------------------------------

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=COLOR_PANEL, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="VRP-DRL-GNN",
                 font=("Segoe UI", 17, "bold"),
                 bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(side="left", padx=18, pady=10)
        tk.Label(hdr, text="Model Comparison Dashboard",
                 font=("Segoe UI", 11), bg=COLOR_PANEL, fg=COLOR_MUTED).pack(side="left")

        dev_txt   = "  GPU \u2713  " if torch.cuda.is_available() else "  CPU  "
        dev_color = COLOR_SUCCESS if torch.cuda.is_available() else COLOR_WARNING
        tk.Label(hdr, text=dev_txt, font=("Segoe UI", 9, "bold"),
                 bg=dev_color, fg=COLOR_BG).pack(side="right", padx=18, pady=16)

        # Body
        body = tk.Frame(self, bg=COLOR_BG)
        body.pack(fill="both", expand=True)

        # Sidebar (scrollable)
        sidebar_outer = tk.Frame(body, bg=COLOR_PANEL, width=296)
        sidebar_outer.pack(side="left", fill="y", padx=(6, 0), pady=6)
        sidebar_outer.pack_propagate(False)

        self._sf_sidebar = ScrollableFrame(sidebar_outer, orient="vertical", bg=COLOR_PANEL)
        self._sf_sidebar.pack(fill="both", expand=True)
        self._sidebar = self._sf_sidebar.inner

        # Right
        self.right = tk.Frame(body, bg=COLOR_BG)
        self.right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self._build_sidebar()
        self._build_right_area()

        # Status bar
        sb = tk.Frame(self, bg=COLOR_BORDER, height=26)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        tk.Label(sb, textvariable=self.status_var,
                 font=FONT_SMALL, bg=COLOR_BORDER, fg=COLOR_MUTED).pack(
                     side="left", padx=12, pady=4)

    # --------------------------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------------------------

    def _build_sidebar(self):
        p = self._sidebar

        # --- Models ---
        self._section(p, "\U0001f4e6  Models  (t\u1ed1i \u0111a 6)")

        list_host = tk.Frame(p, bg=COLOR_CARD,
                             highlightbackground=COLOR_BORDER, highlightthickness=1)
        list_host.pack(fill="x", padx=8, pady=(0, 4))

        self._sf_models = ScrollableFrame(list_host, orient="vertical", bg=COLOR_CARD)
        self._sf_models.configure(height=200)
        self._sf_models.pack(fill="x")
        self.model_inner = self._sf_models.inner

        btn_row = tk.Frame(p, bg=COLOR_PANEL)
        btn_row.pack(fill="x", padx=8, pady=(0, 8))
        self._small_btn(btn_row, "\u2714 Ch\u1ecdn t\u1ea5t c\u1ea3", self._select_all
                        ).pack(side="left", expand=True, fill="x", padx=(0,2))
        self._small_btn(btn_row, "\u2716 B\u1ecf ch\u1ecdn", self._clear_all
                        ).pack(side="left", expand=True, fill="x", padx=(2,0))

        # --- Data ---
        self._section(p, "\U0001f4c2  D\u1eef li\u1ec7u")
        self.data_combo = ttk.Combobox(p, textvariable=self.txt_var,
                                       state="readonly", font=FONT_BODY)
        self.data_combo.pack(fill="x", padx=8, pady=(0, 4))
        self._small_btn(p, "\U0001f4c1  Duy\u1ec7t file...", self._browse_file
                        ).pack(fill="x", padx=8, pady=(0, 8))

        # --- Params ---
        self._section(p, "\u2699\ufe0f  Tham s\u1ed1")
        pf = tk.Frame(p, bg=COLOR_PANEL)
        pf.pack(fill="x", padx=8, pady=(0, 8))
        pf.columnconfigure(1, weight=1)

        tk.Label(pf, text="Batch size:", font=FONT_BODY,
                 bg=COLOR_PANEL, fg=COLOR_TEXT).grid(row=0, column=0, sticky="w", pady=3)
        tk.Spinbox(pf, from_=1, to=512, textvariable=self.batch_var,
                   width=7, font=FONT_BODY, bg=COLOR_CARD, fg=COLOR_TEXT,
                   buttonbackground=COLOR_BORDER, relief="flat",
                   insertbackground=COLOR_TEXT
                   ).grid(row=0, column=1, sticky="e", pady=3)

        tk.Label(pf, text="Decode:", font=FONT_BODY,
                 bg=COLOR_PANEL, fg=COLOR_TEXT).grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(pf, textvariable=self.decode_var,
                     values=["sampling", "greedy"],
                     state="readonly", width=10, font=FONT_BODY
                     ).grid(row=1, column=1, sticky="e", pady=3)

        # --- Layout toggle ---
        self._section(p, "\U0001f5bc\ufe0f  B\u1ed1 c\u1ee5c bi\u1ec3u \u0111\u1ed3")
        lf = tk.Frame(p, bg=COLOR_PANEL)
        lf.pack(fill="x", padx=8, pady=(0, 8))

        for val, lbl in [("grid", "\u229e  L\u01b0\u1edbi 2\u00d73"), ("row", "\u2194  H\u00e0ng ngang")]:
            tk.Radiobutton(lf, text=lbl, variable=self.layout_var, value=val,
                           font=FONT_BODY, bg=COLOR_PANEL, fg=COLOR_TEXT,
                           selectcolor=COLOR_CARD, activebackground=COLOR_PANEL,
                           activeforeground=COLOR_ACCENT,
                           indicatoron=0, relief="flat", cursor="hand2",
                           padx=10, pady=6,
                           command=self._on_layout_change
                           ).pack(side="left", expand=True, fill="x", padx=2)

        # --- Run ---
        self.run_btn = tk.Button(p, text="\u25b6  Ch\u1ea1y So S\u00e1nh",
                                 font=("Segoe UI", 11, "bold"),
                                 bg=COLOR_ACCENT, fg="white",
                                 activebackground=COLOR_ACCENT2,
                                 activeforeground="white",
                                 relief="flat", cursor="hand2",
                                 command=self._run_comparison)
        self.run_btn.pack(fill="x", padx=8, pady=4, ipady=10)

        self.progress = ttk.Progressbar(p, mode="indeterminate")
        self.progress.pack(fill="x", padx=8, pady=(0, 6))

        # --- Log ---
        self._section(p, "\U0001f4cb  Log")
        log_host = tk.Frame(p, bg=COLOR_CARD)
        log_host.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.log_text = tk.Text(log_host, height=9, bg=COLOR_CARD, fg=COLOR_TEXT,
                                font=FONT_MONO, relief="flat", state="disabled",
                                wrap="word", insertbackground=COLOR_TEXT)
        log_sb = ttk.Scrollbar(log_host, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

    # --------------------------------------------------------------------------
    # Right area
    # --------------------------------------------------------------------------

    def _build_right_area(self):
        self.chart_frame = tk.Frame(self.right, bg=COLOR_BG)
        self.chart_frame.pack(fill="both", expand=True)
        self._show_placeholder()

    def _show_placeholder(self):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        tk.Label(self.chart_frame,
                 text="Ch\u1ecdn models v\u00e0 d\u1eef li\u1ec7u,\ns\u00e1u \u0111\u00f3 nh\u1ea5n  \u25b6 Ch\u1ea1y So S\u00e1nh",
                 font=("Segoe UI", 15), bg=COLOR_BG, fg=COLOR_MUTED,
                 justify="center").place(relx=0.5, rely=0.5, anchor="center")

    # --------------------------------------------------------------------------
    # Widget helpers
    # --------------------------------------------------------------------------

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=COLOR_PANEL)
        f.pack(fill="x", padx=8, pady=(12, 4))
        tk.Label(f, text=title, font=FONT_HEADER,
                 bg=COLOR_PANEL, fg=COLOR_TEXT).pack(side="left")
        tk.Frame(f, bg=COLOR_BORDER, height=1).pack(side="bottom", fill="x", pady=(4,0))

    def _small_btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, font=FONT_SMALL,
                         bg=COLOR_CARD, fg=COLOR_TEXT,
                         activebackground=COLOR_BORDER, activeforeground=COLOR_TEXT,
                         relief="flat", cursor="hand2", command=cmd, pady=5)

    # --------------------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------------------

    def _load_weight_list(self):
        for w in self.model_inner.winfo_children():
            w.destroy()
        self.selected_weights.clear()
        self.weight_files.clear()

        if not os.path.isdir(WEIGHTS_DIR):
            self._log(f"Warning: {WEIGHTS_DIR} not found")
            return

        files = sorted(f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pt"))
        if not files:
            self._log("No .pt files in Weights/")
            return

        for fname in files:
            var = tk.BooleanVar(value=False)
            self.selected_weights.append(var)
            self.weight_files.append(fname)

            is_light  = is_light_model(fname)
            tag       = " Light" if is_light else "  GNN"
            tag_color = COLOR_ACCENT2 if is_light else COLOR_ACCENT
            tag_bg    = "#2a1f45" if is_light else "#1a2a45"

            row = tk.Frame(self.model_inner, bg=COLOR_CARD, cursor="hand2")
            row.pack(fill="x", padx=2, pady=1)

            cb = tk.Checkbutton(row, variable=var, bg=COLOR_CARD,
                                activebackground=COLOR_CARD,
                                selectcolor=COLOR_BG, fg=COLOR_TEXT,
                                command=self._on_check)
            cb.pack(side="left", padx=(4, 0))

            def _toggle(e, v=var):
                v.set(not v.get())
                self._on_check()

            name_lbl = tk.Label(row, text=fname, font=FONT_SMALL,
                                bg=COLOR_CARD, fg=COLOR_TEXT, anchor="w")
            name_lbl.pack(side="left", fill="x", expand=True, padx=2)
            name_lbl.bind("<Button-1>", _toggle)

            tag_lbl = tk.Label(row, text=tag, font=("Segoe UI", 8, "bold"),
                               bg=tag_bg, fg=tag_color, padx=5, pady=1)
            tag_lbl.pack(side="right", padx=4)
            tag_lbl.bind("<Button-1>", _toggle)

            for w in (row, cb, name_lbl, tag_lbl):
                self._sf_models.bind_mousewheel_to(w)
                self._sf_sidebar.bind_mousewheel_to(w)

        self._log(f"Loaded {len(files)} weights from Weights/")

    def _load_data_list(self):
        opts = []
        if os.path.isdir(OPENDATA_DIR):
            for f in sorted(os.listdir(OPENDATA_DIR)):
                if f.endswith(".txt") and not f.startswith("opt-") and f != "explain.txt":
                    opts.append(os.path.join(OPENDATA_DIR, f))
        self.data_combo["values"] = opts
        if opts:
            self.txt_var.set(opts[0])
            self._log(f"Found {len(opts)} data files in OpenData/")

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select VRP data file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=OPENDATA_DIR)
        if path:
            vals = list(self.data_combo["values"])
            if path not in vals:
                vals.append(path)
                self.data_combo["values"] = vals
            self.txt_var.set(path)

    # --------------------------------------------------------------------------
    # Checkbox
    # --------------------------------------------------------------------------

    def _on_check(self):
        n = sum(v.get() for v in self.selected_weights)
        if n > MAX_MODELS:
            messagebox.showwarning("Gioi han", f"Chi chon toi da {MAX_MODELS} model!")
            for v in reversed(self.selected_weights):
                if v.get():
                    v.set(False)
                    break

    def _select_all(self):
        for i, v in enumerate(self.selected_weights):
            v.set(i < MAX_MODELS)
        if len(self.selected_weights) > MAX_MODELS:
            messagebox.showinfo("Thong bao", f"Chi chon {MAX_MODELS} model dau tien.")

    def _clear_all(self):
        for v in self.selected_weights:
            v.set(False)

    # --------------------------------------------------------------------------
    # Log
    # --------------------------------------------------------------------------

    def _log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.status_var.set(msg[:100])
        self.update_idletasks()

    # --------------------------------------------------------------------------
    # Layout change
    # --------------------------------------------------------------------------

    def _on_layout_change(self):
        if self.results:
            self.after(0, self._render_results)

    # --------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------

    def _run_comparison(self):
        selected = [self.weight_files[i]
                    for i in range(len(self.weight_files))
                    if self.selected_weights[i].get()]
        if not selected:
            messagebox.showwarning("Chua chon model", "Vui long chon it nhat 1 model.")
            return
        if len(selected) > MAX_MODELS:
            messagebox.showwarning("Qua nhieu model", f"Toi da {MAX_MODELS} model.")
            return
        if not self.txt_var.get():
            messagebox.showwarning("Chua chon du lieu", "Vui long chon file du lieu.")
            return

        self.run_btn.configure(state="disabled")
        self.progress.start(12)
        self.results.clear()

        threading.Thread(
            target=self._run_thread,
            args=(selected, self.txt_var.get(),
                  self.batch_var.get(), self.decode_var.get()),
            daemon=True).start()

    def _run_thread(self, model_names, txt_path, batch, decode):
        try:
            self._log(f"\n{'--'*20}")
            self._log(f"Data: {os.path.basename(txt_path)}")
            self._log(f"Batch={batch}, Decode={decode}")
            self._log(f"{'--'*20}")

            results = []
            for i, wname in enumerate(model_names):
                wp = os.path.join(WEIGHTS_DIR, wname)
                self._log(f"[{i+1}/{len(model_names)}] Loading: {wname}")
                t0 = time.time()

                try:
                    model, n_cust = load_vrp_model(wp, self.device)
                except Exception as e:
                    self._log(f"  ERROR loading model: {e}")
                    import traceback
                    self._log(traceback.format_exc())
                    continue
                # Log kich thuoc model (embed_dim, n_layers) neu la Light
                import torch as _t
                _state = _t.load(wp, map_location=self.device)
                if is_light_model(wname):
                    _ed, _nl = _detect_light_params(_state)
                    self._log(f"  OK ({time.time()-t0:.1f}s) | n_cust={n_cust} | embed={_ed} | layers={_nl} | [Light]")
                else:
                    self._log(f"  OK ({time.time()-t0:.1f}s) | n_cust={n_cust} | embed=128 | layers=3 | [GNN]")

                try:
                    data = prepare_data(txt_path, batch, self.device)
                except Exception as e:
                    self._log(f"  ERROR data: {e}")
                    continue

                t1 = time.time()
                try:
                    costs, pi = run_model(model, data, decode)
                except Exception as e:
                    self._log(f"  ERROR inference: {e}")
                    continue

                idx_best  = torch.argmin(costs).item()
                best_cost = costs[idx_best].item()
                avg_cost  = costs.mean().item()
                self._log(f"  Best={best_cost:.3f} | Avg={avg_cost:.3f} ({time.time()-t1:.1f}s)")

                results.append({
                    "name":      wname,
                    "cost":      best_cost,
                    "avg_cost":  avg_cost,
                    "routes":    get_routes_from_pi(pi[idx_best].cpu().numpy()),
                    "depot":     data[0][idx_best].cpu().numpy(),
                    "customers": data[1][idx_best].cpu().numpy(),
                })

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.results = results
            self.after(0, self._render_results)

        except Exception as e:
            import traceback
            self._log(f"\nFATAL ERROR: {e}")
            self._log(traceback.format_exc())
            self.after(0, self._done_running)

    # --------------------------------------------------------------------------
    # Render
    # --------------------------------------------------------------------------

    def _done_running(self):
        self.progress.stop()
        self.run_btn.configure(state="normal")

    def _render_results(self):
        self._done_running()
        results = self.results
        if not results:
            self._log("No results to display.")
            return

        for w in self.chart_frame.winfo_children():
            w.destroy()

        self._log(f"\nDisplaying {len(results)} models")

        nb = ttk.Notebook(self.chart_frame, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True)

        tab_routes  = tk.Frame(nb, bg=COLOR_BG)
        tab_metrics = tk.Frame(nb, bg=COLOR_BG)
        nb.add(tab_routes,  text="\U0001f5fa  Ban do Routes")
        nb.add(tab_metrics, text="\U0001f4ca  So sanh Chi phi")

        self._build_route_tab(tab_routes,   results)
        self._build_metrics_tab(tab_metrics, results)

    # --------------------------------------------------------------------------
    # Tab: Routes
    # --------------------------------------------------------------------------

    def _build_route_tab(self, parent, results):
        layout = self.layout_var.get()

        # Toolbar row
        tb = tk.Frame(parent, bg=COLOR_PANEL, height=40)
        tb.pack(fill="x")
        tb.pack_propagate(False)

        tk.Label(tb, text="Bo cuc:", font=FONT_SMALL,
                 bg=COLOR_PANEL, fg=COLOR_MUTED).pack(side="left", padx=(12,4), pady=10)

        for val, lbl in [("grid", "\u229e Luoi 2x3"), ("row", "\u2194 Hang ngang")]:
            active = (layout == val)
            tk.Button(tb, text=lbl, font=FONT_SMALL,
                      bg=COLOR_ACCENT if active else COLOR_CARD,
                      fg="white" if active else COLOR_TEXT,
                      activebackground=COLOR_ACCENT2, activeforeground="white",
                      relief="flat", cursor="hand2",
                      command=lambda v=val: self._switch_layout(v)
                      ).pack(side="left", padx=2, pady=8, ipady=2, ipadx=8)

        hint = "<-- Shift+scroll ngang -->" if layout == "row" else "Scroll doc"
        tk.Label(tb, text=hint, font=FONT_SMALL,
                 bg=COLOR_PANEL, fg=COLOR_MUTED).pack(side="right", padx=12)

        # Content
        host = tk.Frame(parent, bg=COLOR_BG)
        host.pack(fill="both", expand=True)

        if layout == "grid":
            self._build_route_grid(host, results)
        else:
            self._build_route_row(host, results)

    def _switch_layout(self, val):
        self.layout_var.set(val)
        self._on_layout_change()

    def _build_route_grid(self, parent, results):
        """2-row x 3-col grid, vertical scroll."""
        n    = len(results)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        sf = ScrollableFrame(parent, orient="vertical", bg=COLOR_BG)
        sf.pack(fill="both", expand=True)

        CELL_W, CELL_H = 4.4, 4.1
        fig = Figure(figsize=(cols * CELL_W, rows * CELL_H), facecolor=COLOR_BG)
        fig.subplots_adjust(hspace=0.38, wspace=0.22,
                            left=0.04, right=0.97, top=0.96, bottom=0.04)

        for i, res in enumerate(results):
            ax = fig.add_subplot(rows, cols, i + 1)
            plot_vrp_on_ax(ax, res["depot"], res["customers"],
                           res["routes"], res["name"], res["cost"])

        canvas = FigureCanvasTkAgg(fig, master=sf.inner)
        canvas.draw()
        cw = canvas.get_tk_widget()
        cw.pack(fill="both", expand=True)
        cw.configure(bg=COLOR_BG)
        sf.bind_mousewheel_to(cw)

        nav_f = tk.Frame(sf.inner, bg=COLOR_PANEL)
        nav_f.pack(fill="x")
        nav = NavigationToolbar2Tk(canvas, nav_f)
        nav.update()
        nav.configure(bg=COLOR_PANEL)

    def _build_route_row(self, parent, results):
        """Horizontal row, one chart per model, horizontal scroll."""
        n    = len(results)
        CELL = 4.3  # inches per chart
        DPI  = 96

        # Tip at bottom
        tip = tk.Label(parent,
                       text="Tip: Shift + cuon chuot de scroll ngang",
                       font=FONT_SMALL, bg=COLOR_BG, fg=COLOR_MUTED)
        tip.pack(side="bottom", pady=2)

        # Nav toolbar at bottom
        nav_host = tk.Frame(parent, bg=COLOR_PANEL)
        nav_host.pack(side="bottom", fill="x")

        # Scrollable horizontal area
        sf = ScrollableFrame(parent, orient="horizontal", bg=COLOR_BG)
        sf.pack(fill="both", expand=True)

        fig = Figure(figsize=(n * CELL, CELL * 1.05), facecolor=COLOR_BG)
        fig.subplots_adjust(hspace=0.1, wspace=0.14,
                            left=0.02, right=0.99, top=0.92, bottom=0.07)

        for i, res in enumerate(results):
            ax = fig.add_subplot(1, n, i + 1)
            plot_vrp_on_ax(ax, res["depot"], res["customers"],
                           res["routes"], res["name"], res["cost"])

        fig_px_w = int(n * CELL * DPI)
        fig_px_h = int(CELL * 1.05 * DPI)

        canvas = FigureCanvasTkAgg(fig, master=sf.inner)
        canvas.draw()
        cw = canvas.get_tk_widget()
        cw.configure(width=fig_px_w, height=fig_px_h, bg=COLOR_BG)
        cw.pack(side="left")
        sf.bind_mousewheel_to(cw)

        nav = NavigationToolbar2Tk(canvas, nav_host)
        nav.update()
        nav.configure(bg=COLOR_PANEL)

    # --------------------------------------------------------------------------
    # Tab: Metrics
    # --------------------------------------------------------------------------

    def _build_metrics_tab(self, parent, results):
        sf    = ScrollableFrame(parent, orient="vertical", bg=COLOR_BG)
        sf.pack(fill="both", expand=True)
        inner = sf.inner

        names    = [os.path.splitext(r["name"])[0].replace("_", "\n") for r in results]
        costs    = [r["cost"]     for r in results]
        avg_c    = [r["avg_cost"] for r in results]
        n_routes = [len(r["routes"]) for r in results]
        best_idx = int(np.argmin(costs))

        # Title banner
        best_name = os.path.splitext(results[best_idx]["name"])[0]
        banner = tk.Label(inner,
                          text=f"Best model: {best_name}   (cost = {costs[best_idx]:.4f})",
                          font=("Segoe UI", 12, "bold"),
                          bg=COLOR_BG, fg=COLOR_SUCCESS)
        banner.pack(pady=(10, 4))
        sf.bind_mousewheel_to(banner)

        # Bar charts
        fig = Figure(figsize=(10, 4.6), facecolor=COLOR_BG)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.26, wspace=0.35)

        ax1 = fig.add_subplot(1, 3, 1)
        self._bar_ax(ax1, names, costs,    "Best Cost",      COLOR_ACCENT)
        ax2 = fig.add_subplot(1, 3, 2)
        self._bar_ax(ax2, names, avg_c,    "Avg Cost",       COLOR_ACCENT2)
        ax3 = fig.add_subplot(1, 3, 3)
        self._bar_ax(ax3, names, n_routes, "So tuyen duong", COLOR_SUCCESS)

        canvas = FigureCanvasTkAgg(fig, master=inner)
        canvas.draw()
        cw = canvas.get_tk_widget()
        cw.pack(fill="x", padx=8, pady=4)
        sf.bind_mousewheel_to(cw)

        # Divider
        sep = tk.Frame(inner, bg=COLOR_BORDER, height=1)
        sep.pack(fill="x", padx=16, pady=8)
        sf.bind_mousewheel_to(sep)

        # Table header
        th = tk.Label(inner, text="Bang tom tat ket qua",
                      font=FONT_HEADER, bg=COLOR_BG, fg=COLOR_TEXT)
        th.pack(anchor="w", padx=16)
        sf.bind_mousewheel_to(th)

        # Table (horizontal scroll for narrow windows)
        tbl_sf = ScrollableFrame(inner, orient="horizontal", bg=COLOR_BG)
        tbl_sf.pack(fill="x", padx=16, pady=(4, 16))
        tbl = tbl_sf.inner

        headers    = ["#", "Model",  "Type",    "Best Cost", "Avg Cost", "# Routes"]
        col_widths = [3,   38,       10,        12,          12,         10]

        for j, (h, cw_) in enumerate(zip(headers, col_widths)):
            lbl = tk.Label(tbl, text=h, width=cw_, font=("Segoe UI", 9, "bold"),
                           bg=COLOR_BORDER, fg=COLOR_ACCENT, anchor="w",
                           relief="flat", padx=6)
            lbl.grid(row=0, column=j, padx=1, pady=1, sticky="ew")
            sf.bind_mousewheel_to(lbl)

        for i, res in enumerate(results):
            is_best = (i == best_idx)
            bg = COLOR_SUCCESS if is_best else (COLOR_CARD if i % 2 == 0 else COLOR_PANEL)
            fg = COLOR_BG      if is_best else COLOR_TEXT
            typ  = "Light GNN" if is_light_model(res["name"]) else "GNN Attn"
            name_disp = os.path.splitext(res["name"])[0]
            if is_best:
                name_disp += " [BEST]"
            vals = [str(i+1), name_disp, typ,
                    f"{res['cost']:.4f}", f"{res['avg_cost']:.4f}",
                    str(len(res["routes"]))]
            for j, (v, cw_) in enumerate(zip(vals, col_widths)):
                lbl = tk.Label(tbl, text=v, width=cw_, font=FONT_SMALL,
                               bg=bg, fg=fg, anchor="w", padx=6, relief="flat")
                lbl.grid(row=i+1, column=j, padx=1, pady=1, sticky="ew")
                sf.bind_mousewheel_to(lbl)
                tbl_sf.bind_mousewheel_to(lbl)

    def _bar_ax(self, ax, names, values, title, color):
        ax.set_facecolor(COLOR_CARD)
        ax.tick_params(colors=COLOR_MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(COLOR_BORDER)
        ax.grid(axis="y", color=COLOR_BORDER, linestyle="--",
                linewidth=0.5, zorder=0)

        x    = np.arange(len(names))
        bars = ax.bar(x, values, color=color, alpha=0.85, width=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=6.5, color=COLOR_TEXT,
                           ha="right", rotation=30)
        ax.set_title(title, color=COLOR_TEXT, fontsize=9, fontweight="bold")
        ax.yaxis.set_tick_params(colors=COLOR_MUTED)

        vmax = max(values) if values else 1
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + vmax * 0.012,
                    f"{val:.2f}" if isinstance(val, float) else str(val),
                    ha="center", va="bottom", fontsize=7, color=COLOR_TEXT)

        min_i = int(np.argmin(values))
        bars[min_i].set_edgecolor(COLOR_SUCCESS)
        bars[min_i].set_linewidth(2.5)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    app = VRPCompareApp()
    app.mainloop()

if __name__ == "__main__":
    main()
