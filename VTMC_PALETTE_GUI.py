import os
import sys
import cv2
import threading
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS / DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

MAKECODE_WIDTH  = 160
MAKECODE_HEIGHT = 120

# Palette                  bg       surface  accent    text     dim-text
CLR_BG      = "#0d0d0d"
CLR_SURFACE = "#1a1a1a"
CLR_PANEL   = "#111111"
CLR_ACCENT  = "#ff6a00"
CLR_ACCENT2 = "#ff9a3c"
CLR_TEXT    = "#f0e6d3"
CLR_DIM     = "#6e6055"
CLR_SUCCESS = "#4caf50"
CLR_ERROR   = "#f44336"
CLR_BAR_BG  = "#2a2a2a"

FONT_TITLE  = ("Courier New", 22, "bold")
FONT_HEAD   = ("Courier New", 11, "bold")
FONT_BODY   = ("Courier New", 10)
FONT_MONO   = ("Courier New", 9)
FONT_STATUS = ("Courier New", 9, "italic")

# MakeCode Arcade default 15-colour palette (indices 1–15; 0 = transparent)
# Source: https://arcade.makecode.com/developer/images
MAKECODE_PALETTE = [
    (255, 255, 255),  # 1  white
    (255, 33,  33 ),  # 2  red
    (255, 147, 196),  # 3  pink
    (255, 129, 53 ),  # 4  orange
    (255, 246, 9  ),  # 5  yellow
    (36,  156, 163),  # 6  teal
    (120, 220, 82 ),  # 7  green
    (0,   38,  101),  # 8  dark blue
    (0,   113, 188),  # 9  blue
    (49,  30,  107),  # 10 purple
    (112, 50,  160),  # 11 violet
    (187, 115, 80 ),  # 12 tan
    (100, 65,  23 ),  # 13 brown
    (145, 70,  61 ),  # 14 dark red / maroon
    (0,   0,   0  ),  # 15 black
]


def _nearest_makecode_index(r, g, b):
    """Return 0-based index into MAKECODE_PALETTE for the nearest colour."""
    best_idx  = 0
    best_dist = float("inf")
    for i, (pr, pg, pb) in enumerate(MAKECODE_PALETTE):
        d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if d < best_dist:
            best_dist = d
            best_idx  = i
    return best_idx


# ─────────────────────────────────────────────────────────────────────────────
#  CONVERSION LOGIC  (decoupled from UI)
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path, frames_dir, start_frame, end_frame, frame_interval,
    log_cb, progress_cb, done_cb
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_cb(f"[ERROR] Cannot open video: {video_path}", "error")
        done_cb(False)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end   = end_frame if end_frame else total
    interval_ms = int(1000 / fps)

    if os.path.exists(frames_dir) and os.listdir(frames_dir):
        log_cb("Frames directory already populated — skipping extraction.", "info")
        cap.release()
        done_cb(interval_ms)
        return

    os.makedirs(frames_dir, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame
    saved = 0

    log_cb(f"Extracting frames  ({fps:.1f} fps, {total} total)…", "info")

    while frame_num < end:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_num - start_frame) % frame_interval == 0:
            fname = os.path.join(frames_dir, f"frame_{frame_num:06d}.png")
            cv2.imwrite(fname, frame)
            saved += 1
        frame_num += 1
        pct = int((frame_num - start_frame) / max(end - start_frame, 1) * 100)
        progress_cb(pct, f"Frame {frame_num}/{end}")

    cap.release()
    log_cb(f"Extraction complete — {saved} frames saved.", "ok")
    done_cb(interval_ms)


def resize_image(img, mode, scale_factor, custom_w, custom_h):
    w, h = img.size
    if mode == "scale":
        return img.resize((int(w * scale_factor), int(h * scale_factor)), Image.NEAREST)
    if mode == "full-width":
        factor = MAKECODE_WIDTH / w
        return img.resize((MAKECODE_WIDTH, int(h * factor)), Image.NEAREST)
    if mode == "full-height":
        factor = MAKECODE_HEIGHT / h
        return img.resize((int(w * factor), MAKECODE_HEIGHT), Image.NEAREST)
    if mode == "full-screen":
        return img.resize((MAKECODE_WIDTH, MAKECODE_HEIGHT), Image.NEAREST)
    if mode == "custom":
        return img.resize((custom_w, custom_h), Image.NEAREST)
    return img


def convert_image(path, mode, scale_factor, custom_w, custom_h, palette_mode="adaptive"):
    """
    palette_mode:
        "adaptive" — quantise each frame to its own best 15 colours (more accurate,
                     requires per-frame palette swaps at runtime).
        "fixed"    — snap every pixel to the nearest colour in the MakeCode default
                     palette (no runtime palette changes needed, simpler output).
    Returns (sprite_data: str, palette: list[(r,g,b)] | None)
        palette is None when palette_mode == "fixed".
    """
    img = Image.open(path).convert("RGBA")
    img = resize_image(img, mode, scale_factor, custom_w, custom_h)
    alpha = img.split()[3]

    width, height = img.size
    alpha_pix = alpha.load()

    if palette_mode == "fixed":
        rgb_pix = img.convert("RGB").load()
        rows = []
        for y in range(height):
            row = ""
            for x in range(width):
                if alpha_pix[x, y] == 0:
                    row += "0"
                else:
                    r, g, b = rgb_pix[x, y]
                    idx = _nearest_makecode_index(r, g, b)
                    row += format(idx + 1, "x")   # 1-based, '1'–'f'
            rows.append(row)
        sprite_data = "img`\n" + "\n".join(rows) + "\n`"
        return sprite_data, None

    else:  # adaptive
        rgb_img = img.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=15)
        raw_pal = rgb_img.getpalette()
        used    = sorted(set(rgb_img.getdata()))

        palette = []
        for i in range(len(used)):
            r, g, b = raw_pal[i*3], raw_pal[i*3+1], raw_pal[i*3+2]
            palette.append((r, g, b))
        while len(palette) < 15:
            palette.append((0, 0, 0))

        pix = rgb_img.load()
        rows = []
        for y in range(height):
            row = ""
            for x in range(width):
                if alpha_pix[x, y] == 0:
                    row += "0"
                else:
                    idx = pix[x, y]
                    if idx >= 15:
                        idx = 14
                    row += format(idx + 1, "x")
            rows.append(row)

        sprite_data = "img`\n" + "\n".join(rows) + "\n`"
        return sprite_data, palette


def estimate_sizes(
    video_path, frames_dir, start_frame, end_frame, frame_interval,
    mode, scale_factor, custom_w, custom_h,
    done_cb
):
    """
    Run on a background thread.  Calls done_cb(result_dict | None).

    result_dict keys:
        frame_count       int
        frames_cached     bool
        frames_size_bytes int   (0 if cached — already on disk)
        frames_size_str   str
        sprite_w          int
        sprite_h          int
        output_size_bytes int
        output_size_str   str
        total_size_bytes  int
        total_size_str    str
        video_w           int
        video_h           int
        fps               float
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            done_cb(None)
            return

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        end         = end_frame if end_frame else total
        span        = max(end - start_frame, 1)
        frame_count = len(range(start_frame, end, max(frame_interval, 1)))

        # ── Frames size ──────────────────────────────────────────────────────
        # PNG stores ~0.5–1.5 bytes/pixel depending on content; 0.9 is a
        # reasonable middle estimate for natural video frames.
        frames_cached = (
            os.path.exists(frames_dir) and
            bool(os.listdir(frames_dir))
        )
        if frames_cached:
            # Measure actual size on disk
            frames_size_bytes = sum(
                os.path.getsize(os.path.join(frames_dir, f))
                for f in os.listdir(frames_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            )
        else:
            bytes_per_frame = int(video_w * video_h * 0.9)
            frames_size_bytes = bytes_per_frame * frame_count

        # ── Output sprite dimensions after resize ────────────────────────────
        dummy = Image.new("RGB", (video_w, video_h))
        dummy = resize_image(dummy, mode, scale_factor, custom_w, custom_h)
        sprite_w, sprite_h = dummy.size

        # ── Output .txt size estimate ────────────────────────────────────────
        # Each sprite: "img`\n" + (sprite_h rows of sprite_w hex chars + \n) + "`\n"
        # Each palette block: ~15 lines of "    [rrr,ggg,bbb],\n" ≈ 20 chars each
        # Boilerplate (applyPalette fn + playback code) ≈ 400 chars
        chars_per_sprite  = 6 + sprite_h * (sprite_w + 1) + 2   # img`\n … `
        chars_per_palette = 15 * 20 + 6                          # [r,g,b],\n × 15
        boilerplate_chars = 400
        output_size_bytes = (
            frame_count * (chars_per_sprite + chars_per_palette)
            + boilerplate_chars
        )

        total_size_bytes = (
            (0 if frames_cached else frames_size_bytes)
            + output_size_bytes
        )

        def fmt(b):
            if b < 1024:
                return f"{b} B"
            elif b < 1024 ** 2:
                return f"{b/1024:.1f} KB"
            elif b < 1024 ** 3:
                return f"{b/1024**2:.1f} MB"
            else:
                return f"{b/1024**3:.2f} GB"

        done_cb({
            "frame_count":       frame_count,
            "frames_cached":     frames_cached,
            "frames_size_bytes": frames_size_bytes,
            "frames_size_str":   fmt(frames_size_bytes),
            "sprite_w":          sprite_w,
            "sprite_h":          sprite_h,
            "output_size_bytes": output_size_bytes,
            "output_size_str":   fmt(output_size_bytes),
            "total_size_bytes":  total_size_bytes,
            "total_size_str":    fmt(total_size_bytes),
            "video_w":           video_w,
            "video_h":           video_h,
            "fps":               fps,
        })

    except Exception as e:
        done_cb(None)


def frames_to_makecode(
    frames_dir, output_file, mode, scale_factor, custom_w, custom_h,
    interval_ms, palette_mode, log_cb, progress_cb, done_cb
):
    files = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )
    total = len(files)
    if total == 0:
        log_cb("[ERROR] No frames found in directory.", "error")
        done_cb(False)
        return

    log_cb(f"Converting {total} frames  [palette: {palette_mode}]…", "info")
    sprites  = []
    palettes = []

    for i, filename in enumerate(files):
        path = os.path.join(frames_dir, filename)
        try:
            sprite, pal = convert_image(
                path, mode, scale_factor, custom_w, custom_h, palette_mode
            )
            sprites.append(sprite)
            if pal is not None:
                palettes.append(pal)
        except Exception as e:
            log_cb(f"[WARN] Skipping {filename}: {e}", "warn")
        pct = int((i + 1) / total * 100)
        progress_cb(pct, f"Converting {i+1}/{total}")

    log_cb("Writing output file…", "info")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("const frames: Image[] = [\n")
        f.write(",\n".join(sprites))
        f.write("\n]\n\n")

        if palette_mode == "adaptive":
            # Per-frame palette array + runtime swap function
            f.write("const palettes = [\n")
            for p in palettes:
                f.write("  [\n")
                for (r, g, b) in p:
                    f.write(f"    [{r},{g},{b}],\n")
                f.write("  ],\n")
            f.write("]\n\n")

            f.write("""let lastPalette = -1

function applyPalette(frame: number) {
    if (frame == lastPalette) return
    lastPalette = frame
    for (let i = 0; i < 15; i++) {
        let c = palettes[frame][i]
        color.setColor(i + 1, color.rgb(c[0], c[1], c[2]))
    }
}
""")
            f.write(f"""
let video = sprites.create(frames[0])
video.setPosition(80, 60)

let frame = 0
applyPalette(0)

game.onUpdateInterval({interval_ms}, function () {{
    frame = (frame + 1) % frames.length
    applyPalette(frame)
    video.setImage(frames[frame])
}})
""")

        else:  # fixed — no palette array, no applyPalette
            f.write(f"""
let video = sprites.create(frames[0])
video.setPosition(80, 60)

let frame = 0

game.onUpdateInterval({interval_ms}, function () {{
    frame = (frame + 1) % frames.length
    video.setImage(frames[frame])
}})
""")

    log_cb(f"Done! Output written to: {output_file}", "ok")
    done_cb(True)


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MakeCode Video Converter")
        self.resizable(False, False)
        self.configure(bg=CLR_BG)

        self._interval_ms = 33
        self._running     = False

        self._build_ui()
        self._on_mode_change()          # sync custom fields visibility
        self._on_palette_mode_change()  # sync palette note

    # ── UI CONSTRUCTION ──────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=16, pady=6)

        # ── TITLE BAR ────────────────────────────────────────────────────────
        title_frame = tk.Frame(self, bg=CLR_ACCENT, height=56)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        tk.Label(
            title_frame,
            text="◈  MAKECODE VIDEO CONVERTER  ◈",
            font=FONT_TITLE,
            bg=CLR_ACCENT,
            fg=CLR_BG,
        ).pack(expand=True)

        # ── MAIN BODY ────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=CLR_BG)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        left  = tk.Frame(body, bg=CLR_BG)
        right = tk.Frame(body, bg=CLR_BG)
        left.pack(side="left", fill="both", expand=True, padx=(12,4), pady=12)
        right.pack(side="right", fill="both", expand=True, padx=(4,12), pady=12)

        # ── LEFT: PATHS ───────────────────────────────────────────────────────
        self._section(left, "PATHS")

        self.video_path = self._path_row(left, "Video file", self._browse_video)
        self.frames_dir = self._path_row(left, "Frames dir", self._browse_frames)
        self.output_file = self._path_row(left, "Output .txt", self._browse_output)

        # ── LEFT: FRAME RANGE ────────────────────────────────────────────────
        self._section(left, "FRAME RANGE")

        rng = tk.Frame(left, bg=CLR_BG)
        rng.pack(fill="x", pady=(0,4))

        self.start_frame = self._int_field(rng, "Start frame", 0, side="left")
        self.end_frame   = self._int_field(rng, "End frame (blank=all)", "", side="left")
        self.frame_interval = self._int_field(rng, "Interval", 1, side="left")

        # ── RIGHT: RESIZE MODE ───────────────────────────────────────────────
        self._section(right, "RESIZE MODE")

        self.mode_var = tk.StringVar(value="full-width")
        modes = [
            ("full-width",  "Fit full width  (160px wide, scaled height)"),
            ("full-height", "Fit full height (120px tall, scaled width)"),
            ("full-screen", "Fit full screen (160×120, stretched)"),
            ("scale",       "Scale factor"),
            ("custom",      "Custom dimensions"),
        ]
        for val, label in modes:
            rb = tk.Radiobutton(
                right, text=label,
                variable=self.mode_var, value=val,
                bg=CLR_BG, fg=CLR_TEXT,
                selectcolor=CLR_ACCENT,
                activebackground=CLR_BG,
                activeforeground=CLR_ACCENT2,
                font=FONT_BODY,
                command=self._on_mode_change,
            )
            rb.pack(anchor="w", padx=8)

        # scale / custom sub-fields
        self.extra_frame = tk.Frame(right, bg=CLR_BG)
        self.extra_frame.pack(fill="x", padx=8, pady=4)

        self.scale_factor   = self._float_field(self.extra_frame, "Scale factor", 0.1)
        self.custom_width   = self._int_field(self.extra_frame, "Width",  160, side="left")
        self.custom_height  = self._int_field(self.extra_frame, "Height", 120, side="left")

        # ── RIGHT: OUTPUT DIMENSIONS DISPLAY ────────────────────────────────
        self._section(right, "TARGET DIMENSIONS")

        self.dim_label = tk.Label(
            right, text="160 × 120 px",
            font=("Courier New", 14, "bold"),
            bg=CLR_BG, fg=CLR_ACCENT2,
        )
        self.dim_label.pack(anchor="w", padx=8, pady=(0,8))

        # ── RIGHT: PALETTE MODE ──────────────────────────────────────────────
        self._section(right, "PALETTE MODE")

        self.palette_mode_var = tk.StringVar(value="adaptive")

        pal_opts = [
            (
                "adaptive",
                "Adaptive  (per-frame colours, more accurate)",
                "Each frame gets its own best-fit 15-colour palette.\n"
                "Runtime palette swaps reproduce the original colours closely.\n"
                "Larger output file.",
            ),
            (
                "fixed",
                "Fixed  (MakeCode default palette, smaller output)",
                "Every pixel is snapped to MakeCode's built-in 16 colours.\n"
                "No runtime palette code needed — simpler & smaller output.\n"
                "Colours are approximate.",
            ),
        ]

        for val, label, tip in pal_opts:
            rb = tk.Radiobutton(
                right, text=label,
                variable=self.palette_mode_var, value=val,
                bg=CLR_BG, fg=CLR_TEXT,
                selectcolor=CLR_ACCENT,
                activebackground=CLR_BG,
                activeforeground=CLR_ACCENT2,
                font=FONT_BODY,
                command=self._on_palette_mode_change,
            )
            rb.pack(anchor="w", padx=8)

        self.palette_note = tk.Label(
            right, text="",
            font=FONT_STATUS, bg=CLR_BG, fg=CLR_DIM,
            justify="left", wraplength=300,
        )
        self.palette_note.pack(anchor="w", padx=20, pady=(0, 6))

        # ── SIZE ESTIMATE PANEL ──────────────────────────────────────────────
        est_outer = tk.Frame(self, bg=CLR_PANEL)
        est_outer.pack(fill="x", padx=12, pady=(0, 6))

        est_title_row = tk.Frame(est_outer, bg=CLR_PANEL)
        est_title_row.pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(
            est_title_row, text="── SIZE ESTIMATE ",
            font=FONT_HEAD, bg=CLR_PANEL, fg=CLR_ACCENT,
        ).pack(side="left")
        tk.Frame(est_title_row, bg=CLR_DIM, height=1).pack(
            side="left", fill="x", expand=True, pady=5,
        )
        self.est_btn = tk.Button(
            est_title_row,
            text="⊞  CALCULATE",
            font=("Courier New", 9, "bold"),
            bg=CLR_SURFACE, fg=CLR_ACCENT2,
            activebackground=CLR_ACCENT, activeforeground=CLR_BG,
            relief="flat", bd=0, padx=10, pady=2,
            cursor="hand2",
            command=self._run_estimate,
        )
        self.est_btn.pack(side="right", padx=(8, 0))

        est_grid = tk.Frame(est_outer, bg=CLR_PANEL)
        est_grid.pack(fill="x", padx=10, pady=(0, 2))

        def est_cell(col, row, label):
            tk.Label(
                est_grid, text=label,
                font=FONT_MONO, bg=CLR_PANEL, fg=CLR_DIM, anchor="w",
            ).grid(row=row, column=col*2,   sticky="w", padx=(8, 2), pady=1)
            var = tk.StringVar(value="—")
            tk.Label(
                est_grid, textvariable=var,
                font=("Courier New", 9, "bold"),
                bg=CLR_PANEL, fg=CLR_ACCENT2, anchor="w",
            ).grid(row=row, column=col*2+1, sticky="w", padx=(0, 24), pady=1)
            return var

        self.est_frames      = est_cell(0, 0, "Frames to extract:")
        self.est_sprite_dim  = est_cell(1, 0, "Sprite dimensions:")
        self.est_fps         = est_cell(2, 0, "Source FPS:")
        self.est_frames_size = est_cell(0, 1, "Frames disk size:")
        self.est_output_size = est_cell(1, 1, "Output .txt size:")
        self.est_total_size  = est_cell(2, 1, "Total new data:")

        self.est_note = tk.Label(
            est_outer, text="",
            font=FONT_STATUS, bg=CLR_PANEL, fg=CLR_DIM,
        )
        self.est_note.pack(anchor="w", padx=18, pady=(0, 6))

        # ── BOTTOM: PROGRESS + LOG ───────────────────────────────────────────
        bottom = tk.Frame(self, bg=CLR_PANEL, bd=0)
        bottom.pack(fill="both", expand=True, padx=12, pady=(0,12))

        # Phase labels
        phase_row = tk.Frame(bottom, bg=CLR_PANEL)
        phase_row.pack(fill="x", padx=12, pady=(10,2))

        self.phase_label = tk.Label(
            phase_row, text="PHASE 1 / 2  —  EXTRACT FRAMES",
            font=FONT_HEAD, bg=CLR_PANEL, fg=CLR_DIM,
        )
        self.phase_label.pack(side="left")

        self.pct_label = tk.Label(
            phase_row, text="0 %",
            font=FONT_HEAD, bg=CLR_PANEL, fg=CLR_ACCENT,
        )
        self.pct_label.pack(side="right")

        # Progress bar
        bar_frame = tk.Frame(bottom, bg=CLR_PANEL)
        bar_frame.pack(fill="x", padx=12, pady=(0,4))

        self.progress_bg = tk.Frame(bar_frame, bg=CLR_BAR_BG, height=18)
        self.progress_bg.pack(fill="x")
        self.progress_bg.pack_propagate(False)

        self.progress_fill = tk.Frame(self.progress_bg, bg=CLR_ACCENT, height=18, width=0)
        self.progress_fill.place(x=0, y=0, relheight=1)

        # Step label
        self.step_label = tk.Label(
            bottom, text="Waiting…",
            font=FONT_STATUS, bg=CLR_PANEL, fg=CLR_DIM,
        )
        self.step_label.pack(anchor="w", padx=12, pady=(0,6))

        # Log box
        log_frame = tk.Frame(bottom, bg=CLR_PANEL)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0,10))

        self.log_box = tk.Text(
            log_frame,
            height=10,
            bg="#0a0a0a", fg=CLR_DIM,
            font=FONT_MONO,
            relief="flat",
            bd=0,
            state="disabled",
            wrap="word",
            insertbackground=CLR_ACCENT,
        )
        self.log_box.pack(side="left", fill="both", expand=True)

        sb = tk.Scrollbar(log_frame, command=self.log_box.yview, bg=CLR_SURFACE)
        sb.pack(side="right", fill="y")
        self.log_box.configure(yscrollcommand=sb.set)

        self.log_box.tag_config("ok",    foreground=CLR_SUCCESS)
        self.log_box.tag_config("error", foreground=CLR_ERROR)
        self.log_box.tag_config("warn",  foreground="#f5c518")
        self.log_box.tag_config("info",  foreground=CLR_ACCENT2)
        self.log_box.tag_config("dim",   foreground=CLR_DIM)

        # Run button
        btn_row = tk.Frame(self, bg=CLR_BG)
        btn_row.pack(fill="x", padx=12, pady=(0,14))

        self.run_btn = tk.Button(
            btn_row,
            text="▶  EXTRACT  +  CONVERT",
            font=("Courier New", 12, "bold"),
            bg=CLR_ACCENT, fg=CLR_BG,
            activebackground=CLR_ACCENT2,
            activeforeground=CLR_BG,
            relief="flat", bd=0,
            padx=24, pady=10,
            cursor="hand2",
            command=self._start,
        )
        self.run_btn.pack(side="right")

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _section(self, parent, text):
        f = tk.Frame(parent, bg=CLR_BG)
        f.pack(fill="x", pady=(10, 2))
        tk.Label(f, text=f"── {text} ", font=FONT_HEAD,
                 bg=CLR_BG, fg=CLR_ACCENT).pack(side="left")
        tk.Frame(f, bg=CLR_DIM, height=1).pack(side="left", fill="x", expand=True)

    def _path_row(self, parent, label, browse_cmd):
        row = tk.Frame(parent, bg=CLR_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=f"{label}:", font=FONT_BODY,
                 bg=CLR_BG, fg=CLR_DIM, width=18, anchor="w").pack(side="left")
        var = tk.StringVar()
        entry = tk.Entry(row, textvariable=var, font=FONT_MONO,
                         bg=CLR_SURFACE, fg=CLR_TEXT,
                         insertbackground=CLR_ACCENT,
                         relief="flat", bd=4)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(row, text="…", font=FONT_BODY,
                  bg=CLR_SURFACE, fg=CLR_ACCENT2,
                  activebackground=CLR_ACCENT,
                  activeforeground=CLR_BG,
                  relief="flat", bd=0, padx=6,
                  cursor="hand2",
                  command=browse_cmd).pack(side="left")
        return var

    def _int_field(self, parent, label, default, side="top"):
        f = tk.Frame(parent, bg=CLR_BG)
        f.pack(side=side, padx=(0, 12), pady=2, anchor="w")
        tk.Label(f, text=f"{label}:", font=FONT_MONO,
                 bg=CLR_BG, fg=CLR_DIM).pack(anchor="w")
        var = tk.StringVar(value=str(default))
        tk.Entry(f, textvariable=var, font=FONT_MONO, width=8,
                 bg=CLR_SURFACE, fg=CLR_TEXT,
                 insertbackground=CLR_ACCENT,
                 relief="flat", bd=4).pack()
        return var

    def _float_field(self, parent, label, default):
        f = tk.Frame(parent, bg=CLR_BG)
        f.pack(anchor="w", pady=2)
        tk.Label(f, text=f"{label}:", font=FONT_MONO,
                 bg=CLR_BG, fg=CLR_DIM).pack(anchor="w")
        var = tk.StringVar(value=str(default))
        tk.Entry(f, textvariable=var, font=FONT_MONO, width=8,
                 bg=CLR_SURFACE, fg=CLR_TEXT,
                 insertbackground=CLR_ACCENT,
                 relief="flat", bd=4).pack()
        return var

    # ── BROWSE CALLBACKS ─────────────────────────────────────────────────────

    def _browse_video(self):
        p = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All", "*.*")]
        )
        if p:
            self.video_path.set(p)

    def _browse_frames(self):
        p = filedialog.askdirectory(title="Select frames output directory")
        if p:
            self.frames_dir.set(p)

    def _browse_output(self):
        p = filedialog.asksaveasfilename(
            title="Save MakeCode output",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All", "*.*")]
        )
        if p:
            self.output_file.set(p)

    # ── MODE CHANGE ──────────────────────────────────────────────────────────

    def _on_mode_change(self):
        mode = self.mode_var.get()
        # Hide all sub-widgets first
        for w in self.extra_frame.winfo_children():
            w.pack_forget()

        if mode == "scale":
            self.scale_factor.get()     # just re-show
            for w in self.extra_frame.winfo_children():
                if "Scale" in str(w) or True:
                    pass
            # show only scale row
            self.extra_frame.winfo_children()[0].pack(anchor="w", pady=2)
        elif mode == "custom":
            for w in self.extra_frame.winfo_children()[1:]:
                w.pack(side="left", padx=(0, 12), pady=2, anchor="w")

        self._update_dim_label()

    def _update_dim_label(self):
        mode = self.mode_var.get()
        if mode == "full-width":
            txt = f"{MAKECODE_WIDTH} px wide  (height auto)"
        elif mode == "full-height":
            txt = f"{MAKECODE_HEIGHT} px tall  (width auto)"
        elif mode == "full-screen":
            txt = f"{MAKECODE_WIDTH} × {MAKECODE_HEIGHT} px  (stretched)"
        elif mode == "scale":
            try:
                sf = float(self.scale_factor.get())
                txt = f"× {sf:.2f}  (original size scaled)"
            except Exception:
                txt = "scale factor"
        elif mode == "custom":
            try:
                w = self.custom_width.get() or "?"
                h = self.custom_height.get() or "?"
                txt = f"{w} × {h} px"
            except Exception:
                txt = "custom"
        else:
            txt = "—"
        self.dim_label.config(text=txt)

    def _on_palette_mode_change(self):
        mode = self.palette_mode_var.get()
        if mode == "adaptive":
            self.palette_note.config(
                text="Per-frame palette swaps at runtime. Best colour accuracy, larger .txt file. Color fading extension is required.",
                fg=CLR_ACCENT2,
            )
        else:
            self.palette_note.config(
                text="Pixels snapped to MakeCode's 15 built-in colours. Smaller, simpler output.",
                fg=CLR_DIM,
            )

    # ── LOGGING ──────────────────────────────────────────────────────────────

    def _log(self, msg, tag="dim"):
        def _do():
            self.log_box.config(state="normal")
            self.log_box.insert("end", f"{msg}\n", tag)
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _do)

    # ── PROGRESS ─────────────────────────────────────────────────────────────

    def _progress(self, pct, step_text=""):
        def _do():
            self.pct_label.config(text=f"{pct} %")
            self.step_label.config(text=step_text)
            # animate bar width
            total_w = self.progress_bg.winfo_width()
            fill_w  = max(2, int(total_w * pct / 100))
            self.progress_fill.place_configure(width=fill_w)
            # colour shift: orange → green at 100
            if pct >= 100:
                self.progress_fill.config(bg=CLR_SUCCESS)
            else:
                self.progress_fill.config(bg=CLR_ACCENT)
        self.after(0, _do)

    def _set_phase(self, text, colour=CLR_DIM):
        self.after(0, lambda: self.phase_label.config(text=text, fg=colour))

    # ── VALIDATION ───────────────────────────────────────────────────────────

    def _validate(self):
        errors = []
        if not self.video_path.get():
            errors.append("Video file path is required.")
        if not self.frames_dir.get():
            errors.append("Frames directory is required.")
        if not self.output_file.get():
            errors.append("Output file path is required.")
        try:
            int(self.start_frame.get())
        except Exception:
            errors.append("Start frame must be an integer.")
        ef = self.end_frame.get().strip()
        if ef:
            try:
                int(ef)
            except Exception:
                errors.append("End frame must be an integer or blank.")
        try:
            fi = int(self.frame_interval.get())
            if fi < 1:
                raise ValueError
        except Exception:
            errors.append("Frame interval must be a positive integer.")
        if self.mode_var.get() == "scale":
            try:
                float(self.scale_factor.get())
            except Exception:
                errors.append("Scale factor must be a number.")
        if self.mode_var.get() == "custom":
            try:
                int(self.custom_width.get())
                int(self.custom_height.get())
            except Exception:
                errors.append("Custom width and height must be integers.")
        return errors

    # ── SIZE ESTIMATION ──────────────────────────────────────────────────────

    def _run_estimate(self):
        if self._running:
            return
        video_path = self.video_path.get().strip()
        if not video_path or not os.path.isfile(video_path):
            messagebox.showerror("Estimate Error", "Please select a valid video file first.")
            return

        # Parse best-effort (fall back to defaults on bad input)
        try:
            start_frame = int(self.start_frame.get())
        except Exception:
            start_frame = 0
        try:
            ef_val  = self.end_frame.get().strip()
            end_frame = int(ef_val) if ef_val else None
        except Exception:
            end_frame = None
        try:
            frame_interval = max(1, int(self.frame_interval.get()))
        except Exception:
            frame_interval = 1

        mode        = self.mode_var.get()
        frames_dir  = self.frames_dir.get().strip()

        try:
            scale_factor = float(self.scale_factor.get()) if mode == "scale" else 0.1
        except Exception:
            scale_factor = 0.1
        try:
            custom_w = int(self.custom_width.get())  if mode == "custom" else None
            custom_h = int(self.custom_height.get()) if mode == "custom" else None
        except Exception:
            custom_w = custom_h = None

        # Show spinner state
        self.est_btn.config(state="disabled", text="⏳  …")
        for var in (self.est_frames, self.est_sprite_dim, self.est_fps,
                    self.est_frames_size, self.est_output_size, self.est_total_size):
            var.set("…")
        self.est_note.config(text="")

        def _done(result):
            self.after(0, lambda: self._show_estimate(result))

        threading.Thread(
            target=estimate_sizes,
            args=(
                video_path, frames_dir, start_frame, end_frame, frame_interval,
                mode, scale_factor, custom_w, custom_h,
                _done,
            ),
            daemon=True,
        ).start()

    def _show_estimate(self, r):
        self.est_btn.config(state="normal", text="⊞  CALCULATE")

        if r is None:
            for var in (self.est_frames, self.est_sprite_dim, self.est_fps,
                        self.est_frames_size, self.est_output_size, self.est_total_size):
                var.set("error")
            self.est_note.config(
                text="Could not read video. Check the path and try again.",
                fg=CLR_ERROR,
            )
            return

        self.est_frames.set(f"{r['frame_count']:,}")
        self.est_sprite_dim.set(f"{r['sprite_w']} × {r['sprite_h']} px")
        self.est_fps.set(f"{r['fps']:.2f}")

        if r["frames_cached"]:
            self.est_frames_size.set(f"{r['frames_size_str']}  (cached)")
        else:
            self.est_frames_size.set(r["frames_size_str"])

        self.est_output_size.set(r["output_size_str"])

        if r["frames_cached"]:
            self.est_total_size.set(f"{r['output_size_str']}  (frames already exist)")
        else:
            self.est_total_size.set(r["total_size_str"])

        # Warning thresholds
        if r["output_size_bytes"] > 50 * 1024 * 1024:
            note = f"⚠  Output file is very large ({r['output_size_str']}) — MakeCode may struggle to load it."
            fg   = CLR_ERROR
        elif r["output_size_bytes"] > 10 * 1024 * 1024:
            note = f"△  Output file is large ({r['output_size_str']}) — consider using a higher frame interval."
            fg   = "#f5c518"
        elif not r["frames_cached"] and r["frames_size_bytes"] > 2 * 1024 ** 3:
            note = f"△  Frames will use ~{r['frames_size_str']} of disk space."
            fg   = "#f5c518"
        else:
            note = f"✔  Estimates look fine.  Source: {r['video_w']}×{r['video_h']} @ {r['fps']:.1f} fps"
            fg   = CLR_SUCCESS

        self.est_note.config(text=note, fg=fg)

    # ── START PIPELINE ────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return

        errors = self._validate()
        if errors:
            messagebox.showerror("Input Error", "\n".join(errors))
            return

        self._running = True
        self.run_btn.config(state="disabled", text="⏳  RUNNING…")
        self.progress_fill.config(bg=CLR_ACCENT)

        # Parse fields
        video_path     = self.video_path.get()
        frames_dir     = self.frames_dir.get()
        output_file    = self.output_file.get()
        start_frame    = int(self.start_frame.get())
        ef_val         = self.end_frame.get().strip()
        end_frame      = int(ef_val) if ef_val else None
        frame_interval = int(self.frame_interval.get())
        mode           = self.mode_var.get()
        scale_factor   = float(self.scale_factor.get()) if mode == "scale" else 0.1
        custom_w       = int(self.custom_width.get()) if mode == "custom" else None
        custom_h       = int(self.custom_height.get()) if mode == "custom" else None
        palette_mode   = self.palette_mode_var.get()

        self._log("━" * 54, "dim")
        self._log("▶  Starting pipeline…", "info")

        # ── PHASE 1 ──
        self._set_phase("PHASE 1 / 2  —  EXTRACTING FRAMES", CLR_ACCENT2)
        self._progress(0, "Initialising…")

        def after_extraction(result):
            if result is False:
                self._finish(success=False)
                return
            self._interval_ms = result
            self._log("", "dim")
            self._progress(0, "Starting conversion…")
            self._set_phase("PHASE 2 / 2  —  CONVERTING TO MAKECODE", CLR_ACCENT2)

            # ── PHASE 2 ──
            def after_conversion(ok):
                self._finish(success=bool(ok))

            threading.Thread(
                target=frames_to_makecode,
                args=(
                    frames_dir, output_file, mode,
                    scale_factor, custom_w, custom_h,
                    self._interval_ms,
                    palette_mode,
                    self._log, self._progress, after_conversion,
                ),
                daemon=True,
            ).start()

        threading.Thread(
            target=extract_frames,
            args=(
                video_path, frames_dir, start_frame, end_frame, frame_interval,
                self._log, self._progress, after_extraction,
            ),
            daemon=True,
        ).start()

    def _finish(self, success: bool):
        def _do():
            self._running = False
            if success:
                self._set_phase("✔  COMPLETE", CLR_SUCCESS)
                self._progress(100, "All done!")
                self.run_btn.config(
                    state="normal",
                    text="▶  EXTRACT  +  CONVERT",
                    bg=CLR_SUCCESS,
                )
            else:
                self._set_phase("✖  ERROR", CLR_ERROR)
                self.progress_fill.config(bg=CLR_ERROR)
                self.run_btn.config(
                    state="normal",
                    text="▶  EXTRACT  +  CONVERT",
                    bg=CLR_ACCENT,
                )
        self.after(0, _do)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()