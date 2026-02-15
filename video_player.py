import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import time
import threading
import queue
from collections import OrderedDict


DEFAULT_PALETTE_HEX = [
    "#00000000", "#ffffff", "#ff2121", "#ff93c4",
    "#ff8135", "#fff609", "#249ca3", "#78dc52",
    "#003fad", "#87f2ff", "#8e2ec4", "#a4839f",
    "#5c406c", "#e5cdc4", "#91463d", "#000000",
]


def create_color_map(palette):
    mapping = {}
    hex_chars = "0123456789abcdef"
    for char in hex_chars:
        if char == "0":
            mapping[char] = (0, 0, 0, 0)
        else:
            idx = int(char, 16)
            if idx < len(palette):
                mapping[char] = palette[idx]
            else:
                mapping[char] = (0, 0, 0, 0)
    return mapping


def render_frame_fast(frame_lines, color_map, scale=4):
    if not frame_lines:
        return Image.new("RGBA", (1, 1))

    h = len(frame_lines)
    w = len(frame_lines[0])

    flat_data = [
        color_map.get(char, (0, 0, 0, 0))
        for row in frame_lines
        for char in row
    ]

    img = Image.new("RGBA", (w, h))
    img.putdata(flat_data)
    return img.resize((w * scale, h * scale), Image.NEAREST)


def decode_rle_block(data):
    rows = data.strip().splitlines()
    output = []

    for row in rows:
        decoded = []
        i = 0
        while i < len(row):
            count = 0
            while i < len(row) and row[i].isdigit():
                count = count * 10 + int(row[i])
                i += 1
            if i < len(row):
                color = row[i]
                i += 1
                decoded.extend([color] * count)
        output.append("".join(decoded))

    return output


def parse_palettes_from_text(data):
    if "const palettes" not in data:
        return None

    palettes = []
    block = data.split("const palettes")[1]
    block = block.split("];")[0]

    frame_blocks = block.split("[")[2:]  # skip first two

    current = []
    for part in frame_blocks:
        nums = part.replace("]", "").replace(",", " ").split()
        if len(nums) >= 3:
            try:
                r, g, b = int(nums[0]), int(nums[1]), int(nums[2])
                current.append((r, g, b, 255))
                if len(current) == 15:
                    palettes.append(current)
                    current = []
            except:
                pass

    return palettes if palettes else None


class SpritePlayer:
    def __init__(self, root, scale=4):
        self.root = root
        self.scale = scale
        root.title("MakeCode Sprite Player (Enhanced)")

        self.queue = queue.Queue()
        self.stop_playback = threading.Event()

        self.raw_frames = []
        self.dynamic_palettes = None
        self.tk_cache = OrderedDict()
        self.max_cache_size = 200

        self.palette = [
            (0, 0, 0, 0) if c == "#00000000" else
            (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 255)
            for c in DEFAULT_PALETTE_HEX
        ]
        self.color_map = create_color_map(self.palette)

        self.index = 0
        self.playing = False
        self.fps_var = tk.IntVar(value=60)
        self.loop_var = tk.BooleanVar(value=True)

        self.preview = tk.Label(root, text="No Sprite Loaded", bg="#202020", fg="white")
        self.preview.pack(pady=10, fill=tk.BOTH, expand=True)

        self.slider = tk.Scale(root, from_=0, to=0, orient="horizontal",
                               command=self.scrub)
        self.slider.pack(fill=tk.X, padx=20)

        controls = tk.Frame(root)
        controls.pack(pady=5)

        tk.Button(controls, text="⏮", command=self.prev).grid(row=0, column=0, padx=5)
        tk.Button(controls, text="▶ / ⏸", command=self.toggle).grid(row=0, column=1, padx=5)
        tk.Button(controls, text="⏭", command=self.next).grid(row=0, column=2, padx=5)

        tk.Label(controls, text="FPS").grid(row=0, column=3)
        tk.Entry(controls, textvariable=self.fps_var, width=4).grid(row=0, column=4)
        tk.Checkbutton(controls, text="Loop", variable=self.loop_var).grid(row=0, column=5)

        self.bottom_frame = tk.Frame(root, bd=1, relief=tk.SUNKEN)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_bar = ttk.Progressbar(self.bottom_frame, mode="determinate")
        self.status_label = tk.Label(self.bottom_frame, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        menu = tk.Menu(root)
        root.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Sprite", command=self.start_load_sprite)

        self.check_queue()


    def start_load_sprite(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        self.playing = False
        self.stop_playback.set()

        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)
        self.progress_bar['value'] = 0
        self.status_label.config(text="Loading...")

        threading.Thread(target=self.thread_worker_load, args=(path,), daemon=True).start()

    def thread_worker_load(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()

            frames = []
            palettes = parse_palettes_from_text(data)

            if "decodeRLE" in data:
                blocks = data.split("decodeRLE(")[1:]
                total = len(blocks)

                for i, b in enumerate(blocks):
                    content = b.split("`")[1]
                    decoded = decode_rle_block(content)
                    frames.append(decoded)

                    if i % max(1, total // 100) == 0:
                        self.queue.put(("PROGRESS", int((i / total) * 100)))
            else:
                blocks = data.split("img`")[1:]
                total = len(blocks)

                for i, b in enumerate(blocks):
                    lines = b.split("`")[0].strip().splitlines()
                    frames.append([l.strip() for l in lines])

                    if i % max(1, total // 100) == 0:
                        self.queue.put(("PROGRESS", int((i / total) * 100)))

            self.queue.put(("LOAD_COMPLETE", (frames, palettes)))

        except Exception as e:
            self.queue.put(("ERROR", str(e)))


    def get_tk_frame(self, idx):
        if idx in self.tk_cache:
            self.tk_cache.move_to_end(idx)
            return self.tk_cache[idx]

        if len(self.tk_cache) >= self.max_cache_size:
            self.tk_cache.popitem(last=False)

        if self.dynamic_palettes:
            palette = self.dynamic_palettes[min(idx, len(self.dynamic_palettes) - 1)]
            temp_map = create_color_map([(0, 0, 0, 0)] + palette)
            pil = render_frame_fast(self.raw_frames[idx], temp_map, self.scale)
        else:
            pil = render_frame_fast(self.raw_frames[idx], self.color_map, self.scale)

        tk_img = ImageTk.PhotoImage(pil)
        self.tk_cache[idx] = tk_img
        return tk_img

    def refresh_ui(self):
        if not self.raw_frames:
            return
        img = self.get_tk_frame(self.index)
        self.preview.config(image=img, background="#000000")


    def scrub(self, value):
        if not self.playing:
            self.index = int(value)
            self.refresh_ui()

    def toggle(self):
        if self.playing:
            self.playing = False
            self.stop_playback.set()
        else:
            self.playing = True
            self.stop_playback.clear()
            threading.Thread(target=self.playback_worker, daemon=True).start()

    def playback_worker(self):
        fps = max(1, self.fps_var.get())
        frame_time = 1.0 / fps

        while not self.stop_playback.is_set():
            next_index = self.index + 1
            if next_index >= len(self.raw_frames):
                if self.loop_var.get():
                    next_index = 0
                else:
                    break

            self.index = next_index
            self.root.after(0, self.refresh_ui)
            time.sleep(frame_time)

    def prev(self):
        self.index = max(0, self.index - 1)
        self.refresh_ui()

    def next(self):
        self.index = min(len(self.raw_frames) - 1, self.index + 1)
        self.refresh_ui()


    def check_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()

                if msg_type == "PROGRESS":
                    self.progress_bar['value'] = data

                elif msg_type == "LOAD_COMPLETE":
                    frames, palettes = data
                    self.raw_frames = frames
                    self.dynamic_palettes = palettes
                    self.slider.config(to=max(0, len(frames) - 1))
                    self.progress_bar.pack_forget()
                    self.status_label.config(text=f"Loaded {len(frames)} frames")
                    self.index = 0
                    self.tk_cache.clear()
                    self.refresh_ui()

                elif msg_type == "ERROR":
                    messagebox.showerror("Error", data)

        except queue.Empty:
            pass

        self.root.after(16, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x600")
    SpritePlayer(root)
    root.mainloop()
