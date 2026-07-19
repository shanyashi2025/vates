import base64
import json
import tkinter as tk
import zlib
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext


def _encode(data: dict) -> str:
    """
    Encode Python dictionary to a shareable string.
    :param data: Python dictionary to be encoded
    :return: Generated shareable string
    """
    json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    compressed_bytes = zlib.compress(json_str.encode("utf-8"))
    code = "!" + base64.b64encode(compressed_bytes).decode("utf-8")
    return code


def _decode(code: str) -> dict:
    """
    Decode Python dictionary from a shareable string.
    :param code: Shareable string to be decoded
    :return: Restored Python dictionary
    """
    if code.startswith("!"):
        pure_string = code[1:]
        decoded_bytes = base64.b64decode(pure_string)
        decompressed_json_str = zlib.decompress(decoded_bytes).decode("utf-8")
        return json.loads(decompressed_json_str)
    else:
        raise ValueError("Invalid share code format. It should start with '!'.")


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.write("\n")


def format_json_text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=4)


def timestamped_message(message: str) -> str:
    return f"[{datetime.now().strftime('%H:%M:%S')}] {message}"


class ShareCodeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JSON Share Code Tool")
        self.root.geometry("900x650")

        self.status_var = tk.StringVar(value=timestamped_message("Paste a share code or load a JSON file to begin."))
        tk.Label(
            root,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            wraplength=860,
            fg="blue",
        ).pack(fill="x", padx=10, pady=(8, 4))

        share_code_frame = tk.Frame(root)
        share_code_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(share_code_frame, text="Share Code").pack(anchor="w")
        share_code_buttons = tk.Frame(share_code_frame)
        share_code_buttons.pack(fill="x", pady=(4, 0))
        tk.Button(share_code_buttons, text="Decode", command=self.decode_share_code).pack(side="left")
        tk.Button(share_code_buttons, text="Clear", command=self.clear_share_code).pack(side="left", padx=(8, 0))
        tk.Button(share_code_buttons, text="Copy to clipboard", command=self.copy_share_code).pack(side="left", padx=(8, 0))
        self.share_code_entry = tk.Entry(share_code_frame)
        self.share_code_entry.pack(fill="x", pady=(4, 0))

        json_editor_frame = tk.Frame(root)
        json_editor_frame.pack(fill="x", padx=10, pady=(4, 0))
        tk.Label(json_editor_frame, text="JSON Content").pack(anchor="w")
        json_editor_buttons = tk.Frame(json_editor_frame)
        json_editor_buttons.pack(fill="x", pady=(4, 0))
        tk.Button(json_editor_buttons, text="Load JSON file", command=self.load_json_file).pack(side="left")
        tk.Button(json_editor_buttons, text="Save as JSON file", command=self.save_json_file).pack(side="left", padx=(8, 0))
        tk.Button(json_editor_buttons, text="Generate Share Code", command=self.generate_share_code).pack(side="left", padx=(8, 0))
        tk.Button(json_editor_buttons, text="Clear", command=self.clear_json_text).pack(side="left", padx=(8, 0))
        tk.Button(json_editor_buttons, text="Copy to clipboard", command=self.copy_json_text).pack(side="left", padx=(8, 0))
        self.json_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=120, height=22)
        self.json_text.pack(fill="both", expand=True, padx=10, pady=(4, 8))


    def load_json_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            data = load_json_file(file_path)
            self.json_text.delete("1.0", tk.END)
            self.json_text.insert("1.0", format_json_text(data))
            self.status_var.set(timestamped_message(f"Loaded JSON from: {file_path}"))
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load JSON file:\n{exc}")

    def generate_share_code(self) -> None:
        text = self.json_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter or load JSON content first.")
            return

        try:
            data = json.loads(text)
            share_code = _encode(data)
            self.share_code_entry.delete(0, tk.END)
            self.share_code_entry.insert(0, share_code)
            self.status_var.set(timestamped_message("Share code generated successfully."))
        except json.JSONDecodeError as exc:
            messagebox.showerror("Error", f"Invalid JSON content:\n{exc}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to generate share code:\n{exc}")

    def decode_share_code(self) -> None:
        share_code = self.share_code_entry.get().strip()
        if not share_code:
            messagebox.showwarning("Warning", "Please paste a share code first.")
            return

        try:
            restored_data = _decode(share_code)
            self.json_text.delete("1.0", tk.END)
            self.json_text.insert("1.0", format_json_text(restored_data))
            self.status_var.set(timestamped_message("Share code decoded successfully."))
        except ValueError as exc:
            messagebox.showerror("Error", f"Invalid share code:\n{exc}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to decode share code:\n{exc}")

    def save_json_file(self) -> None:
        text = self.json_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter or decode JSON content first.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save JSON file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not output_path:
            return

        try:
            data = json.loads(text)
            save_json_file(output_path, data)
            self.status_var.set(timestamped_message(f"Saved JSON to: {output_path}"))
        except json.JSONDecodeError as exc:
            messagebox.showerror("Error", f"Invalid JSON content:\n{exc}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save JSON file:\n{exc}")

    def clear_json_text(self) -> None:
        self.json_text.delete("1.0", tk.END)
        self.status_var.set(timestamped_message("JSON content cleared."))

    def copy_json_text(self) -> None:
        text = self.json_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "JSON content is empty.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
        self.status_var.set(timestamped_message("JSON content copied to clipboard."))

    def clear_share_code(self) -> None:
        self.share_code_entry.delete(0, tk.END)
        self.status_var.set(timestamped_message("Share code cleared."))

    def copy_share_code(self) -> None:
        text = self.share_code_entry.get().strip()
        if not text:
            messagebox.showwarning("Warning", "Share code is empty.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
        self.status_var.set(timestamped_message("Share code copied to clipboard."))


def main() -> None:
    root = tk.Tk()
    app = ShareCodeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()