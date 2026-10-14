"""
Chat UI for the SFT and DPO models.

Type a prompt in the browser and get an answer from the SFT model and an
answer from the DPO model. Nothing is scored or compared.

Run from the project root:

    python ui/server.py
    python ui/server.py --port 8000

Then open http://127.0.0.1:7860

Design notes:
  - Only the Python standard library is used for the web server, so no
    new packages are needed.
  - The 4-bit base model is loaded ONCE. Both LoRA adapters are attached
    to it and switched per request (same approach as src/eval/chat.py).
  - A lock makes sure only one generation runs at a time, because there
    is one GPU and adapter switching is global state.
  - The server only listens on 127.0.0.1 (this computer).
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Make `src` importable and make relative paths (configs/, checkpoints/)
# work no matter where the script is started from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT)
os.chdir(ROOT)

from peft import PeftModel  # noqa: E402

from src.eval.generate_answers import generate  # noqa: E402
from src.models.load_model import load_base_model, load_tokenizer  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402


logger = get_logger("ui.server")


# ============================================================
# CONFIG
# ============================================================

MODEL_CONFIG_PATH = "configs/model_config.yaml"
SFT_CONFIG_PATH = "configs/sft_config.yaml"
DPO_CONFIG_PATH = "configs/dpo_config.yaml"

INDEX_PATH = os.path.join(ROOT, "ui", "index.html")

MAX_PROMPT_CHARS = 2000
MAX_BODY_BYTES = 20_000

# "sft" / "dpo" are the adapter names used inside the PEFT model.
VALID_MODELS = ("sft", "dpo")


# ============================================================
# MODEL (loaded once at startup)
# ============================================================

class ModelService:
    """
    Holds the base model with both adapters and generates answers.
    """

    def __init__(self):

        model_cfg = load_config(MODEL_CONFIG_PATH)
        sft_path = load_config(SFT_CONFIG_PATH)["output_dir"]
        dpo_path = load_config(DPO_CONFIG_PATH)["output_dir"]

        logger.info("Loading tokenizer...")

        self.tokenizer = load_tokenizer(model_cfg, padding_side="left")

        logger.info("Loading 4-bit base model (this takes a moment)...")

        base = load_base_model(model_cfg)

        logger.info(f"Attaching SFT adapter: {sft_path}")

        self.model = PeftModel.from_pretrained(
            base,
            sft_path,
            adapter_name="sft",
            is_trainable=False,
        )

        logger.info(f"Attaching DPO adapter: {dpo_path}")

        self.model.load_adapter(
            dpo_path,
            adapter_name="dpo",
            is_trainable=False,
        )

        # load_base_model() turns the KV cache off for training.
        self.model.config.use_cache = True

        self.model.eval()

        # One generation at a time (single GPU, global adapter state).
        self.lock = threading.Lock()

        logger.info("Both models are ready.")

    def answer(self, which: str, prompt: str) -> str:

        with self.lock:

            self.model.set_adapter(which)

            # Greedy decoding with the same settings used in evaluation,
            # so the UI shows what the evaluation measured.
            return generate(self.model, self.tokenizer, prompt)


SERVICE = None


# ============================================================
# HTTP
# ============================================================

class Handler(BaseHTTPRequestHandler):

    server_version = "AlignmentPlayground/1.0"

    # --------------------------------------------------------
    # helpers
    # --------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str):

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict):

        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def log_message(self, fmt, *args):
        # Keep the terminal quiet; model logs are enough.
        pass

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def do_GET(self):

        if self.path in ("/", "/index.html"):

            with open(INDEX_PATH, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")

            return

        if self.path == "/api/health":
            self._json(200, {"ready": SERVICE is not None})
            return

        self._json(404, {"error": "Not found"})

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(self):

        if self.path != "/api/generate":
            self._json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(400, {"error": "Invalid request size."})
            return

        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"error": "Invalid JSON."})
            return

        prompt = str(data.get("prompt", "")).strip()
        which = str(data.get("model", "")).strip().lower()

        if not prompt:
            self._json(400, {"error": "Please type a prompt."})
            return

        if len(prompt) > MAX_PROMPT_CHARS:
            self._json(
                400,
                {"error": f"Prompt is too long (max {MAX_PROMPT_CHARS} characters)."},
            )
            return

        if which not in VALID_MODELS:
            self._json(400, {"error": "model must be 'sft' or 'dpo'."})
            return

        started = time.time()

        try:
            answer = SERVICE.answer(which, prompt)

        except Exception as error:  # report any failure to the page
            logger.exception("Generation failed")
            self._json(500, {"error": f"Generation failed: {error}"})
            return

        self._json(
            200,
            {
                "model": which,
                "answer": answer,
                "seconds": round(time.time() - started, 1),
            },
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global SERVICE

    parser = argparse.ArgumentParser(description="SFT / DPO chat UI")

    parser.add_argument("--port", type=int, default=7860)

    args = parser.parse_args()

    SERVICE = ModelService()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)

    print()
    print("=" * 60)
    print(f"Open in your browser:  http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print()
        print("Stopping...")

    finally:
        server.server_close()


if __name__ == "__main__":
    main()