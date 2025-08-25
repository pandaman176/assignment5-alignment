import pathlib
import socket

hostname = socket.gethostname()
is_local: bool = "MacBook" in hostname

MODEL_PATH = pathlib.Path(__file__).parent.parent / "model" if is_local else pathlib.Path("/root/autodl-fs")
PROMPT_PATH = pathlib.Path(__file__).parent / "prompts"
LOG_PATH = pathlib.Path(__file__).parent.parent / "logs" if is_local else pathlib.Path("/root/autodl-fs")
DATA_PATH = pathlib.Path(__file__).parent.parent / "data"