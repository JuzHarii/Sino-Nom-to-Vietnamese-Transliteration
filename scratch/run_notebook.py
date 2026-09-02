import io
import json
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# Ensure utf-8 stdout/stderr in runner
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(r"c:\Study\HTK\final_proj\Sino-Nom-to-Vietnamese-Transliteration")
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_Evaluate_HCMUS_API.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global execution context
exec_globals = {
    '__name__': '__main__',
    '__file__': str(NOTEBOOK_PATH),
}

# Custom display function for notebook cells
display_outputs = []
def custom_display(*args):
    for arg in args:
        display_outputs.append(arg)

exec_globals['display'] = custom_display

print(f"[*] Starting execution of {NOTEBOOK_PATH.name}...")

exec_count = 1
for cell_idx, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue

    source_code = "".join(cell.get("source", []))
    print(f"--- Running Code Cell {exec_count} ---")
    
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    display_outputs = []
    
    cell["execution_count"] = exec_count
    cell["outputs"] = []

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            # Execute code in persistent globals
            exec(source_code, exec_globals)
        
        out_text = stdout_buf.getvalue()
        err_text = stderr_buf.getvalue()

        if out_text:
            cell["outputs"].append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in out_text.splitlines()]
            })
            print(out_text.strip())

        if err_text:
            cell["outputs"].append({
                "name": "stderr",
                "output_type": "stream",
                "text": [line + "\n" for line in err_text.splitlines()]
            })
            print(f"[stderr]: {err_text.strip()}")

        for item in display_outputs:
            if hasattr(item, "to_html") and hasattr(item, "to_string"):
                cell["outputs"].append({
                    "data": {
                        "text/html": [item.to_html() + "\n"],
                        "text/plain": [item.to_string() + "\n"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                print(item.to_string())
            else:
                cell["outputs"].append({
                    "data": {
                        "text/plain": [str(item) + "\n"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                print(str(item))
                
    except Exception as e:
        print(f"[Error in cell {exec_count}]: {e}")
        cell["outputs"].append({
            "ename": type(e).__name__,
            "evalue": str(e),
            "output_type": "error",
            "traceback": [f"{type(e).__name__}: {e}"]
        })
        break

    exec_count += 1

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"\n[+] Successfully executed all cells and updated {NOTEBOOK_PATH.name}!")
