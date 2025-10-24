"""Setup helper for installing vLLM."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

def install_vllm() -> None:
    """Run the vLLM installation script."""
    # Navigate up from src/vllm_rbln_exec/setup.py to project root
    project_root = Path(__file__).parent.parent.parent
    script = project_root / "scripts" / "install-vllm.sh"
    
    if not script.exists():
        print(f"❌ Installation script not found: {script}")
        print("Please run from the repository root:")
        print("  bash scripts/install-vllm.sh")
        sys.exit(1)
    
    print(f"Running vLLM installation script: {script}")
    try:
        subprocess.run(["bash", str(script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ bash not found. Please run the script manually:")
        print(f"  bash {script}")
        sys.exit(1)

if __name__ == "__main__":
    install_vllm()