#!/usr/bin/env python3
"""
Patch vLLM model files to add defensive guards for missing parameters.
This allows num_hidden_layers override without KeyError.
"""

import os
import sys
import re
from pathlib import Path

VLLM_SOURCE = os.environ.get("VLLM_SOURCE", "./vllm_source")

# Models to patch
MODEL_FILES = [
    "vllm/model_executor/models/llama.py",
    "vllm/model_executor/models/qwen2.py",
    "vllm/model_executor/models/qwen2_moe.py",
    "vllm/model_executor/models/qwen3.py",
    "vllm/model_executor/models/qwen3_moe.py",
    "vllm/model_executor/models/deepseek_v2.py",
]

GUARD_COMMENT = "# PATCH: guard for num_hidden_layers override"
GUARD_CODE = """                    if name not in params_dict:
                        continue
"""

def patch_file(file_path: Path) -> bool:
    """Add 'if name not in params_dict: continue' guards before param access."""
    
    if not file_path.exists():
        print(f"  ⚠️  File not found: {file_path.name}")
        return False
    
    # Create backup
    backup = file_path.with_suffix(file_path.suffix + ".backup")
    if not backup.exists():
        backup.write_text(file_path.read_text())
        print(f"  ✓ Backup created: {file_path.name}.backup")
    
    content = file_path.read_text()
    
    # Check if already patched
    if GUARD_COMMENT in content:
        print(f"  ⊙ Already patched: {file_path.name}")
        return False
    
    # Find all 'param = params_dict[name]' lines
    lines = content.split('\n')
    new_lines = []
    patched_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line has 'param = params_dict[name]'
        if re.search(r'\bparam\s*=\s*params_dict\[name\]', line):
            # Get indentation
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent
            
            # Check if already guarded (look at previous lines)
            already_guarded = False
            for j in range(max(0, i-10), i):
                if 'if name not in params_dict' in lines[j]:
                    already_guarded = True
                    break
            
            if not already_guarded:
                # Add guard before this line
                new_lines.append(f"{spaces}{GUARD_COMMENT}")
                new_lines.append(f"{spaces}if name not in params_dict:")
                new_lines.append(f"{spaces}    continue")
                patched_count += 1
        
        new_lines.append(line)
        i += 1
    
    if patched_count > 0:
        file_path.write_text('\n'.join(new_lines))
        print(f"  ✓ Patched: {file_path.name} ({patched_count} guards added)")
        return True
    else:
        print(f"  ⊙ No changes needed: {file_path.name}")
        return False

def main():
    vllm_path = Path(VLLM_SOURCE)
    
    if not vllm_path.exists():
        print(f"❌ vLLM source not found at: {VLLM_SOURCE}")
        print("Set VLLM_SOURCE environment variable or run from project root")
        sys.exit(1)
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Patching vLLM models for num_hidden_layers compatibility ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    patched_files = []
    
    for model_file in MODEL_FILES:
        file_path = vllm_path / model_file
        print(f"Processing: {file_path.name}")
        
        if patch_file(file_path):
            patched_files.append(file_path.name)
    
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  ✅ Patching complete!                                     ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    if patched_files:
        print("Patched files:")
        for fname in patched_files:
            print(f"  • {fname}")
    else:
        print("All files were already patched or no changes needed.")
    
    print()
    print("These models now support --num-hidden-layers override:")
    print("  • Llama (all variants)")
    print("  • Qwen2")
    print("  • Qwen2-MoE")
    print("  • DeepSeek-V2")
    print()
    print("To restore original files:")
    print(f"  cd {vllm_path}/vllm/model_executor/models/")
    print("  for f in *.backup; do mv \"$f\" \"${f%.backup}\"; done")

if __name__ == "__main__":
    main()