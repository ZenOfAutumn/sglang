#!/usr/bin/env python3
import subprocess
import os

REPO_DIR = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True, text=True
).stdout.strip()

def get_conflicted_files():
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        capture_output=True, text=True, cwd=REPO_DIR
    )
    return [f for f in result.stdout.strip().split("\n") if f]

def is_zh_comment(line):
    s = line.strip()
    return s.startswith("#") and ("中译" in s)

def resolve_block(head_lines, main_lines):
    if head_lines == main_lines:
        return head_lines
    head_cmts = [l for l in head_lines if is_zh_comment(l)]
    if not head_cmts:
        # No Chinese comments in HEAD, just use main
        return main_lines
    head_code = [l for l in head_lines if not is_zh_comment(l)]
    if not head_code:
        # HEAD is only comments, keep them + main
        result = head_cmts + main_lines
    else:
        result = head_cmts + main_lines
    seen = set()
    deduped = []
    for line in result:
        key = line.strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    while deduped and deduped[-1].strip() == "":
        deduped.pop()
    return deduped

def resolve_file(filepath):
    abs_path = os.path.join(REPO_DIR, filepath)
    try:
        with open(abs_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return False
    lines = content.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("<<<<<<< HEAD"):
            j = i + 1
            sep = None
            while j < len(lines) and not lines[j].startswith(">>>>>>> "):
                if lines[j].startswith("======="):
                    sep = j
                j += 1
            if sep is None:
                result.append(line)
                i += 1
                continue
            head = lines[i+1:sep]
            main = lines[sep+1:j]
            resolved = resolve_block(head, main)
            result.extend(resolved)
            i = j + 1
        else:
            result.append(line)
            i += 1
    new_content = "\n".join(result)
    if new_content != content:
        with open(abs_path, "w") as f:
            f.write(new_content)
        return True
    return False

def main():
    files = get_conflicted_files()
    print(f"Found {len(files)} conflicted files")
    ok = 0
    for f in files:
        if resolve_file(f):
            ok += 1
            print(f"  Resolved: {f}")
    print(f"\nResolved {ok}/{len(files)} files")

if __name__ == "__main__":
    main()

