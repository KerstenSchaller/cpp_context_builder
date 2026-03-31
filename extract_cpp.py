import os
import shlex
import sys
import json
from clang.cindex import Index, CursorKind, Config

# Adjust this path if needed
Config.set_library_file("/usr/lib/llvm-14/lib/libclang.so")

from clang.cindex import TranslationUnit





def clean_args(args, source_file=None, directory=None):
    filtered = []

    # Flags that take a following value we should skip entirely.
    skip_with_value = {
        "-o",
        "-MF",
        "-MT",
        "-MQ",
        "-MJ",
    }

    skip_prefixes = [
        "-Wl,",        # linker-only options
        "-Winvalid",   # warning policy; not needed for AST extraction
        "-Werror",
        "-O",          # optimization flags not needed
    ]

    skip_exact = {
        "-c",
    }

    source_file_abs = os.path.abspath(source_file) if source_file else None
    i = 0
    while i < len(args):
        arg = args[i]

        if arg in skip_with_value:
            i += 2
            continue

        if arg in skip_exact or any(arg.startswith(p) for p in skip_prefixes):
            i += 1
            continue

        # Drop standalone source path arguments from the compile database command.
        # The translation unit path is already provided directly to libclang.
        candidate = arg
        if not os.path.isabs(candidate) and directory:
            candidate = os.path.join(directory, candidate)
        if source_file_abs and os.path.abspath(candidate) == source_file_abs:
            i += 1
            continue

        filtered.append(arg)
        i += 1

    return filtered

def get_compile_args(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "..", "buildCustom", "compile_commands.json")

    with open(db_path) as f:
        db = json.load(f)

    input_path = os.path.abspath(file_path)

    print(f"Looking for compile args for {input_path}")

    for entry in db:
        entry_path = os.path.abspath(entry["file"])

        if entry_path == input_path:
            print(f"✅ Found compile command for {entry_path}")

            entry_dir = entry.get("directory")
            if "arguments" in entry and entry["arguments"]:
                parts = entry["arguments"]
            else:
                parts = shlex.split(entry["command"])

            args = parts[1:]  # remove compiler executable
            args = clean_args(args, source_file=input_path, directory=entry_dir)

            return args

    print("❌ No compile args found, using default")
    return ["-std=c++17"]




index = Index.create()


def get_cursor_kind_safe(cursor):
    try:
        return cursor.kind
    except ValueError:
        return None

def extract(cursor):
    results = []

    for c in cursor.get_children():
        kind = get_cursor_kind_safe(c)
        if kind is None:
            continue

        if kind == CursorKind.CLASS_DECL or kind == CursorKind.STRUCT_DECL:
            results.append({
                "type": "class",
                "name": c.spelling,
                "methods": extract_methods(c)
            })

        results.extend(extract(c))

    return results

def extract_methods(class_cursor):
    methods = []
    for c in class_cursor.get_children():
        kind = get_cursor_kind_safe(c)
        if kind == CursorKind.CXX_METHOD:
            methods.append({
                "name": c.spelling,
                "signature": c.displayname,
                "return_type": c.result_type.spelling
            })
    return methods



def parse_file(file_path, compile_args):
    file_path = os.path.abspath(file_path)

    try:
        tu = index.parse(
            file_path,
            args=compile_args,
            options=TranslationUnit.PARSE_INCOMPLETE,
        )
    except Exception as e:
        print("❌ Parse failed with full args")
        print(e)
        print("⚠️ Retrying with minimal args...")
        fallback_args = ["-std=c++20", "-xc++"]
        tu = index.parse(
            file_path,
            args=fallback_args,
            options=TranslationUnit.PARSE_INCOMPLETE,
        )

    for diag in tu.diagnostics:
        print("DIAG:", diag)

    return extract(tu.cursor)


if __name__ == "__main__":
    file_path = os.path.abspath(sys.argv[1])

    # Minimal args (we improve later)
    compile_args = get_compile_args(file_path)
    compile_args.append("-fparse-all-comments")
    compile_args.append("-xc++")

    print("ARGS:", compile_args)

    data = parse_file(file_path, compile_args)
    print(json.dumps(data, indent=2))
