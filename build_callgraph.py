"""
build_callgraph.py

Builds a call-graph for a C++ project by traversing libclang ASTs.

For every function / method *definition* found in the requested source files
it records which other functions it calls, resolved to their canonical
qualified name (ClassName::method or ::free_function).

Usage
-----
    # single file
    python build_callgraph.py path/to/foo.cpp

    # whole project  (walks the compile_commands.json entries)
    python build_callgraph.py --all

Output
------
A JSON file (callgraph.json by default) with two top-level keys:

    {
      "nodes": [
        { "id": "Namespace::Class::method(args)",
          "file": "relative/path.cpp",
          "line": 42 }
      ],
      "edges": [
        { "caller": "Namespace::Class::method(args)",
          "callee": "OtherClass::helper()" }
      ]
    }

The graph can be loaded into networkx, Graphviz, or a vector-DB pipeline.
"""

import os
import sys
import json
import shlex
import argparse
from clang.cindex import Index, CursorKind, Config, TranslationUnit

# ── libclang library path ────────────────────────────────────────────────────
Config.set_library_file("/usr/lib/llvm-14/lib/libclang.so")

# ── compile-commands helpers (reused from extract_cpp.py) ────────────────────

SKIP_WITH_VALUE = {"-o", "-MF", "-MT", "-MQ", "-MJ"}
SKIP_PREFIXES   = ["-Wl,", "-Winvalid", "-Werror", "-O"]
SKIP_EXACT      = {"-c"}


def clean_args(args, source_file=None, directory=None):
    filtered = []
    source_file_abs = os.path.abspath(source_file) if source_file else None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in SKIP_WITH_VALUE:
            i += 2
            continue
        if arg in SKIP_EXACT or any(arg.startswith(p) for p in SKIP_PREFIXES):
            i += 1
            continue
        candidate = arg
        if not os.path.isabs(candidate) and directory:
            candidate = os.path.join(directory, candidate)
        if source_file_abs and os.path.abspath(candidate) == source_file_abs:
            i += 1
            continue
        filtered.append(arg)
        i += 1
    return filtered


def _load_compile_db():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "..", "buildX", "compile_commands.json")
    with open(db_path) as f:
        return json.load(f)


def get_compile_args(file_path, db=None):
    if db is None:
        db = _load_compile_db()
    input_path = os.path.abspath(file_path)
    for entry in db:
        if os.path.abspath(entry["file"]) == input_path:
            entry_dir = entry.get("directory")
            parts = entry["arguments"] if entry.get("arguments") else shlex.split(entry["command"])
            args = clean_args(parts[1:], source_file=input_path, directory=entry_dir)
            return args
    return ["-std=c++17"]


# ── AST helpers ──────────────────────────────────────────────────────────────

def _kind_safe(cursor):
    try:
        return cursor.kind
    except ValueError:
        return None


# Cursor kinds that introduce a function/method *definition* we want to track.
DEFINITION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.CONVERSION_FUNCTION,
}


def qualified_name(cursor):
    """Return a fully-qualified name like Namespace::Class::method(args)."""
    parts = []
    c = cursor
    while c and c.kind not in (CursorKind.TRANSLATION_UNIT,):
        spelling = c.displayname or c.spelling
        if spelling:
            parts.append(spelling)
        c = c.semantic_parent
        if c is None or _kind_safe(c) is None:
            break
    parts.reverse()
    return "::".join(parts) if parts else cursor.displayname


def _collect_calls(cursor, source_file_abs, calls_set):
    """Recursively collect all CALL_EXPR cursors under *cursor*."""
    for child in cursor.get_children():
        kind = _kind_safe(child)
        if kind is None:
            continue
        if kind == CursorKind.CALL_EXPR:
            ref = child.referenced
            if ref is not None and _kind_safe(ref) is not None:
                # Only record calls where we can resolve the callee
                callee = qualified_name(ref)
                if callee:
                    calls_set.add(callee)
        _collect_calls(child, source_file_abs, calls_set)


def extract_callgraph(tu, source_file_abs):
    """
    Walk the translation unit and return (nodes, edges) for *source_file_abs*.

    nodes: list of dicts  { id, file, line, type }
    edges: list of dicts  { caller, callee }
    """
    nodes = []
    edges = []
    seen_node_ids = set()

    # Kinds for classes, structs, enums
    CLASS_KINDS = {CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL}
    ENUM_KINDS = {CursorKind.ENUM_DECL}

    def walk(cursor):
        kind = _kind_safe(cursor)
        if kind is None:
            return

        loc = cursor.location


        # Functions/methods (with call edges)
        if kind in DEFINITION_KINDS and cursor.is_definition():
            node_id = qualified_name(cursor)
            if node_id and node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                rel_path = os.path.relpath(loc.file.name)
                nodes.append({
                    "id":   node_id,
                    "file": rel_path.replace("\\", "/"),
                    "line": loc.line,
                    "type": "function"
                })
                calls = set()
                _collect_calls(cursor, source_file_abs, calls)
                for callee_id in sorted(calls):
                    edges.append({"caller": node_id, "callee": callee_id})

        # Classes/structs (no edges)
        elif kind in CLASS_KINDS and cursor.is_definition():
            node_id = qualified_name(cursor)
            if node_id and node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                rel_path = os.path.relpath(loc.file.name)
                nodes.append({
                    "id":   node_id,
                    "file": rel_path.replace("\\", "/"),
                    "line": loc.line,
                    "type": "class" if kind == CursorKind.CLASS_DECL else "struct"
                })

        # Enums (no edges)
        elif kind in ENUM_KINDS and cursor.is_definition():
            node_id = qualified_name(cursor)
            if node_id and node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                rel_path = os.path.relpath(loc.file.name)
                nodes.append({
                    "id":   node_id,
                    "file": rel_path.replace("\\", "/"),
                    "line": loc.line,
                    "type": "enum"
                })

        for child in cursor.get_children():
            walk(child)

    walk(tu.cursor)
    return nodes, edges


# ── parsing ──────────────────────────────────────────────────────────────────

_index = Index.create()


def parse_file(file_path, compile_args):
    file_path = os.path.abspath(file_path)
    extra = ["-fparse-all-comments", "-xc++"]
    args = compile_args + extra
    try:
        tu = _index.parse(
            file_path,
            args=args,
            options=TranslationUnit.PARSE_INCOMPLETE,
        )
    except Exception as e:
        print(f"  ⚠️  Parse failed ({e}), retrying with minimal args…")
        tu = _index.parse(
            file_path,
            args=["-std=c++20", "-xc++"],
            options=TranslationUnit.PARSE_INCOMPLETE,
        )

    fatal = [d for d in tu.diagnostics if d.severity >= 3]
    if fatal:
        print(f"  ⚠️  {len(fatal)} fatal diagnostic(s) in {os.path.basename(file_path)}")

    return extract_callgraph(tu, os.path.abspath(file_path))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build a C++ call-graph via libclang")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs="?", help="Single .cpp file to parse")
    group.add_argument("--all", action="store_true",
                       help="Parse every file listed in compile_commands.json")
    parser.add_argument("-o", "--output", default="callgraph.json",
                        help="Output JSON file (default: callgraph.json)")
    args = parser.parse_args()

    all_nodes: list = []
    all_edges: list = []
    seen_node_ids: set = set()

    if args.all:
        db = _load_compile_db()
        files = [entry["file"] for entry in db]
        print(f"Processing {len(files)} file(s) from compile_commands.json …")
    else:
        db = None
        files = [args.file]

    for fpath in files:
        fpath_abs = os.path.abspath(fpath)
        if not os.path.isfile(fpath_abs):
            print(f"  ⚠️  File not found, skipping: {fpath}")
            continue
        print(f"  Parsing {os.path.basename(fpath_abs)} …")
        cargs = get_compile_args(fpath_abs, db=db)
        nodes, edges = parse_file(fpath_abs, cargs)
        for n in nodes:
            if n["id"] not in seen_node_ids:
                seen_node_ids.add(n["id"])
                all_nodes.append(n)
        all_edges.extend(edges)

    # Deduplicate edges
    edge_set = {(e["caller"], e["callee"]) for e in all_edges}
    deduped_edges = [{"caller": c, "callee": x} for c, x in sorted(edge_set)]

    result = {"nodes": all_nodes, "edges": deduped_edges}

    output_path = os.path.abspath(args.output)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅  Wrote {len(all_nodes)} nodes, {len(deduped_edges)} edges → {output_path}")


if __name__ == "__main__":
    main()
