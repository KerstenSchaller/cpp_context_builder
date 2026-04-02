from clang.cindex import *
import clang
Config.set_library_file("/usr/lib/llvm-14/lib/libclang.so")

index = Index.create()
tu = index.parse("test.cpp")

# === Parse a C/C++ file ===
index = Index.create()
tu = index.parse("test.cpp")  # Replace with your file

IGNORE_KINDS = {
    CursorKind.COMPOUND_STMT,
}


# === Helper function to print AST recursively ===
def print_ast(cursor, indent="", last=True):
    """Prints the AST with tree structure and useful info."""
    marker = "└─ " if last else "├─ "
    node_info = f"{cursor.is_definition() and 'Def ' or 'Decl '}{cursor.kind.name} {cursor.spelling}"
    if cursor.type.kind != clang.cindex.TypeKind.INVALID:
        node_info += f" : {cursor.type.spelling}"
    if cursor.location.file:
        node_info += f" ({cursor.location.file}:{cursor.location.line})"
    
    if cursor.kind not in IGNORE_KINDS:
        print(indent + marker + node_info)
    
    children = list(cursor.get_children())
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        print_ast(child, indent + ("   " if last else "│  "), is_last)

# === Print the full AST ===
print("=== Full AST ===")
print_ast(tu.cursor)

