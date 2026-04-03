
import argparse
import json
import os
import shlex
from clang.cindex import *
import clang
Config.set_library_file("/usr/lib/llvm-14/lib/libclang.so")

filter_strings = [
		'depends',
        'usr/include',
        'include/c++',
        '_deps',
        'x86_64-linux-gnu',
        'fmt-src',
        'stdio.h'
]

number_of_files_parsed = 0
number_of_files_errors = 0
number_of_classes = 0
number_of_structs = 0
number_of_enums = 0
number_of_functions = 0

# Cursor kinds that introduce a function/method *definition* we want to track.
DEFINITION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.CONVERSION_FUNCTION,
}

# Kinds for classes, structs, enums
CLASS_KINDS = {CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL}
ENUM_KINDS = {CursorKind.ENUM_DECL}

def getCursorKey(cursor):
    """Generate a unique key for a cursor based on its location and spelling."""
    if cursor.location.file:
        return f"{cursor.location.file}:{cursor.location.line}:{cursor.spelling}"
    return cursor.spelling

def logToFile(message, path = "AST_ERROR.log"):
    with open(path, "a") as log_file:
        log_file.write(message + "\n")

def _kind_safe(cursor):
    try:
        return cursor.kind
    except ValueError:
        return None

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

def getSource(cursor):
    """Get the source code for a cursor"""
    if cursor.location.file:
        with open(cursor.location.file.name, 'r') as f:
            lines = f.readlines()
            # Extract lines from start to end of the declaration
            start_line = cursor.extent.start.line - 1  # zero-based index
            end_line = cursor.extent.end.line
            return ''.join(lines[start_line:end_line])
    return None

def getFields(cursor):
    """Get the fields of a class/struct cursor"""
    fields = []
    for child in cursor.get_children():
        if child.kind == CursorKind.FIELD_DECL:
            fields.append(f"{child.type.spelling} {child.spelling}")
    return fields


def getNamespaces(cursor):
    """Get the namespaces of a cursor"""
    namespaces = []
    parent = cursor.semantic_parent
    while parent and parent.kind != CursorKind.TRANSLATION_UNIT:
        if parent.kind == CursorKind.NAMESPACE:
            namespaces.append(parent.spelling)
        parent = parent.semantic_parent
    return '::'.join(reversed(namespaces))



def getMethods(cursor):
    """Get the methods of a class/struct cursor"""
    methods = []
    for child in cursor.get_children():
        if child.kind in DEFINITION_KINDS:
            methodStr = f"{child.type.spelling} "
            methodStr = methodStr.replace(" ", f" {child.spelling}", 1)
            methods.append(f"{methodStr}")
    return methods

def parseClass(cursor):
    """Parse a class/struct declaration and print its fields"""
    global number_of_classes, number_of_structs
    number_of_classes += 1 if cursor.kind == CursorKind.CLASS_DECL else 0
    number_of_structs += 1 if cursor.kind == CursorKind.STRUCT_DECL else 0
    typeStr = cursor.kind is CursorKind.CLASS_DECL and "class" or "struct"
    return {
        "type": typeStr,
        "symbol": qualified_name(cursor),
        "comment": "Implementation details of a {}".format(typeStr),
        "location": f"{cursor.location.file}:{cursor.location.line}" if cursor.location.file else "unknown",
        "namespaces": getNamespaces(cursor),
        "name": cursor.spelling,
        "fields": getFields(cursor),
        "methods": getMethods(cursor),
        "source": getSource(cursor)
    }

def parseEnum(cursor):
    """Parse an enum declaration and return its info as a dict."""
    global number_of_enums
    number_of_enums += 1
    enum_constants = []
    for child in cursor.get_children():
        if child.kind == CursorKind.ENUM_CONSTANT_DECL:
            enum_constants.append(child.spelling + (f" = {child.enum_value}" if child.enum_value is not None else ""))
    return {
        "type": "enum",
        "symbol": qualified_name(cursor),
        "comment": "Implementation details of an enum",
        "location": f"{cursor.location.file}:{cursor.location.line}" if cursor.location.file else "unknown",
        "namespaces": getNamespaces(cursor),
        "name": cursor.spelling,
        "constants": enum_constants,
        "source": getSource(cursor)
    }

def getFunctionCallNamespaces(cursor, isConstructorCall):
    """Get the namespaces of a function call cursor"""
    # regular functions
    recurse = (lambda self, cur: sum([
        [child.spelling] if child.kind == CursorKind.NAMESPACE_REF else self(self, child)
        for child in cur.get_children()
    ], []))
    namespaces = recurse(recurse, cursor)
    # class member functions have a child MEMBER_REF_EXPR which has a child DECL_REF_EXPR
    if namespaces == []:
        for child in cursor.get_children():
            if child.kind == CursorKind.MEMBER_REF_EXPR:
                for grandchild in child.get_children():
                    if grandchild.kind == CursorKind.DECL_REF_EXPR:
                        namespaces.append(grandchild.type.spelling)
    # constructor calls are TYPE_REF followed by CALL_EXPR
    if namespaces == []:
        if isConstructorCall:
            namespaces.append(cursor.type.spelling)
    return '::'.join(reversed(namespaces))    

def filterFunctionCalls(calls):
    """Filter function calls to remove duplicates and sort them."""
    unique_calls = list(set(calls))
    unique_calls.sort()
    functionFilterString = ["operator",
                            "std::", 
                            "fmt::", 
                            "spdlog::", 
                            "boost::", 
                            "catch::",
                            "memset"]

    return [call for call in unique_calls if not any(f in call for f in functionFilterString)]

def getFunctionCalls(cursor):
    """Get the function calls made within a function/method cursor."""
    calls = []
    children = list(cursor.get_children())
    for i in range(len(children)):
        child = children[i]
        if child.kind == CursorKind.CALL_EXPR:
            called_func = child.displayname.split('(')[0]  # Get function name without arguments
            if children[i - 1].kind == CursorKind.TYPE_REF:
                namespaces = getFunctionCallNamespaces(child, True)
            else:
                namespaces = getFunctionCallNamespaces(child, False)
            if namespaces != "":
                called_func = f"{namespaces}::{called_func}"
            calls.append(called_func)
        calls.extend(getFunctionCalls(child))  # Recurse into children
    # make calls unique
    calls = filterFunctionCalls(calls)
    return calls

def parseFunction(cursor):
    """Parse a function/method declaration and return its info as a dict."""
    global number_of_functions
    number_of_functions += 1
    return {
        "type": "function",
        "symbol": qualified_name(cursor),
        "comment": "Implementation details of a function",
        "location": f"{cursor.location.file}:{cursor.location.line}" if cursor.location.file else "unknown",
        "namespaces": getNamespaces(cursor),
        "name": cursor.spelling,
        "return_type": cursor.result_type.spelling if hasattr(cursor, 'result_type') else None,
        "arguments": [f"{arg.type.spelling} {arg.spelling}" for arg in cursor.get_arguments()] if hasattr(cursor, 'get_arguments') else [],
        "calls": getFunctionCalls(cursor),
        "source": getSource(cursor)
    }

def printDictRecursively(d, indent=0):
    """Helper function to print a dictionary recursively with indentation."""
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            printDictRecursively(value, indent + 4)
        elif isinstance(value, list):
            print(" " * indent + f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    printDictRecursively(item, indent + 4)
                else:
                    print(" " * (indent + 4) + str(item))
        else:
            print(" " * indent + f"{key}: {value}")

def handle_class(cursor):
    classInfo = parseClass(cursor)
    printDictRecursively(classInfo)
    
def handle_enum(cursor):
    enumInfo = parseEnum(cursor)
    printDictRecursively(enumInfo)

def handle_function(cursor):
    functionInfo = parseFunction(cursor)
    printDictRecursively(functionInfo)



def isLocationFiltered(cursor):
    if cursor.location.file:
        for filter_str in filter_strings:
            if filter_str in cursor.location.file.name:
                return True
    return False

def print_ast(cursor, indent="", last=True, canonical = False):
    """Prints the AST with tree structure and useful info."""
    if not hasattr(print_ast, "parsed_keys"):
        print_ast.parsed_keys = set()

    if canonical:
        cursor = cursor.canonical
    marker = "└─ " if last else "├─ "
    node_info = f"{cursor.is_definition() and 'Def ' or 'Decl '}{cursor.kind.name} {cursor.spelling}"
    if cursor.type.kind != clang.cindex.TypeKind.INVALID:
        node_info += f" : {cursor.type.spelling}"
    if cursor.location.file:
        node_info += f" ({cursor.location.file}:{cursor.location.line})"
    
    IGNORE_KINDS = {
        CursorKind.COMPOUND_STMT,
    }

    if cursor.kind not in IGNORE_KINDS:
        print(indent + marker + node_info)
    
    children = list(cursor.get_children())
    for i, child in enumerate(children):
        key = getCursorKey(child)
        if key in print_ast.parsed_keys:
            continue
        print_ast.parsed_keys.add(key)
        if isLocationFiltered(child):
            continue
        is_last = i == len(children) - 1
        print_ast(child, indent + ("   " if last else "│  "), is_last, canonical)  

def walk_ast( cursor):

    # Use a static attribute to persist across recursive calls
    if not hasattr(walk_ast, "parsed_keys"):
        walk_ast.parsed_keys = set()

    for child in cursor.get_children():
        key = getCursorKey(child)
        if key in walk_ast.parsed_keys:
            continue
        walk_ast.parsed_keys.add(key)
        if isLocationFiltered(child):
            continue
        if child.kind in CLASS_KINDS:
            handle_class(child)
        elif child.kind in ENUM_KINDS:
            handle_enum(child)
        elif child.kind in DEFINITION_KINDS:
            handle_function(child)
        walk_ast(child)




def parse_args():
    parser = argparse.ArgumentParser(description="Walk C++ AST with Clang Python bindings.")
    parser.add_argument('-c', '--compile-commands', type=str, help='Path to compile_commands.json')
    parser.add_argument('-tf', '--testFile', type=str, help='C++ source file to parse if not using compile_commands.json')
    parser.add_argument('-sf', '--specFile', type=str, help='If set, only parse this file from compile_commands.json')
    parser.add_argument('-pa', '--printAST', action='store_true', help='If set, print the AST using print_ast instead of walk_ast')
    return parser.parse_args()

def get_sources_from_compile_commands(path):
    with open(path, 'r') as f:
        data = json.load(f)
    sources = [entry['file'] for entry in data if os.path.isfile(entry['file'])]
    return sources

def filter_source(source):
    # return true if any of the filter strings are in the source path
    for str in filter_strings:
        if str in source:
            return True
    return False




def clean_args(args, source_file=None, directory=None):
    filtered = []
    SKIP_WITH_VALUE = {"-o", "-MF", "-MT", "-MQ", "-MJ"}
    SKIP_PREFIXES   = ["-Wl,", "-Winvalid", "-Werror", "-O"]
    SKIP_EXACT      = {"-c"}
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

def get_compile_args(file_path, db_path):
    if db_path is None:
        return ["-std=c++17"]
    with open(db_path) as f:
        db = json.load(f)

    input_path = os.path.abspath(file_path)
    for entry in db:
        if os.path.abspath(entry["file"]) == input_path:
            entry_dir = entry.get("directory")
            parts = entry["arguments"] if entry.get("arguments") else shlex.split(entry["command"])
            args = clean_args(parts[1:], source_file=input_path, directory=entry_dir)
            return args
    return ["-std=c++17"]

def parseFiles(file_path, compile_args):
    global number_of_files_parsed
    if filter_source(file_path):
        return
    print(f"##### Parsing file: {file_path} #####")
    _index = Index.create()

    file_path = os.path.abspath(file_path)

    extra = ["-fparse-all-comments", "-xc++"] 
    args = compile_args + extra
    try:
        tu = _index.parse(
            file_path,
            args=args,
            options=TranslationUnit.PARSE_INCOMPLETE,
        )
        if getattr(parseFiles, 'print_ast_mode', False):
            print_ast(tu.cursor)
        else:
            walk_ast(tu.cursor)
        number_of_files_parsed += 1
    except Exception as e:
        print(f"   Parse failed ({e}), retrying with minimal args…")
        tu = _index.parse(
            file_path,
            args=["-std=c++20", "-xc++"],
            options=TranslationUnit.PARSE_INCOMPLETE,
        )
        if getattr(parseFiles, 'print_ast_mode', False):
            print_ast(tu.cursor)
        else:
            walk_ast(tu.cursor)
            number_of_files_parsed += 1

            fatal = [d for d in tu.diagnostics if d.severity >= 3]
            if fatal:
                print(f"  ⚠️  {len(fatal)} fatal diagnostic(s) in {os.path.basename(file_path)}")
                print("    " + "\n    ".join(str(d) for d in fatal))
                logToFile(f"Fatal diagnostics in {file_path}:\n" + "\n".join(str(d) for d in fatal))
                global number_of_files_errors
                number_of_files_errors += 1

def main():

    import time
    start_time = time.time()

    # Reset parsed_keys before each run
    if hasattr(walk_ast, "parsed_keys"):
        walk_ast.parsed_keys.clear()

    args = parse_args()
    # Consistency checks
    if args.compile_commands and not os.path.isfile(args.compile_commands):
        exit(1)
    if args.compile_commands is None and args.testFile is None:
        print("Error: Either --compile-commands (-c) or --testFile (-tf) must be provided.", flush=True)
        exit(1)
    if args.specFile and not args.compile_commands:
        print("Error: --specFile (-sf) can only be used with --compile-commands (-c).", flush=True)
        exit(1)


    # Set print_ast_mode flag for parseFiles
    parseFiles.print_ast_mode = getattr(args, 'printAST', False)


    # get sources to parse (testfile or compile_commands.json)
    if args.testFile:
        sources = [args.testFile]
        compile_args = get_compile_args(sources[0],args.compile_commands)
        parseFiles(sources[0], compile_args)
    elif args.compile_commands:
        sources = get_sources_from_compile_commands(args.compile_commands)
        if args.specFile:
            # Only parse the specified file if present in sources
            spec_abs = os.path.abspath(args.specFile)
            print(f"Filtering sources to only include: {spec_abs}")
            sources = [src for src in sources if args.specFile in os.path.abspath(src) ]
        for src in sources:
            compile_args = get_compile_args(src,args.compile_commands)
            parseFiles(src, compile_args)

    end_time = time.time()
    print(f"Parsed {number_of_files_parsed} files")
    print(f"   Execution time: {end_time - start_time:.3f} seconds")
    print(f"   Execution time: {(end_time - start_time)/60:.3f} minutes")
    print(f"   Found {number_of_classes} classes")
    print(f"   Found {number_of_structs} structs")
    print(f"   Found {number_of_enums} enums")
    print(f"   Found {number_of_functions} functions")

# Only run main if this script is executed directly
if __name__ == "__main__":
    main()