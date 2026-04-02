from clang.cindex import *
import clang
Config.set_library_file("/usr/lib/llvm-14/lib/libclang.so")

index = Index.create()
tu = index.parse("test.cpp")

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

def getMethods(cursor):
    """Get the methods of a class/struct cursor"""
    methods = []
    for child in cursor.get_children():
        if child.kind in DEFINITION_KINDS:
            methodStr = f"{child.type.spelling} "
            methodStr = methodStr.replace(" ", f" {child.spelling}", 1)
            methods.append(f"{methodStr}")
    return methods

def getNamespaces(cursor):
    """Get the namespaces of a cursor"""
    namespaces = []
    parent = cursor.semantic_parent
    while parent and parent.kind != CursorKind.TRANSLATION_UNIT:
        if parent.kind == CursorKind.NAMESPACE:
            namespaces.append(parent.spelling)
        parent = parent.semantic_parent
    return '::'.join(reversed(namespaces))

def parseClass(cursor):
    """Parse a class/struct declaration and print its fields"""
    typeStr = cursor.kind is CursorKind.CLASS_DECL and "class" or "struct"
    print(f"typeStr {typeStr}")
    return {
        "type": typeStr,
        "comment": "Implementation info of a {}".format(typeStr),
        "location": f"{cursor.location.file}:{cursor.location.line}" if cursor.location.file else "unknown",
        "namespaces": getNamespaces(cursor),
        "name": cursor.spelling,
        "fields": getFields(cursor),
        "methods": getMethods(cursor),
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
    
  



def walk_ast(cursor):
    for child in cursor.get_children():
        match child.kind:
            case CursorKind.CLASS_DECL | CursorKind.STRUCT_DECL:
                handle_class(child)
            case _:
                walk_ast(child)

# === Parse a C/C++ file ===
index = Index.create()
tu = index.parse("test.cpp")  # Replace with your file

# Start walking from the translation unit root
walk_ast(tu.cursor)

