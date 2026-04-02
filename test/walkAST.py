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

def parseEnum(cursor):
    """Parse an enum declaration and return its info as a dict."""
    enum_constants = []
    for child in cursor.get_children():
        if child.kind == CursorKind.ENUM_CONSTANT_DECL:
            enum_constants.append(child.spelling + (f" = {child.enum_value}" if child.enum_value is not None else ""))
    return {
        "type": "enum",
        "comment": "Implementation info of an enum",
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
    return calls

def parseFunction(cursor):
    """Parse a function/method declaration and return its info as a dict."""
    return {
        "type": "function",
        "comment": "Implementation info of a function",
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





def walk_ast(cursor):
    for child in cursor.get_children():
        if child.kind in CLASS_KINDS:
            handle_class(child)
        elif child.kind in ENUM_KINDS:
            handle_enum(child)
        elif child.kind in DEFINITION_KINDS:
            handle_function(child)
        
        walk_ast(child)

# === Parse a C/C++ file ===
index = Index.create()
tu = index.parse("test.cpp")  # Replace with your file

# Start walking from the translation unit root
walk_ast(tu.cursor)

