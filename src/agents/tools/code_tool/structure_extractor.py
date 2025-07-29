"""Tool for extracting code structures from Go and Python files using AST and Tree-sitter."""

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from pydantic_ai.tools import Tool
from tree_sitter import Parser, Language
import tree_sitter_go

class CodeStructure(BaseModel):
    """Represents a code structure (function, class, struct, etc.)."""
    name: str = Field(..., description="Name of the structure")
    type: str = Field(..., description="Type: function, class, struct, interface, method")
    signature: str = Field(..., description="Function/method signature or struct/class definition")
    documentation: Optional[str] = Field(None, description="Documentation/docstring/comments")
    line_number: int = Field(..., description="Line number where structure is defined")
    end_line_number: Optional[int] = Field(None, description="End line number where structure ends")
    file_path: str = Field(..., description="File path where structure is defined")
    parameters: Optional[List[str]] = Field(None, description="Parameters for functions/methods")
    return_type: Optional[str] = Field(None, description="Return type for functions/methods")
    fields: Optional[List[str]] = Field(None, description="Fields for structs/classes")
    complexity_indicators: Optional[Dict[str, int]] = Field(None, description="Code complexity metrics")
    decorators: Optional[List[str]] = Field(None, description="Decorators applied to the structure")
    is_async: bool = Field(default=False, description="Whether the function/method is async")
    nested_structures: Optional[List[str]] = Field(None, description="Names of nested functions/classes")


class StructureExtractorTool:
    """Tool for extracting code structures from source files."""

    def get_tool(self) -> Tool:
        """Get the structure extractor tool."""
        return Tool(
            name="extract_code_structures",
            description="Extract functions, classes, structs, interfaces and their documentation from Go or Python files",
            function=self._extract_structures,
        )

    def _extract_structures(self, file_path: str) -> List[CodeStructure]:
        """
        Extract code structures from a file.
        
        Args:
            file_path: Path to the Go or Python file to analyze
            
        Returns:
            List of CodeStructure objects with extracted information
        """
        try:
            if not os.path.exists(file_path):
                return []
                
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == ".py":
                return self._extract_python_structures(file_path)
            elif file_extension == ".go":
                return self._extract_go_structures(file_path)
            else:
                return []
                
        except Exception as e:
            print(f"Error extracting structures from {file_path}: {e}")
            return []

    def _extract_python_structures(self, file_path: str) -> List[CodeStructure]:
        """Extract structures from Python files using AST with advanced analysis."""
        structures = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            tree = ast.parse(content)
            lines = content.splitlines()
            
            # Extract module-level docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                structures.append(CodeStructure(
                    name="__module__",
                    type="module",
                    signature=f"module {Path(file_path).stem}",
                    documentation=module_doc,
                    line_number=1,
                    file_path=file_path
                ))
            
            # Use a visitor to analyze the AST tree systematically
            visitor = PythonStructureVisitor(file_path, lines)
            visitor.visit(tree)
            structures.extend(visitor.structures)
                    
        except Exception as e:
            print(f"Error parsing Python file {file_path}: {e}")
            
        return structures

    def _extract_go_structures(self, file_path: str) -> List[CodeStructure]:
        """Extract structures from Go files using Tree-sitter for semantic parsing."""
        structures = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Initialize Go parser using tree-sitter.
            parser = Parser(language=Language(tree_sitter_go.language()))
            tree = parser.parse(content.encode('utf-8'))

            # Use visitor to extract structures.
            visitor = GoStructureVisitor(file_path, content)
            visitor.visit(tree.root_node)
            structures.extend(visitor.structures)

        except Exception as e:
            print(f"Error parsing Go file with Tree-sitter {file_path}: {e}")
            raise e
        return structures

class PythonStructureVisitor(ast.NodeVisitor):
    """AST visitor for extracting Python code structures with advanced analysis."""
    
    def __init__(self, file_path: str, lines: List[str]):
        self.file_path = file_path
        self.lines = lines
        self.structures = []
        self.current_class = None
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        
        # Extract class information
        structure = self._extract_class_structure(node)
        self.structures.append(structure)
        
        # Continue visiting child nodes (methods, nested classes)
        self.generic_visit(node)
        
        self.current_class = old_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions."""
        structure = self._extract_function_structure(node, is_async=False)
        self.structures.append(structure)
        
        # Don't visit nested functions in this implementation
        # Could be extended to handle nested functions
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definitions."""
        structure = self._extract_function_structure(node, is_async=True)
        self.structures.append(structure)
        
    def _extract_class_structure(self, node: ast.ClassDef) -> CodeStructure:
        """Extract detailed class structure information."""
        # Build inheritance signature
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            else:
                bases.append(ast.unparse(base))
                
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"
            
        # Extract decorators
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        if decorators:
            decorator_str = "\n".join(f"@{dec}" for dec in decorators)
            signature = f"{decorator_str}\n{signature}"
            
        # Extract methods and properties
        methods = []
        nested_classes = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
            elif isinstance(item, ast.ClassDef):
                nested_classes.append(item.name)
                
        # Calculate complexity indicators
        complexity = self._calculate_class_complexity(node)
        
        return CodeStructure(
            name=node.name,
            type="class",
            signature=signature,
            documentation=ast.get_docstring(node),
            line_number=node.lineno,
            end_line_number=getattr(node, 'end_lineno', None),
            file_path=self.file_path,
            fields=methods,
            decorators=decorators,
            complexity_indicators=complexity,
            nested_structures=nested_classes
        )
        
    def _extract_function_structure(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool = False) -> CodeStructure:
        """Extract detailed function structure information."""
        # Build comprehensive signature
        signature = self._build_function_signature(node, is_async)
        
        # Extract decorators
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        
        # Extract parameters with full information
        parameters = self._extract_function_parameters(node.args)
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
            
        # Calculate complexity indicators
        complexity = self._calculate_function_complexity(node)
        
        # Determine function type (function vs method)
        func_type = "method" if self.current_class else ("async_function" if is_async else "function")
        
        return CodeStructure(
            name=node.name,
            type=func_type,
            signature=signature,
            documentation=ast.get_docstring(node),
            line_number=node.lineno,
            end_line_number=getattr(node, 'end_lineno', None),
            file_path=self.file_path,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            complexity_indicators=complexity,
            is_async=is_async
        )
        
    def _build_function_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool) -> str:
        """Build a comprehensive function signature."""
        # Build parameter list with defaults, annotations, etc.
        args = []
        
        # Position-only arguments
        for arg in node.args.posonlyargs:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
            
        if node.args.posonlyargs:
            args.append("/")
            
        # Regular arguments with defaults
        defaults_offset = len(node.args.args) - len(node.args.defaults)
        for i, arg in enumerate(node.args.args):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            # Add default value if present
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                default_val = ast.unparse(node.args.defaults[default_idx])
                arg_str += f" = {default_val}"
            args.append(arg_str)
            
        # *args
        if node.args.vararg:
            vararg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(vararg_str)
        elif node.args.kwonlyargs:
            args.append("*")
            
        # Keyword-only arguments
        for i, arg in enumerate(node.args.kwonlyargs):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            # Add default value if present
            if i < len(node.args.kw_defaults) and node.args.kw_defaults[i]:
                default_val = ast.unparse(node.args.kw_defaults[i])
                arg_str += f" = {default_val}"
            args.append(arg_str)
            
        # **kwargs
        if node.args.kwarg:
            kwarg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(kwarg_str)
            
        # Build full signature
        func_prefix = "async def" if is_async else "def"
        signature = f"{func_prefix} {node.name}({', '.join(args)})"
        
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"
            
        # Add decorators
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        if decorators:
            decorator_str = "\n".join(f"@{dec}" for dec in decorators)
            signature = f"{decorator_str}\n{signature}"
            
        return signature
        
    def _extract_function_parameters(self, args: ast.arguments) -> List[str]:
        """Extract function parameter names."""
        params = []
        
        # All argument types
        for arg in args.posonlyargs + args.args:
            params.append(arg.arg)
            
        if args.vararg:
            params.append(f"*{args.vararg.arg}")
            
        for arg in args.kwonlyargs:
            params.append(arg.arg)
            
        if args.kwarg:
            params.append(f"**{args.kwarg.arg}")
            
        return params
        
    def _calculate_function_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, int]:
        """Calculate complexity indicators for a function."""
        complexity = {
            "cyclomatic_complexity": 1,  # Base complexity
            "if_statements": 0,
            "for_loops": 0,
            "while_loops": 0,
            "try_blocks": 0,
            "with_statements": 0,
            "nested_functions": 0,
            "return_statements": 0,
            "total_lines": (getattr(node, 'end_lineno', node.lineno) - node.lineno + 1)
        }
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.IfExp)):
                complexity["if_statements"] += 1
                complexity["cyclomatic_complexity"] += 1
            elif isinstance(child, ast.For):
                complexity["for_loops"] += 1  
                complexity["cyclomatic_complexity"] += 1
            elif isinstance(child, ast.While):
                complexity["while_loops"] += 1
                complexity["cyclomatic_complexity"] += 1
            elif isinstance(child, ast.Try):
                complexity["try_blocks"] += 1
                complexity["cyclomatic_complexity"] += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity["with_statements"] += 1
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child != node:
                complexity["nested_functions"] += 1
            elif isinstance(child, ast.Return):
                complexity["return_statements"] += 1
                
        return complexity
        
    def _calculate_class_complexity(self, node: ast.ClassDef) -> Dict[str, int]:
        """Calculate complexity indicators for a class."""
        complexity = {
            "methods_count": 0,
            "properties_count": 0,
            "nested_classes": 0,
            "inheritance_depth": len(node.bases),
            "total_lines": (getattr(node, 'end_lineno', node.lineno) - node.lineno + 1)
        }
        
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity["methods_count"] += 1
                # Check if it's a property
                for decorator in child.decorator_list:
                    if (isinstance(decorator, ast.Name) and decorator.id == "property") or \
                       (isinstance(decorator, ast.Attribute) and decorator.attr == "setter"):
                        complexity["properties_count"] += 1
                        break
            elif isinstance(child, ast.ClassDef):
                complexity["nested_classes"] += 1
                
        return complexity

    def _extract_python_function(self, node: ast.FunctionDef, file_path: str) -> CodeStructure:
        """Legacy function - replaced by AST visitor pattern."""
        pass


class GoStructureVisitor:
    """Tree-sitter visitor for extracting Go code structures with semantic analysis."""
    
    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.lines = content.splitlines()
        self.structures = []
        
    def visit(self, node):
        """Visit a tree-sitter node and extract structures."""
        # Handle different Go node types
        if node.type == "source_file":
            self._extract_package_info(node)
            
        elif node.type == "function_declaration":
            self._extract_function(node)
            
        elif node.type == "method_declaration":
            self._extract_method(node)
            
        elif node.type == "type_declaration":
            self._extract_type_declaration(node)
            
        # Recursively visit children
        for child in node.children:
            self.visit(child)
            
    def _extract_package_info(self, node):
        """Extract package information and documentation."""
        package_name = "unknown"
        package_doc = None
        
        for child in node.children:
            if child.type == "package_clause":
                package_name_node = child.child_by_field_name("name")
                if package_name_node:
                    package_name = self._get_node_text(package_name_node)
                    
            elif child.type == "comment":
                # Check if this comment is before package declaration
                if package_doc is None:
                    comment_text = self._get_node_text(child)
                    if comment_text.startswith("//"):
                        package_doc = comment_text.strip("// ").strip()
                    elif comment_text.startswith("/*") and comment_text.endswith("*/"):
                        package_doc = comment_text.strip("/* */").strip()
                    
        if package_doc or package_name != "unknown":
            self.structures.append(CodeStructure(
                name="__package__",
                type="package",
                signature=f"package {package_name}",
                documentation=package_doc,
                line_number=1,
                file_path=self.file_path
            ))
            
    def _extract_function(self, node):
        """Extract function declaration with full signature and analysis."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
            
        function_name = self._get_node_text(name_node)
        
        # Build complete signature
        signature = self._build_go_function_signature(node)
        
        # Extract documentation from preceding comments
        documentation = self._extract_go_comments_before(node)
        
        # Extract parameters
        parameters = self._extract_go_function_parameters(node)
        
        # Extract return type
        return_type = self._extract_go_return_type(node)
        
        # Calculate complexity
        complexity = self._calculate_go_function_complexity(node)
        
        self.structures.append(CodeStructure(
            name=function_name,
            type="function",
            signature=signature,
            documentation=documentation,
            line_number=node.start_point.row + 1,
            end_line_number=node.end_point.row + 1,
            file_path=self.file_path,
            parameters=parameters,
            return_type=return_type,
            complexity_indicators=complexity
        ))
        
    def _extract_method(self, node):
        """Extract method declaration (function with receiver)."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
            
        method_name = self._get_node_text(name_node)
        
        # Build complete signature including receiver
        signature = self._build_go_method_signature(node)
        
        # Extract documentation
        documentation = self._extract_go_comments_before(node)
        
        # Extract parameters (including receiver)
        parameters = self._extract_go_method_parameters(node)
        
        # Extract return type
        return_type = self._extract_go_return_type(node)
        
        # Calculate complexity
        complexity = self._calculate_go_function_complexity(node)
        
        self.structures.append(CodeStructure(
            name=method_name,
            type="method",
            signature=signature,
            documentation=documentation,
            line_number=node.start_point.row + 1,
            end_line_number=node.end_point.row + 1,
            file_path=self.file_path,
            parameters=parameters,
            return_type=return_type,
            complexity_indicators=complexity
        ))
        
    def _extract_type_declaration(self, node):
        """Extract type declarations (structs, interfaces, type aliases)."""
        for child in node.children:
            if child.type == "type_spec":
                self._extract_type_spec(child, node)
                
    def _extract_type_spec(self, spec_node, parent_node):
        """Extract specific type specification."""
        name_node = spec_node.child_by_field_name("name")
        type_node = spec_node.child_by_field_name("type")
        
        if not name_node:
            return
            
        type_name = self._get_node_text(name_node)
        
        if type_node:
            if type_node.type == "struct_type":
                self._extract_struct(spec_node, parent_node, type_name)
            elif type_node.type == "interface_type":
                self._extract_interface(spec_node, parent_node, type_name)
            else:
                self._extract_type_alias(spec_node, parent_node, type_name)
                
    def _extract_struct(self, spec_node, parent_node, struct_name):
        """Extract struct definition."""
        type_node = spec_node.child_by_field_name("type")
        
        # Build signature
        signature = f"type {struct_name} struct"
        
        # Extract fields
        fields = []
        if type_node and type_node.type == "struct_type":
            field_list = type_node.child_by_field_name("fields")
            if field_list:
                for field in field_list.children:
                    if field.type == "field_declaration":
                        field_names = self._extract_struct_field_names(field)
                        fields.extend(field_names)
                        
        # Extract documentation
        documentation = self._extract_go_comments_before(parent_node)
        
        # Calculate complexity
        complexity = {
            "field_count": len(fields),
            "total_lines": parent_node.end_point.row - parent_node.start_point.row + 1
        }
        
        self.structures.append(CodeStructure(
            name=struct_name,
            type="struct",
            signature=signature,
            documentation=documentation,
            line_number=parent_node.start_point.row + 1,
            end_line_number=parent_node.end_point.row + 1,
            file_path=self.file_path,
            fields=fields,
            complexity_indicators=complexity
        ))
        
    def _extract_interface(self, spec_node, parent_node, interface_name):
        """Extract interface definition."""
        type_node = spec_node.child_by_field_name("type")
        
        # Build signature
        signature = f"type {interface_name} interface"
        
        # Extract methods
        methods = []
        if type_node and type_node.type == "interface_type":
            method_list = type_node.child_by_field_name("methods")
            if method_list:
                for method in method_list.children:
                    if method.type == "method_spec":
                        method_sig = self._get_node_text(method)
                        methods.append(method_sig.strip())
                        
        # Extract documentation
        documentation = self._extract_go_comments_before(parent_node)
        
        # Calculate complexity
        complexity = {
            "method_count": len(methods),
            "total_lines": parent_node.end_point.row - parent_node.start_point.row + 1
        }
        
        self.structures.append(CodeStructure(
            name=interface_name,
            type="interface",
            signature=signature,
            documentation=documentation,
            line_number=parent_node.start_point.row + 1,
            end_line_number=parent_node.end_point.row + 1,
            file_path=self.file_path,
            fields=methods,  # Store method signatures in fields
            complexity_indicators=complexity
        ))
        
    def _extract_type_alias(self, spec_node, parent_node, type_name):
        """Extract type alias definition."""
        type_node = spec_node.child_by_field_name("type")
        
        if type_node:
            type_def = self._get_node_text(type_node)
            signature = f"type {type_name} {type_def}"
        else:
            signature = f"type {type_name}"
            
        # Extract documentation
        documentation = self._extract_go_comments_before(parent_node)
        
        self.structures.append(CodeStructure(
            name=type_name,
            type="type",
            signature=signature,
            documentation=documentation,
            line_number=parent_node.start_point.row + 1,
            end_line_number=parent_node.end_point.row + 1,
            file_path=self.file_path
        ))
        
    def _build_go_function_signature(self, node):
        """Build complete Go function signature."""
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        result_node = node.child_by_field_name("result")
        
        name = self._get_node_text(name_node) if name_node else "unknown"
        params = self._get_node_text(params_node) if params_node else "()"
        
        signature = f"func {name}{params}"
        
        if result_node:
            result = self._get_node_text(result_node)
            signature += f" {result}"
            
        return signature
        
    def _build_go_method_signature(self, node):
        """Build complete Go method signature including receiver."""
        receiver_node = node.child_by_field_name("receiver")
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        result_node = node.child_by_field_name("result")
        
        receiver = self._get_node_text(receiver_node) if receiver_node else ""
        name = self._get_node_text(name_node) if name_node else "unknown"
        params = self._get_node_text(params_node) if params_node else "()"
        
        signature = f"func {receiver} {name}{params}"
        
        if result_node:
            result = self._get_node_text(result_node)
            signature += f" {result}"
            
        return signature
        
    def _extract_go_function_parameters(self, node):
        """Extract parameter names from Go function."""
        params_node = node.child_by_field_name("parameters")
        parameters = []
        
        if params_node:
            for child in params_node.children:
                if child.type == "parameter_declaration":
                    name_nodes = [c for c in child.children if c.type == "identifier"]
                    for name_node in name_nodes:
                        param_name = self._get_node_text(name_node)
                        parameters.append(param_name)
                        
        return parameters
        
    def _extract_go_method_parameters(self, node):
        """Extract parameter names from Go method (including receiver)."""
        parameters = []
        
        # Add receiver
        receiver_node = node.child_by_field_name("receiver")
        if receiver_node:
            for child in receiver_node.children:
                if child.type == "parameter_declaration":
                    name_nodes = [c for c in child.children if c.type == "identifier"]
                    for name_node in name_nodes:
                        param_name = self._get_node_text(name_node)
                        parameters.append(f"receiver:{param_name}")
                        
        # Add regular parameters
        regular_params = self._extract_go_function_parameters(node)
        parameters.extend(regular_params)
        
        return parameters
        
    def _extract_go_return_type(self, node):
        """Extract return type from Go function/method."""
        result_node = node.child_by_field_name("result")
        if result_node:
            return self._get_node_text(result_node).strip()
        return None
        
    def _extract_struct_field_names(self, field_node):
        """Extract field names from struct field declaration."""
        field_names = []
        for child in field_node.children:
            if child.type == "field_identifier":
                field_names.append(self._get_node_text(child))
        return field_names
        
    def _calculate_go_function_complexity(self, node):
        """Calculate complexity metrics for Go function/method."""
        complexity = {
            "cyclomatic_complexity": 1,
            "if_statements": 0,
            "for_loops": 0,
            "switch_statements": 0,
            "defer_statements": 0,
            "return_statements": 0,
            "total_lines": node.end_point.row - node.start_point.row + 1
        }
        
        def visit_for_complexity(n):
            if n.type == "if_statement":
                complexity["if_statements"] += 1
                complexity["cyclomatic_complexity"] += 1
            elif n.type == "for_statement":
                complexity["for_loops"] += 1
                complexity["cyclomatic_complexity"] += 1
            elif n.type == "switch_statement":
                complexity["switch_statements"] += 1
                complexity["cyclomatic_complexity"] += 1
            elif n.type == "defer_statement":
                complexity["defer_statements"] += 1
            elif n.type == "return_statement":
                complexity["return_statements"] += 1
                
            for child in n.children:
                visit_for_complexity(child)
                
        visit_for_complexity(node)
        return complexity
        
    def _extract_go_comments_before(self, node):
        """Extract Go comments that appear before a node."""
        # Find comments in the lines before this node
        start_line = node.start_point.row
        
        comments = []
        for line_num in range(max(0, start_line - 10), start_line):
            if line_num < len(self.lines):
                line = self.lines[line_num].strip()
                if line.startswith("//"):
                    comments.append(line[2:].strip())
                elif line.startswith("/*") and line.endswith("*/"):
                    comments.append(line[2:-2].strip())
                elif line and not line.startswith("//") and not line.startswith("/*"):
                    # Non-comment, non-empty line found, stop looking
                    comments = []
                    
        return "\n".join(comments) if comments else None
        
    def _get_node_text(self, node):
        """Get the text content of a tree-sitter node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return self.content[start_byte:end_byte]
