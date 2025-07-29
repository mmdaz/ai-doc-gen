"""Tool for extracting complete implementations of code structures using AST and Tree-sitter."""

import ast
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field
from pydantic_ai import Tool

import tree_sitter_go as ts_go
from tree_sitter import Parser, Language

from utils import Logger


class CodeImplementation(BaseModel):
    """Represents the complete implementation of a code structure."""
    name: str = Field(..., description="Name of the structure")
    type: str = Field(..., description="Type: function, class, struct, interface, method")
    full_implementation: str = Field(..., description="Complete implementation code")
    documentation: Optional[str] = Field(None, description="Documentation/docstring/comments")
    line_start: int = Field(..., description="Starting line number")
    line_end: int = Field(..., description="Ending line number")
    file_path: str = Field(..., description="File path where structure is defined")
    complexity_analysis: Optional[Dict[str, Any]] = Field(None, description="Detailed complexity analysis")
    control_flow_summary: Optional[Dict[str, Any]] = Field(None, description="Control flow patterns summary")
    business_logic_indicators: Optional[List[str]] = Field(None, description="Indicators of business logic complexity")

class ImplementationExtractorTool:
    """Tool for extracting complete implementations of code structures."""

    def get_tool(self) -> Tool:
        """Get the implementation extractor tool."""
        return Tool(
            self._extract_implementation,
            name="extract_code_implementation",
            description="Extract the complete implementation of a specific function, class, struct, or interface by name from a Go or Python file",
            takes_ctx=False,
            max_retries=2
        )

    def _extract_implementation(self, file_path: str, structure_name: str, structure_type: Optional[str] = None) -> Optional[CodeImplementation]:
        """
        Extract the complete implementation of a specific code structure.
        
        Args:
            file_path: Path to the Go or Python file
            structure_name: Name of the function, class, struct, or interface to extract
            structure_type: Optional type hint (function, class, struct, interface, method)
            
        Returns:
            CodeImplementation object with the complete implementation or None if not found
        """
        Logger.info(f"Extracting implementation for {file_path}")
        try:
            if not os.path.exists(file_path):
                return None
                
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == ".py":
                py_impl = self._extract_python_implementation(file_path, structure_name)
                Logger.info(f"Extracted {py_impl}")
                return py_impl
            elif file_extension == ".go":
                go_impl = self._extract_go_implementation(file_path, structure_name)
                Logger.info(f"Extracted {go_impl}")
                return go_impl
            else:
                return None

        except Exception as e:
            print(f"Error extracting implementation from {file_path}: {e}")
            return None

    def _extract_python_implementation(self, file_path: str, structure_name: str) -> Optional[CodeImplementation]:
        """Extract implementation from Python files using AST with advanced analysis."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            tree = ast.parse(content)
            lines = content.splitlines()
            
            # Use visitor to find and analyze the target structure
            visitor = PythonImplementationVisitor(file_path, lines, structure_name)
            visitor.visit(tree)
            
            return visitor.found_implementation
                    
        except Exception as e:
            print(f"Error parsing Python file {file_path}: {e}")
            raise e

    def _extract_go_implementation(self, file_path: str, structure_name: str) -> Optional[CodeImplementation]:
        """Extract implementation from Go files using Tree-sitter."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Initialize Go parser using tree-sitter
            parser = Parser(language=Language(ts_go.language()))

            # Parse the content
            tree = parser.parse(content.encode('utf-8'))

            # Use visitor to find and analyze the target structure
            visitor = GoImplementationVisitor(file_path, content, structure_name)
            visitor.visit(tree.root_node)

            return visitor.found_implementation

        except Exception as e:
            print(f"Error parsing Go file with Tree-sitter {file_path}: {e}")
            raise e


class PythonImplementationVisitor(ast.NodeVisitor):
    """AST visitor for extracting detailed Python implementation analysis."""
    
    def __init__(self, file_path: str, lines: List[str], target_name: str):
        self.file_path = file_path
        self.lines = lines
        self.target_name = target_name
        self.found_implementation = None
        self.current_class = None
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        
        if node.name == self.target_name:
            self.found_implementation = self._extract_class_implementation(node)
            return
            
        # Continue visiting child nodes
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions."""
        if node.name == self.target_name:
            self.found_implementation = self._extract_function_implementation(node, is_async=False)
            
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definitions."""
        if node.name == self.target_name:
            self.found_implementation = self._extract_function_implementation(node, is_async=True)
            
    def _extract_function_implementation(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool = False) -> CodeImplementation:
        """Extract detailed function implementation with business logic analysis."""
        # Get precise line boundaries
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', None)
        
        if end_line is None:
            # Fallback: find end by indentation
            end_line = self._find_function_end_line(node, start_line)
            
        # Extract complete implementation
        implementation_lines = self.lines[start_line - 1:end_line]
        full_implementation = "\n".join(implementation_lines)
        
        # Perform advanced analysis
        complexity_analysis = self._analyze_function_complexity(node)
        control_flow_summary = self._analyze_control_flow(node)
        business_logic_indicators = self._identify_business_logic_patterns(node)
        
        func_type = "method" if self.current_class else ("async_function" if is_async else "function")
        
        return CodeImplementation(
            name=node.name,
            type=func_type,
            full_implementation=full_implementation,
            documentation=ast.get_docstring(node),
            line_start=start_line,
            line_end=end_line,
            file_path=self.file_path,
            complexity_analysis=complexity_analysis,
            control_flow_summary=control_flow_summary,
            business_logic_indicators=business_logic_indicators
        )
        
    def _extract_class_implementation(self, node: ast.ClassDef) -> CodeImplementation:
        """Extract detailed class implementation with analysis."""
        # Get precise line boundaries
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', None)
        
        if end_line is None:
            end_line = self._find_class_end_line(node, start_line)
            
        # Extract complete implementation
        implementation_lines = self.lines[start_line - 1:end_line]
        full_implementation = "\n".join(implementation_lines)
        
        # Perform analysis
        complexity_analysis = self._analyze_class_complexity(node)
        business_logic_indicators = self._identify_class_business_logic_patterns(node)
        
        return CodeImplementation(
            name=node.name,
            type="class",
            full_implementation=full_implementation,
            documentation=ast.get_docstring(node),
            line_start=start_line,
            line_end=end_line,
            file_path=self.file_path,
            complexity_analysis=complexity_analysis,
            business_logic_indicators=business_logic_indicators
        )
        
    def _analyze_function_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Perform detailed complexity analysis of a function."""
        analysis = {
            "cyclomatic_complexity": 1,
            "cognitive_complexity": 0,
            "nesting_depth": 0,
            "decision_points": [],
            "loop_patterns": [],
            "exception_handling": [],
            "database_operations": [],
            "external_calls": [],
            "state_modifications": [],
            "conditional_logic": []
        }
        
        # Analyze all nodes in the function
        for child in ast.walk(node):
            self._analyze_node_complexity(child, analysis, 0)
            
        return analysis
        
    def _analyze_control_flow(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Analyze control flow patterns in the function."""
        control_flow = {
            "branching_factor": 0,
            "early_returns": 0,
            "nested_conditions": 0,
            "loop_types": [],
            "exception_patterns": [],
            "async_patterns": [],
            "decision_trees": []
        }
        
        # Walk through function body to analyze control flow
        self._analyze_control_flow_recursive(node.body, control_flow, 0)
        
        return control_flow
        
    def _identify_business_logic_patterns(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Identify patterns that indicate complex business logic."""
        indicators = []
        
        # Look for specific patterns
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                # Complex conditional logic
                if self._is_complex_condition(child.test):
                    indicators.append(f"Complex conditional at line {child.lineno}")
                    
            elif isinstance(child, ast.For):
                # Data processing loops
                indicators.append(f"Data processing loop at line {child.lineno}")
                
            elif isinstance(child, ast.Try):
                # Error handling logic
                indicators.append(f"Exception handling at line {child.lineno}")
                
            elif isinstance(child, ast.Call):
                # Look for database, validation, calculation patterns
                if isinstance(child.func, ast.Attribute):
                    method_name = child.func.attr.lower()
                    if any(pattern in method_name for pattern in ['validate', 'check', 'verify', 'calculate', 'process']):
                        indicators.append(f"Business logic call '{method_name}' at line {child.lineno}")
                        
        return indicators
        
    def _analyze_node_complexity(self, node: ast.AST, analysis: Dict[str, Any], depth: int):
        """Analyze complexity of individual AST nodes."""
        if isinstance(node, (ast.If, ast.IfExp)):
            analysis["cyclomatic_complexity"] += 1
            analysis["cognitive_complexity"] += 1 + depth
            analysis["decision_points"].append({
                "type": "if_statement",
                "line": node.lineno,
                "condition_complexity": self._calculate_condition_complexity(node.test if hasattr(node, 'test') else node.test)
            })
            
        elif isinstance(node, (ast.For, ast.While)):
            analysis["cyclomatic_complexity"] += 1
            analysis["cognitive_complexity"] += 1 + depth
            analysis["loop_patterns"].append({
                "type": type(node).__name__.lower(),
                "line": node.lineno,
                "nesting_depth": depth
            })
            
        elif isinstance(node, ast.Try):
            analysis["exception_handling"].append({
                "line": node.lineno,
                "handlers_count": len(node.handlers),
                "has_finally": bool(node.finalbody)
            })
            
        elif isinstance(node, ast.Call):
            self._analyze_call_complexity(node, analysis)
            
    def _analyze_call_complexity(self, node: ast.Call, analysis: Dict[str, Any]):
        """Analyze function/method calls for business logic indicators."""
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr.lower()
            
            # Database operation patterns
            if any(db_op in attr_name for db_op in ['execute', 'query', 'insert', 'update', 'delete', 'select']):
                analysis["database_operations"].append({
                    "operation": attr_name,
                    "line": node.lineno
                })
                
            # External service calls
            elif any(ext_op in attr_name for ext_op in ['request', 'call', 'invoke', 'send']):
                analysis["external_calls"].append({
                    "operation": attr_name,
                    "line": node.lineno
                })
                
    def _is_complex_condition(self, condition: ast.AST) -> bool:
        """Determine if a condition is complex."""
        if isinstance(condition, (ast.BoolOp, ast.Compare)):
            return True
        elif isinstance(condition, ast.Call):
            return True
        return False
        
    def _calculate_condition_complexity(self, condition: ast.AST) -> int:
        """Calculate complexity score for a condition."""
        complexity = 1
        
        for child in ast.walk(condition):
            if isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.Compare):
                complexity += len(child.ops)
                
        return complexity
        
    def _analyze_control_flow_recursive(self, statements: List[ast.stmt], control_flow: Dict[str, Any], depth: int):
        """Recursively analyze control flow patterns."""
        for stmt in statements:
            if isinstance(stmt, ast.If):
                control_flow["branching_factor"] += 1
                if depth > 0:
                    control_flow["nested_conditions"] += 1
                    
                # Analyze nested statements
                self._analyze_control_flow_recursive(stmt.body, control_flow, depth + 1)
                self._analyze_control_flow_recursive(stmt.orelse, control_flow, depth + 1)
                
            elif isinstance(stmt, ast.Return):
                control_flow["early_returns"] += 1
                
            elif isinstance(stmt, (ast.For, ast.While)):
                loop_type = type(stmt).__name__.lower()
                control_flow["loop_types"].append(loop_type)
                self._analyze_control_flow_recursive(stmt.body, control_flow, depth + 1)
                
    def _find_function_end_line(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], start_line: int) -> int:
        """Find the end line of a function by analyzing indentation."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
            
        # Fallback: find by indentation
        function_indent = len(self.lines[start_line - 1]) - len(self.lines[start_line - 1].lstrip())
        
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= function_indent and i > start_line:
                if line.lstrip().startswith(('def ', 'class ', '@')) or not line.startswith(' ' * (function_indent + 1)):
                    return i
                    
        return len(self.lines)
        
    def _find_class_end_line(self, node: ast.ClassDef, start_line: int) -> int:
        """Find the end line of a class."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
            
        # Similar logic as function
        class_indent = len(self.lines[start_line - 1]) - len(self.lines[start_line - 1].lstrip())
        
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= class_indent and i > start_line:
                if line.lstrip().startswith(('def ', 'class ', '@')):
                    return i
                    
        return len(self.lines)
        
    def _analyze_class_complexity(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze class complexity."""
        return {
            "methods_count": len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
            "inheritance_depth": len(node.bases),
            "total_lines": getattr(node, 'end_lineno', len(self.lines)) - node.lineno + 1,
            "public_methods": len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith('_')]),
            "private_methods": len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('_')])
        }
        
    def _identify_class_business_logic_patterns(self, node: ast.ClassDef) -> List[str]:
        """Identify business logic patterns in a class."""
        indicators = []
        
        for method in node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_indicators = self._identify_business_logic_patterns(method)
                indicators.extend([f"In method {method.name}: {indicator}" for indicator in method_indicators])
                
        return indicators

    def _extract_python_function_implementation(self, node: ast.FunctionDef, lines: List[str], file_path: str) -> CodeImplementation:
        """Legacy function - replaced by AST visitor pattern."""
        pass

    def _extract_python_class_implementation(self, node: ast.ClassDef, lines: List[str], file_path: str) -> CodeImplementation:
        """Legacy function - replaced by AST visitor pattern."""
        pass


class GoImplementationVisitor:
    """Tree-sitter visitor for extracting detailed Go implementation analysis."""
    
    def __init__(self, file_path: str, content: str, target_name: str):
        self.file_path = file_path
        self.content = content
        self.lines = content.splitlines()
        self.target_name = target_name
        self.found_implementation = None
        
    def visit(self, node):
        """Visit a tree-sitter node and look for target structure."""
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node and self._get_node_text(name_node) == self.target_name:
                self.found_implementation = self._extract_function_implementation(node)
                return
                
        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node and self._get_node_text(name_node) == self.target_name:
                self.found_implementation = self._extract_method_implementation(node)
                return
                
        elif node.type == "type_declaration":
            # Check for struct, interface, or type alias with target name
            for child in node.children:
                if child.type == "type_spec":
                    name_node = child.child_by_field_name("name")
                    if name_node and self._get_node_text(name_node) == self.target_name:
                        self.found_implementation = self._extract_type_implementation(node, child)
                        return
                        
        # Continue visiting children
        for child in node.children:
            if self.found_implementation is None:
                self.visit(child)
                
    def _extract_function_implementation(self, node):
        """Extract detailed Go function implementation."""
        name_node = node.child_by_field_name("name")
        function_name = self._get_node_text(name_node)
        
        # Get implementation text
        full_implementation = self._get_node_text(node)
        
        # Extract documentation
        documentation = self._extract_go_comments_before(node)
        
        # Perform complexity analysis
        complexity_analysis = self._analyze_go_function_complexity(node)
        control_flow_summary = self._analyze_go_control_flow(node)
        business_logic_indicators = self._identify_go_business_logic_patterns(node)
        
        return CodeImplementation(
            name=function_name,
            type="function",
            full_implementation=full_implementation,
            documentation=documentation,
            line_start=node.start_point.row + 1,
            line_end=node.end_point.row + 1,
            file_path=self.file_path,
            complexity_analysis=complexity_analysis,
            control_flow_summary=control_flow_summary,
            business_logic_indicators=business_logic_indicators
        )
        
    def _extract_method_implementation(self, node):
        """Extract detailed Go method implementation."""
        name_node = node.child_by_field_name("name")
        method_name = self._get_node_text(name_node)
        
        # Get implementation text
        full_implementation = self._get_node_text(node)
        
        # Extract documentation
        documentation = self._extract_go_comments_before(node)
        
        # Perform complexity analysis
        complexity_analysis = self._analyze_go_function_complexity(node)
        control_flow_summary = self._analyze_go_control_flow(node)
        business_logic_indicators = self._identify_go_business_logic_patterns(node)
        
        return CodeImplementation(
            name=method_name,
            type="method",
            full_implementation=full_implementation,
            documentation=documentation,
            line_start=node.start_point.row + 1,
            line_end=node.end_point.row + 1,
            file_path=self.file_path,
            complexity_analysis=complexity_analysis,
            control_flow_summary=control_flow_summary,
            business_logic_indicators=business_logic_indicators
        )
        
    def _extract_type_implementation(self, parent_node, spec_node):
        """Extract detailed Go type implementation."""
        name_node = spec_node.child_by_field_name("name")
        type_name = self._get_node_text(name_node)
        
        # Get implementation text (full type declaration)
        full_implementation = self._get_node_text(parent_node)
        
        # Extract documentation
        documentation = self._extract_go_comments_before(parent_node)
        
        # Basic complexity analysis for types
        complexity_analysis = {
            "total_lines": parent_node.end_point.row - parent_node.start_point.row + 1,
            "definition_complexity": self._calculate_type_complexity(spec_node)
        }
        
        return CodeImplementation(
            name=type_name,
            type="type",
            full_implementation=full_implementation,
            documentation=documentation,
            line_start=parent_node.start_point.row + 1,
            line_end=parent_node.end_point.row + 1,
            file_path=self.file_path,
            complexity_analysis=complexity_analysis
        )
        
    def _analyze_go_function_complexity(self, node):
        """Analyze complexity of Go function/method."""
        analysis = {
            "cyclomatic_complexity": 1,
            "cognitive_complexity": 0,
            "nesting_depth": 0,
            "decision_points": [],
            "loop_patterns": [],
            "error_handling": [],
            "defer_statements": [],
            "goroutine_usage": [],
            "channel_operations": [],
            "interface_assertions": []
        }
        
        def analyze_recursive(n, depth=0):
            if n.type == "if_statement":
                analysis["cyclomatic_complexity"] += 1
                analysis["cognitive_complexity"] += 1 + depth
                analysis["decision_points"].append({
                    "type": "if_statement",
                    "line": n.start_point.row + 1,
                    "nesting_depth": depth
                })
                
            elif n.type == "for_statement":
                analysis["cyclomatic_complexity"] += 1
                analysis["cognitive_complexity"] += 1 + depth
                analysis["loop_patterns"].append({
                    "type": "for_loop",
                    "line": n.start_point.row + 1,
                    "nesting_depth": depth
                })
                
            elif n.type == "switch_statement":
                analysis["cyclomatic_complexity"] += 1
                analysis["cognitive_complexity"] += 1 + depth
                analysis["decision_points"].append({
                    "type": "switch_statement",
                    "line": n.start_point.row + 1,
                    "nesting_depth": depth
                })
                
            elif n.type == "defer_statement":
                analysis["defer_statements"].append({
                    "line": n.start_point.row + 1
                })
                
            elif n.type == "go_statement":
                analysis["goroutine_usage"].append({
                    "line": n.start_point.row + 1
                })
                
            elif n.type == "send_statement" or n.type == "receive_expression":
                analysis["channel_operations"].append({
                    "type": n.type,
                    "line": n.start_point.row + 1
                })
                
            elif n.type == "type_assertion_expression":
                analysis["interface_assertions"].append({
                    "line": n.start_point.row + 1
                })
                
            # Recursively analyze children
            for child in n.children:
                analyze_recursive(child, depth + 1)
                
        analyze_recursive(node)
        return analysis
        
    def _analyze_go_control_flow(self, node):
        """Analyze control flow patterns in Go function."""
        control_flow = {
            "branching_factor": 0,
            "early_returns": 0,
            "panic_statements": 0,
            "nested_conditions": 0,
            "error_checks": 0,
            "loop_types": [],
            "concurrency_patterns": []
        }
        
        def analyze_flow(n, depth=0):
            if n.type == "if_statement":
                control_flow["branching_factor"] += 1
                if depth > 0:
                    control_flow["nested_conditions"] += 1
                    
                # Check for error handling pattern
                condition_text = self._get_node_text(n.child_by_field_name("condition") or n)
                if "err" in condition_text and "!=" in condition_text:
                    control_flow["error_checks"] += 1
                    
            elif n.type == "return_statement":
                control_flow["early_returns"] += 1
                
            elif n.type == "call_expression":
                func_text = self._get_node_text(n.child_by_field_name("function") or n)
                if "panic" in func_text:
                    control_flow["panic_statements"] += 1
                    
            elif n.type == "for_statement":
                control_flow["loop_types"].append("for")
                
            elif n.type == "go_statement":
                control_flow["concurrency_patterns"].append("goroutine")
                
            elif n.type == "select_statement":
                control_flow["concurrency_patterns"].append("select")
                
            for child in n.children:
                analyze_flow(child, depth + 1)
                
        analyze_flow(node)
        return control_flow
        
    def _identify_go_business_logic_patterns(self, node):
        """Identify business logic patterns in Go function."""
        indicators = []
        
        def find_patterns(n):
            if n.type == "if_statement":
                condition_text = self._get_node_text(n.child_by_field_name("condition") or n)
                if any(pattern in condition_text.lower() for pattern in ['validate', 'check', 'verify', 'timeout']):
                    indicators.append(f"Business validation logic at line {n.start_point.row + 1}")
                    
            elif n.type == "call_expression":
                func_node = n.child_by_field_name("function")
                if func_node:
                    func_text = self._get_node_text(func_node).lower()
                    if any(pattern in func_text for pattern in ['calculate', 'process', 'transform', 'validate', 'lock', 'acquire']):
                        indicators.append(f"Business logic call '{func_text}' at line {n.start_point.row + 1}")
                        
            elif n.type == "for_statement":
                indicators.append(f"Data processing loop at line {n.start_point.row + 1}")
                
            elif n.type == "switch_statement":
                indicators.append(f"Decision logic (switch) at line {n.start_point.row + 1}")
                
            for child in n.children:
                find_patterns(child)
                
        find_patterns(node)
        return indicators
        
    def _calculate_type_complexity(self, spec_node):
        """Calculate complexity for type definitions."""
        type_node = spec_node.child_by_field_name("type")
        if not type_node:
            return {"type": "simple"}
            
        if type_node.type == "struct_type":
            field_count = len([c for c in type_node.children if c.type == "field_declaration"])
            return {"type": "struct", "field_count": field_count}
        elif type_node.type == "interface_type":
            method_count = len([c for c in type_node.children if c.type == "method_spec"])
            return {"type": "interface", "method_count": method_count}
        else:
            return {"type": "alias", "definition": self._get_node_text(type_node)}
            
    def _extract_go_comments_before(self, node):
        """Extract Go comments before a node."""
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
                    comments = []
                    
        return "\n".join(comments) if comments else None

    def _get_node_text(self, node):
        """Get text content of a tree-sitter node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return self.content[start_byte:end_byte]
