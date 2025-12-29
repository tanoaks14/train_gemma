import argparse
import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "target",
    ".idea",
    ".vscode",
    "gemma_finetune_env",
    "functiongemma-finetuned",
}


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": self.parameters or {"type": "object", "properties": {}, "required": []},
            },
        }


def should_skip_path(path: Path, excludes: set[str]) -> bool:
    parts = {p for p in path.parts}
    return any(part in excludes for part in parts)


def json_type_from_annotation(node: Optional[ast.expr]) -> str:
    if node is None:
        return "string"

    # Names: str, int, float, bool, dict, list
    if isinstance(node, ast.Name):
        mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "dict": "object",
            "list": "array",
        }
        return mapping.get(node.id, "string")

    # Subscripts: list[str], dict[str, int], Optional[str], etc.
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Name):
            if base.id in {"List", "list", "Sequence", "Iterable", "Tuple", "Set", "set"}:
                return "array"
            if base.id in {"Dict", "dict", "Mapping"}:
                return "object"
            if base.id in {"Optional", "Union"}:
                # Optional[T] -> treat as type of T
                sub = node.slice
                if isinstance(sub, ast.Tuple) and sub.elts:
                    return json_type_from_annotation(sub.elts[0])
                return json_type_from_annotation(sub)
        return "string"

    # Attribute: typing.List, etc.
    if isinstance(node, ast.Attribute):
        if node.attr in {"List", "Sequence", "Iterable", "Tuple", "Set"}:
            return "array"
        if node.attr in {"Dict", "Mapping"}:
            return "object"

    return "string"


def dummy_value_for_schema(schema: Dict[str, Any]) -> Any:
    t = (schema or {}).get("type", "string")
    if isinstance(t, list):
        t = t[0] if t else "string"

    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]

    if t == "string":
        return "example"
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return True
    if t == "array":
        return []
    if t == "object":
        return {}
    return "example"


def build_parameters_from_signature(fn: ast.FunctionDef) -> Dict[str, Any]:
    args = fn.args
    positional = list(args.posonlyargs) + list(args.args)

    # Skip implicit self/cls
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]

    defaults = list(args.defaults)
    defaults_pad = [None] * (len(positional) - len(defaults)) + defaults

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for a, default_node in zip(positional, defaults_pad):
        name = a.arg
        prop_schema: Dict[str, Any] = {
            "type": json_type_from_annotation(a.annotation),
            "description": "",
        }
        properties[name] = prop_schema
        if default_node is None:
            required.append(name)

    # kwargs / *args ignored for now
    return {"type": "object", "properties": properties, "required": required}


def find_python_tools(project_root: Path, excludes: set[str]) -> List[ToolDef]:
    tools: List[ToolDef] = []

    def decorator_matches(dec: ast.expr) -> Tuple[bool, Optional[str]]:
        """Return (is_tool_decorator, explicit_name_if_any)."""
        # @tool or @mcp.tool
        if isinstance(dec, ast.Name) and dec.id == "tool":
            return True, None
        if isinstance(dec, ast.Attribute) and dec.attr == "tool":
            return True, None

        # @tool("name") / @mcp.tool(name="...") / @server.tool("name")
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Name) and func.id == "tool":
                pass
            elif isinstance(func, ast.Attribute) and func.attr == "tool":
                pass
            else:
                return False, None

            # name from first arg if string
            if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                return True, dec.args[0].value

            for kw in dec.keywords or []:
                if kw.arg in {"name", "tool_name"} and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    return True, kw.value.value

            return True, None

        return False, None

    for py_file in project_root.rglob("*.py"):
        if should_skip_path(py_file, excludes):
            continue

        try:
            src = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        try:
            tree = ast.parse(src, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            tool_name: Optional[str] = None
            is_tool = False

            for dec in node.decorator_list:
                matched, explicit = decorator_matches(dec)
                if matched:
                    is_tool = True
                    tool_name = explicit or node.name
                    break

            if not is_tool:
                continue

            description = ast.get_docstring(node) or ""
            description = description.strip().splitlines()[0] if description else ""

            params = build_parameters_from_signature(node)
            tools.append(ToolDef(name=tool_name, description=description, parameters=params))

    return tools


def find_node_tools(project_root: Path, excludes: set[str]) -> List[ToolDef]:
    tools: List[ToolDef] = []

    def _js_skip_ws(src: str, i: int) -> int:
        while i < len(src) and src[i].isspace():
            i += 1
        return i

    def _js_parse_string_literal(text: str) -> Optional[str]:
        if not text:
            return None
        quote = text[0]
        if quote not in {"'", '"', "`"}:
            return None
        out: List[str] = []
        i = 1
        while i < len(text):
            ch = text[i]
            if ch == "\\" and i + 1 < len(text):
                # keep escaped char as-is (best-effort)
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                return "".join(out)
            out.append(ch)
            i += 1
        return None

    def _js_parse_call_args(src: str, open_paren: int) -> Tuple[List[str], int]:
        """Parse JS/TS call args starting at '('. Returns (args, close_paren_index)."""
        args: List[str] = []
        buf: List[str] = []
        i = open_paren + 1
        stack: List[str] = ["("]
        in_str: Optional[str] = None
        in_line_comment = False
        in_block_comment = False

        while i < len(src):
            ch = src[i]

            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                buf.append(ch)
                i += 1
                continue

            if in_block_comment:
                if ch == "*" and i + 1 < len(src) and src[i + 1] == "/":
                    buf.append("*/")
                    i += 2
                    in_block_comment = False
                    continue
                buf.append(ch)
                i += 1
                continue

            if in_str:
                buf.append(ch)
                if ch == "\\" and i + 1 < len(src):
                    buf.append(src[i + 1])
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue

            # start comments
            if ch == "/" and i + 1 < len(src) and src[i + 1] == "/":
                in_line_comment = True
                buf.append("//")
                i += 2
                continue
            if ch == "/" and i + 1 < len(src) and src[i + 1] == "*":
                in_block_comment = True
                buf.append("/*")
                i += 2
                continue

            # start strings
            if ch in {"'", '"', "`"}:
                in_str = ch
                buf.append(ch)
                i += 1
                continue

            if ch in "([{":
                stack.append(ch)
                buf.append(ch)
                i += 1
                continue
            if ch in ")]}":
                if not stack:
                    i += 1
                    continue
                opener = stack[-1]
                if (opener, ch) in {("(", ")"), ("[", "]"), ("{", "}")}: 
                    stack.pop()
                buf.append(ch)
                i += 1
                if not stack:
                    # end of call
                    arg = "".join(buf[:-1]).strip()  # exclude the final ')'
                    if arg:
                        args.append(arg)
                    return args, i - 1
                continue

            # top-level comma
            if ch == "," and stack == ["("]:
                arg = "".join(buf).strip()
                if arg:
                    args.append(arg)
                buf = []
                i += 1
                continue

            buf.append(ch)
            i += 1

        return args, len(src) - 1

    def _extract_description(text: str) -> str:
        m = re.search(r"\bdescription\s*:\s*(['\"`])([^'\"`]+)\1", text, re.MULTILINE)
        return m.group(2).strip() if m else ""

    def _extract_object_or_call_expression(text: str, start: int) -> Optional[str]:
        """Extract a JS expression starting at start, stopping at top-level comma/end.

        Supports object literals and call expressions like z.object(...).
        """
        i = _js_skip_ws(text, start)
        if i >= len(text):
            return None

        # String literal
        if text[i] in {"'", '"', "`"}:
            s = _js_parse_string_literal(text[i:])
            if s is None:
                return None
            # Reconstruct original literal from parsed length by scanning to closing quote
            quote = text[i]
            j = i + 1
            while j < len(text):
                if text[j] == "\\" and j + 1 < len(text):
                    j += 2
                    continue
                if text[j] == quote:
                    return text[i : j + 1]
                j += 1
            return None

        # Object literal: return balanced {...}
        if text[i] == "{":
            in_str: Optional[str] = None
            depth = 0
            j = i
            while j < len(text):
                ch = text[j]
                if in_str:
                    if ch == "\\" and j + 1 < len(text):
                        j += 2
                        continue
                    if ch == in_str:
                        in_str = None
                    j += 1
                    continue
                if ch in {"'", '"', "`"}:
                    in_str = ch
                    j += 1
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
                j += 1
            return None

        # Call expression (best-effort): return balanced (...) including nested braces
        if text.startswith("z.object", i):
            open_idx = text.find("(", i)
            if open_idx == -1:
                return None
            in_str: Optional[str] = None
            stack: List[str] = []
            j = i
            while j < len(text):
                ch = text[j]
                if in_str:
                    if ch == "\\" and j + 1 < len(text):
                        j += 2
                        continue
                    if ch == in_str:
                        in_str = None
                    j += 1
                    continue
                if ch in {"'", '"', "`"}:
                    in_str = ch
                    j += 1
                    continue
                if ch in "({[":
                    stack.append(ch)
                elif ch in ")}]":
                    if stack:
                        opener = stack[-1]
                        if (opener, ch) in {("(", ")"), ("[", "]"), ("{", "}")}: 
                            stack.pop()
                            if not stack and j >= open_idx:
                                return text[i : j + 1]
                j += 1
            return None

        # Fallback: read to top-level comma
        j = i
        while j < len(text) and text[j] != ",":
            j += 1
        expr = text[i:j].strip()
        return expr or None

    def _extract_property_value(meta_text: str, prop: str) -> Optional[str]:
        # Find "prop:" and extract the expression following it.
        m = re.search(rf"\b{re.escape(prop)}\s*:\s*", meta_text)
        if not m:
            return None
        return _extract_object_or_call_expression(meta_text, m.end())

    def _try_parse_jsonish_object(obj_text: str) -> Optional[Dict[str, Any]]:
        obj_text = obj_text.strip()
        if not obj_text.startswith("{"):
            return None
        # Fast path: valid JSON already
        try:
            return json.loads(obj_text)
        except Exception:
            pass

        # Best-effort conversion: quote unquoted keys, normalize quotes, strip trailing commas
        candidate = obj_text
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        candidate = re.sub(r"([,{]\s*)([A-Za-z_$][\w$]*)(\s*:)", r'\1"\2"\3', candidate)
        # replace single-quoted strings with double-quoted strings (very heuristic)
        candidate = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", lambda m: '"' + m.group(1).replace('"', '\\"') + '"', candidate)
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def _schema_from_zod_object(arg_text: str) -> Optional[Dict[str, Any]]:
        # inputSchema: z.object({ a: z.string(), b: z.number() })
        m = re.search(r"\bz\.object\s*\(\s*(\{.*)\)\s*$", arg_text.strip(), re.DOTALL)
        if not m:
            return None
        # Grab the first '{...}' we can balance at the front of the capture
        payload = m.group(1).lstrip()
        if not payload.startswith("{"):
            return None
        # crude brace balance for the object
        depth = 0
        end = None
        in_str: Optional[str] = None
        i = 0
        while i < len(payload):
            ch = payload[i]
            if in_str:
                if ch == "\\" and i + 1 < len(payload):
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch in {"'", '"', "`"}:
                in_str = ch
                i += 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        if end is None:
            return None
        obj_body = payload[: end + 1]
        # Extract keys at top-level (best-effort)
        # We'll treat all as strings; this is still useful for tool-call shaping.
        keys = re.findall(r"\b([A-Za-z_$][\w$]*)\s*:\s*z\.", obj_body)
        properties = {k: {"type": "string", "description": ""} for k in keys}
        return {"type": "object", "properties": properties, "required": sorted(properties.keys())}

    def _extract_input_schema(meta_text: str) -> Optional[Dict[str, Any]]:
        rhs = _extract_property_value(meta_text, "inputSchema")
        if not rhs:
            return None
        rhs = rhs.strip()
        if rhs.startswith("{"):
            parsed = _try_parse_jsonish_object(rhs)
            if isinstance(parsed, dict) and parsed.get("type") == "object":
                return parsed
            if isinstance(parsed, dict):
                return parsed
            return None
        if "z.object" in rhs:
            return _schema_from_zod_object(rhs)
        return None

    call_patterns = [
        re.compile(r"\bregisterTool\s*\(", re.MULTILINE),
        re.compile(r"\.registerTool\s*\(", re.MULTILINE),
        re.compile(r"\btool\s*\(", re.MULTILINE),
        re.compile(r"\.tool\s*\(", re.MULTILINE),
    ]

    for ext in ("*.ts", "*.js", "*.mts", "*.cts", "*.tsx", "*.jsx"):
        for file in project_root.rglob(ext):
            if should_skip_path(file, excludes):
                continue
            try:
                src = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            for pat in call_patterns:
                for m in pat.finditer(src):
                    open_paren = m.end() - 1
                    if open_paren >= len(src) or src[open_paren] != "(":
                        continue
                    args, _close = _js_parse_call_args(src, open_paren)
                    if not args:
                        continue

                    # Common signatures:
                    # - registerTool("name", { description, inputSchema }, handler)
                    # - tool({ name, description, inputSchema }, handler)
                    name = ""
                    desc = ""
                    params_schema: Optional[Dict[str, Any]] = None

                    a0 = args[0].strip()
                    s0 = _js_parse_string_literal(a0)
                    if s0:
                        name = s0.strip()
                        if len(args) > 1:
                            meta = args[1]
                            desc = _extract_description(meta)
                            params_schema = _extract_input_schema(meta)
                    else:
                        # tool({ name: "x", ... }, handler)
                        if a0.lstrip().startswith("{"):
                            meta = a0
                            # name from meta
                            nm = re.search(r"\bname\s*:\s*(['\"`])([^'\"`]+)\1", meta)
                            if nm:
                                name = nm.group(2).strip()
                            desc = _extract_description(meta)
                            params_schema = _extract_input_schema(meta)

                    if not name:
                        continue

                    tools.append(
                        ToolDef(
                            name=name,
                            description=desc,
                            parameters=params_schema or {"type": "object", "properties": {}, "required": []},
                        )
                    )

    return tools


def find_java_tools(project_root: Path, excludes: set[str]) -> List[ToolDef]:
    tools: List[ToolDef] = []

    # Heuristic patterns
    # - @Tool(name = "x")
    # - @Tool("x")
    # - registerTool("x")
    anno_pattern = re.compile(r"@(?:McpTool|Tool)\s*(?:\((?P<args>[^)]*)\))?", re.MULTILINE)
    register_pattern = re.compile(r"\bregisterTool\s*\(\s*\"([^\"]+)\"", re.MULTILINE)
    add_tool_ctor_pattern = re.compile(
        r"\baddTool\s*\(\s*new\s+(?:[A-Za-z_][\w]*\.)*(?:Tool|[A-Za-z_][\w]*Tool)\s*\(\s*\"([^\"]+)\"\s*(?:,\s*\"([^\"]*)\")?",
        re.MULTILINE,
    )
    tool_builder_pattern = re.compile(r"\bTool\s*\.\s*builder\s*\(\s*\"([^\"]+)\"", re.MULTILINE)

    def _java_type_to_json_type(t: str) -> str:
        t = t.strip()
        t = re.sub(r"\s+", " ", t)
        t = t.replace("[]", "")
        t = re.sub(r"<.*?>", "", t)
        mapping = {
            "String": "string",
            "CharSequence": "string",
            "UUID": "string",
            "int": "integer",
            "Integer": "integer",
            "long": "integer",
            "Long": "integer",
            "double": "number",
            "Double": "number",
            "float": "number",
            "Float": "number",
            "BigDecimal": "number",
            "boolean": "boolean",
            "Boolean": "boolean",
        }
        base = t.split(" ")[0].split(".")[-1]
        return mapping.get(base, "string")

    def _split_java_params(params: str) -> List[str]:
        parts: List[str] = []
        buf: List[str] = []
        depth_angle = 0
        in_str: Optional[str] = None
        i = 0
        while i < len(params):
            ch = params[i]
            if in_str:
                buf.append(ch)
                if ch == "\\" and i + 1 < len(params):
                    buf.append(params[i + 1])
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch in {"'", '"'}:
                in_str = ch
                buf.append(ch)
                i += 1
                continue
            if ch == "<":
                depth_angle += 1
            elif ch == ">":
                depth_angle = max(0, depth_angle - 1)
            if ch == "," and depth_angle == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    def _params_schema_from_java_signature(params: str) -> Dict[str, Any]:
        props: Dict[str, Any] = {}
        required: List[str] = []
        cleaned = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", params)
        for p in _split_java_params(cleaned):
            p = p.strip()
            if not p:
                continue
            p = re.sub(r"\bfinal\b\s+", "", p)
            tokens = [t for t in re.split(r"\s+", p) if t]
            if len(tokens) < 2:
                continue
            name = tokens[-1]
            type_str = " ".join(tokens[:-1])
            props[name] = {"type": _java_type_to_json_type(type_str), "description": ""}
            required.append(name)
        return {"type": "object", "properties": props, "required": required}

    for file in project_root.rglob("*.java"):
        if should_skip_path(file, excludes):
            continue
        try:
            src = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        # Spring AI / MCP annotations: @McpTool (optionally with name) preceding a method
        for m in anno_pattern.finditer(src):
            anno_args = (m.groupdict() or {}).get("args") or ""
            # Find the next method signature after the annotation occurrence
            tail = src[m.end():]
            mm = re.search(
                r"\b(?:public|protected|private)?\s*(?:static\s+)?[\w<>,\[\].\s]+\s+([A-Za-z_][\w]*)\s*\(([^)]*)\)",
                tail,
                re.MULTILINE,
            )
            method_name = mm.group(1) if mm else ""
            method_params = mm.group(2) if mm else ""

            name_m = re.search(r"\b(?:name|value)\s*=\s*\"([^\"]+)\"", anno_args)
            tool_name = name_m.group(1).strip() if name_m else ""
            if not tool_name and re.search(r"\"([^\"]+)\"", anno_args):
                tool_name = re.search(r"\"([^\"]+)\"", anno_args).group(1).strip()  # type: ignore[union-attr]
            if not tool_name:
                tool_name = method_name

            if tool_name:
                tools.append(
                    ToolDef(
                        name=tool_name,
                        description="",
                        parameters=_params_schema_from_java_signature(method_params) if method_params else {"type": "object", "properties": {}, "required": []},
                    )
                )

        for m in register_pattern.finditer(src):
            name = m.group(1)
            tools.append(ToolDef(name=name.strip(), description="", parameters={"type": "object", "properties": {}, "required": []}))

        # MCP Java SDK patterns: server.addTool(new Tool("name", "desc", ...))
        for m in add_tool_ctor_pattern.finditer(src):
            name = (m.group(1) or "").strip()
            desc = (m.group(2) or "").strip() if m.lastindex and m.lastindex >= 2 else ""
            if name:
                tools.append(ToolDef(name=name, description=desc, parameters={"type": "object", "properties": {}, "required": []}))

        # Tool.builder("name")...
        for m in tool_builder_pattern.finditer(src):
            name = (m.group(1) or "").strip()
            if name:
                tools.append(ToolDef(name=name, description="", parameters={"type": "object", "properties": {}, "required": []}))

    return tools


def detect_languages(project_root: Path) -> List[str]:
    langs: set[str] = set()
    if any(project_root.rglob("*.py")):
        langs.add("python")
    if (project_root / "package.json").exists() or any(project_root.rglob("*.ts")) or any(project_root.rglob("*.js")):
        langs.add("node")
    if any(project_root.rglob("pom.xml")) or any(project_root.rglob("build.gradle")) or any(project_root.rglob("*.java")):
        langs.add("java")
    return sorted(langs)


def merge_tools(existing: List[Dict[str, Any]], incoming: List[ToolDef]) -> List[Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for tool in existing:
        name = tool.get("function", {}).get("name")
        if name:
            by_name[name] = tool

    for t in incoming:
        if t.name not in by_name:
            by_name[t.name] = t.to_openai_tool()
        else:
            # Fill missing description/parameters if existing is empty
            existing_tool = by_name[t.name]
            fn = existing_tool.get("function", {})
            if not fn.get("description") and t.description:
                fn["description"] = t.description
            if not fn.get("parameters") and t.parameters:
                fn["parameters"] = t.parameters
            existing_tool["function"] = fn
            by_name[t.name] = existing_tool

    return [by_name[k] for k in sorted(by_name.keys())]


def generate_sample_for_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool["function"]
    name = fn["name"]
    desc = fn.get("description", "")
    params = fn.get("parameters") or {"type": "object", "properties": {}, "required": []}
    props = params.get("properties", {})
    required = params.get("required", [])

    args: Dict[str, Any] = {}
    for key in required:
        schema = props.get(key, {"type": "string"})
        args[key] = dummy_value_for_schema(schema)

    user = f"Use the tool `{name}` to help with: {desc}".strip()

    return {
        "id": f"auto-{name}-1",
        "tool_names": [name],
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]},
            {"role": "tool", "name": name, "content": json.dumps({"ok": True})},
            {"role": "assistant", "content": "Done."},
        ],
    }


def read_json_file(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, records: List[Dict[str, Any]]):
    existing_ids: set[str] = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "id" in obj:
                    existing_ids.add(str(obj["id"]))

    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            rec_id = str(rec.get("id", ""))
            if rec_id and rec_id in existing_ids:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Scan a project for MCP-registered tools and generate tools.json + finetune dataset JSONL.")
    parser.add_argument("--project", required=True, help="Path to the target project to scan")
    parser.add_argument("--tools-out", default="./data/tools.json", help="Path to tools.json to write/update")
    parser.add_argument("--dataset-out", default="./data/finetune_dataset.jsonl", help="Path to JSONL dataset to append to")
    parser.add_argument("--languages", default="auto", help="Comma-separated: python,node,java or 'auto'")
    parser.add_argument("--exclude", default=",".join(sorted(DEFAULT_EXCLUDES)), help="Comma-separated folder names to skip")
    parser.add_argument("--append-dataset", action="store_true", help="Append generated samples to dataset JSONL")
    parser.add_argument("--generate-samples", action="store_true", help="Generate one minimal training sample per discovered tool")
    args = parser.parse_args()

    project_root = Path(args.project).resolve()
    excludes = set([e.strip() for e in args.exclude.split(",") if e.strip()])

    if args.languages == "auto":
        langs = detect_languages(project_root)
    else:
        langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    discovered: List[ToolDef] = []
    if "python" in langs:
        discovered.extend(find_python_tools(project_root, excludes))
    if "node" in langs:
        discovered.extend(find_node_tools(project_root, excludes))
    if "java" in langs:
        discovered.extend(find_java_tools(project_root, excludes))

    tools_out = Path(args.tools_out)
    existing_tools = read_json_file(tools_out, default=[])
    merged_tools = merge_tools(existing_tools, discovered)

    tools_out.parent.mkdir(parents=True, exist_ok=True)
    with open(tools_out, "w", encoding="utf-8") as f:
        json.dump(merged_tools, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(merged_tools)} tools to: {tools_out}")

    if args.generate_samples:
        samples = [generate_sample_for_tool(t) for t in merged_tools]
        dataset_out = Path(args.dataset_out)
        dataset_out.parent.mkdir(parents=True, exist_ok=True)
        if args.append_dataset:
            append_jsonl(dataset_out, samples)
            print(f"Appended samples to: {dataset_out}")
        else:
            with open(dataset_out, "w", encoding="utf-8") as f:
                for rec in samples:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Wrote samples to: {dataset_out}")


if __name__ == "__main__":
    main()
