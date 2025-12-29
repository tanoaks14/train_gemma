import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# Allow importing scan_mcp_tools.py from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scan_mcp_tools  # noqa: E402


class ScanMcpToolsTests(unittest.TestCase):
    def _write(self, root: Path, rel: str, content: str) -> Path:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_python_fastmcp_tool_decorators(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "app.py",
                """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('x')

@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

@mcp.tool('sum_two')
def sum_two(x: float, y: float = 1.0) -> float:
    \"\"\"Sum two floats.\"\"\"
    return x + y
""".strip(),
            )

            tools = scan_mcp_tools.find_python_tools(root, excludes=set())
            by_name = {t.name: t for t in tools}

            self.assertIn("add", by_name)
            self.assertIn("sum_two", by_name)

            add = by_name["add"].to_openai_tool()["function"]
            self.assertEqual(add["description"], "Add two numbers.")
            self.assertEqual(add["parameters"]["type"], "object")
            self.assertEqual(set(add["parameters"]["properties"].keys()), {"a", "b"})
            self.assertEqual(set(add["parameters"]["required"]), {"a", "b"})

            sum_two = by_name["sum_two"].to_openai_tool()["function"]
            self.assertEqual(sum_two["description"], "Sum two floats.")
            self.assertIn("x", sum_two["parameters"]["properties"])
            self.assertIn("y", sum_two["parameters"]["properties"])
            # y has a default -> should not be required
            self.assertIn("x", sum_two["parameters"]["required"])
            self.assertNotIn("y", sum_two["parameters"]["required"])

    def test_node_registertool_extracts_description_and_inputschema_object(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "index.ts",
                """
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

const server = new McpServer({ name: 'demo', version: '1.0.0' });

server.registerTool(
  "weather",
  {
    description: "Get the weather",
    inputSchema: {
      "type": "object",
      "properties": { "city": { "type": "string" } },
      "required": ["city"]
    }
  },
  async (args) => {
    return { content: [{ type: 'text', text: 'ok' }] };
  }
);
""".strip(),
            )

            tools = scan_mcp_tools.find_node_tools(root, excludes=set())
            by_name = {t.name: t for t in tools}

            self.assertIn("weather", by_name)
            weather = by_name["weather"].to_openai_tool()["function"]
            self.assertEqual(weather["description"], "Get the weather")
            self.assertEqual(weather["parameters"].get("type"), "object")
            # inputSchema should be propagated as parameters (best-effort)
            self.assertIn("city", weather["parameters"].get("properties", {}))
            self.assertIn("city", weather["parameters"].get("required", []))

    def test_node_registertool_extracts_zod_object_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "tools.ts",
                """
import { z } from 'zod';

registerTool(
  "search",
  {
    description: "Search docs",
    inputSchema: z.object({ query: z.string(), limit: z.number() })
  },
  async (args) => ({ ok: true })
);
""".strip(),
            )

            tools = scan_mcp_tools.find_node_tools(root, excludes=set())
            by_name = {t.name: t for t in tools}

            self.assertIn("search", by_name)
            search = by_name["search"].to_openai_tool()["function"]
            self.assertEqual(search["description"], "Search docs")
            props = search["parameters"].get("properties", {})
            self.assertIn("query", props)
            self.assertIn("limit", props)
            self.assertEqual(set(search["parameters"].get("required", [])), {"query", "limit"})

    def test_java_mcptool_method_params(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "src/main/java/com/example/Tools.java",
                """
package com.example;

import org.springframework.ai.mcp.server.annotation.McpTool;

public class Tools {

  @McpTool(name = \"echo\")
  public String echo(String text, int times) {
    return text;
  }

  @McpTool
  public void ping() {
  }
}
""".strip(),
            )

            tools = scan_mcp_tools.find_java_tools(root, excludes=set())
            by_name = {t.name: t for t in tools}

            self.assertIn("echo", by_name)
            echo = by_name["echo"].to_openai_tool()["function"]
            props = echo["parameters"].get("properties", {})
            self.assertIn("text", props)
            self.assertIn("times", props)
            self.assertIn("text", echo["parameters"].get("required", []))
            self.assertIn("times", echo["parameters"].get("required", []))

            self.assertIn("ping", by_name)
            ping = by_name["ping"].to_openai_tool()["function"]
            self.assertEqual(ping["parameters"].get("properties", {}), {})

    def test_java_addtool_constructor_patterns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "Server.java",
                """
public class Server {
  public void init(Object server) {
    server.addTool(new Tool(\"calc\", \"Calculate\", (args) -> null));
    server.addTool(new SomeToolImpl(\"other\"));
    Tool.builder(\"built\").description(\"x\");
  }
}
""".strip(),
            )

            tools = scan_mcp_tools.find_java_tools(root, excludes=set())
            names = {t.name for t in tools}
            self.assertIn("calc", names)
            self.assertIn("built", names)

    def test_merge_tools_fills_missing_fields(self):
        existing = [
            {
                "type": "function",
                "function": {"name": "t1", "description": "", "parameters": None},
            }
        ]
        incoming = [
            scan_mcp_tools.ToolDef(
                name="t1",
                description="desc",
                parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            )
        ]
        merged = scan_mcp_tools.merge_tools(existing, incoming)
        self.assertEqual(len(merged), 1)
        fn = merged[0]["function"]
        self.assertEqual(fn["description"], "desc")
        self.assertEqual(fn["parameters"]["type"], "object")

    def test_integration_dry_run_scans_all_languages_and_writes_tools_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Python
            self._write(
                root,
                "py/app.py",
                """
class X:
  pass

@mcp.tool('py_tool')
def py_tool(q: str):
  \"\"\"Python tool desc.\"\"\"
  return q
""".strip(),
            )
            # Node
            self._write(
                root,
                "node/index.js",
                """
registerTool('js_tool', { description: 'JS tool desc', inputSchema: {"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}}, () => {});
""".strip(),
            )
            # Java
            self._write(
                root,
                "java/Tools.java",
                """
import org.springframework.ai.mcp.server.annotation.McpTool;

public class Tools {
  @McpTool
  public String java_tool(String q) { return q; }
}
""".strip(),
            )

            out = root / "tools_out.json"
            # Run scanner script as a subprocess using the current interpreter
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scan_mcp_tools.py"),
                "--project",
                str(root),
                "--tools-out",
                str(out),
                "--languages",
                "auto",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsInstance(data, list)
            # Ensure it wrote tools in OpenAI shape
            for tool in data:
                self.assertEqual(tool.get("type"), "function")
                self.assertIn("function", tool)
                self.assertIn("name", tool["function"])
                self.assertIn("parameters", tool["function"])


if __name__ == "__main__":
    unittest.main()
