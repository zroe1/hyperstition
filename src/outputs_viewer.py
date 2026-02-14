#!/usr/bin/env python3
"""Tiny web app for browsing model responses in outputs/."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"


def _safe_relpath(path: Path, base: Path) -> str:
    return str(path.relative_to(base)).replace(os.sep, "/")


def list_output_files() -> list[dict[str, Any]]:
    if not OUTPUTS_DIR.exists():
        return []

    files: list[dict[str, Any]] = []
    for path in OUTPUTS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue

        try:
            stat = path.stat()
        except OSError:
            continue

        files.append(
            {
                "path": _safe_relpath(path, REPO_ROOT),
                "name": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )

    files.sort(key=lambda item: item["mtime"], reverse=True)
    return files


def _decode_possible_unicode_escapes(text: str) -> str:
    # JSON parsing usually turns \uXXXX into real characters already.
    # This fallback handles double-escaped sequences like "\\u2764\\ufe0f".
    if "\\u" not in text and "\\U" not in text:
        return text
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return text


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _decode_possible_unicode_escapes(value)
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def _parse_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                rows.append({"_parse_error": str(exc), "_raw_line": raw})
    return rows


def _load_file_payload(relative_path: str) -> dict[str, Any]:
    target = (REPO_ROOT / relative_path).resolve()
    if not str(target).startswith(str(OUTPUTS_DIR.resolve())):
        raise PermissionError("Path must stay inside outputs/")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(relative_path)

    suffix = target.suffix.lower()
    if suffix == ".json":
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix == ".jsonl":
        data = _parse_jsonl(target)
    else:
        raise ValueError("Unsupported file type")

    return {
        "path": _safe_relpath(target, REPO_ROOT),
        "type": suffix[1:],
        "data": _normalize_value(data),
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hyperstition Outputs Viewer</title>
  <style>
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: #ffffff;
      color: #1f2937;
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }
    aside { border-right: 1px solid #e5e7eb; padding: 12px; overflow: auto; background: #f9fafb; }
    main { padding: 14px; overflow: auto; background: #ffffff; }
    h1 { margin: 0 0 10px; font-size: 18px; }
    #search {
      width: 100%; box-sizing: border-box; margin-bottom: 10px;
      padding: 9px 10px; border: 1px solid #d1d5db; border-radius: 8px;
      background: #ffffff; color: #111827;
    }
    #tree { font-size: 13px; line-height: 1.4; }
    .tree-row {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 5px 6px;
      border-radius: 6px;
      margin: 1px 0;
      user-select: none;
    }
    .tree-row:hover { background: #f3f4f6; }
    .tree-row.file { cursor: pointer; }
    .tree-row.file.active { background: #eaf2ff; outline: 1px solid #c8dcff; }
    .caret {
      width: 14px;
      color: #6b7280;
      text-align: center;
      cursor: pointer;
      font-size: 11px;
      flex-shrink: 0;
    }
    .caret-placeholder {
      width: 14px;
      flex-shrink: 0;
    }
    .folder-name { color: #374151; font-weight: 600; }
    .file-name { color: #1f2937; word-break: break-all; }
    .file-size { margin-left: auto; color: #6b7280; font-size: 11px; }
    .status { font-size: 13px; color: #4b5563; margin-bottom: 12px; }
    .example {
      border: 1px solid #e5e7eb; border-radius: 10px;
      background: #ffffff; padding: 11px; margin-bottom: 11px;
    }
    .example h3 { margin: 0 0 8px; font-size: 14px; color: #111827; }
    .role {
      font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
      color: #4f46e5; margin-bottom: 3px;
    }
    .content {
      white-space: pre-wrap; line-height: 1.4; margin-bottom: 10px;
      font-size: 14px; word-break: break-word;
    }
    .content:last-child { margin-bottom: 0; }
    pre {
      background: #f9fafb; border: 1px solid #e5e7eb; padding: 10px;
      border-radius: 8px; overflow: auto; white-space: pre-wrap;
    }
    .hint { color: #6b7280; font-size: 14px; }
  </style>
</head>
<body>
  <aside>
    <h1>Outputs Viewer</h1>
    <input id="search" placeholder="Filter files..." />
    <div id="tree"></div>
  </aside>
  <main>
    <div class="status" id="status">Loading files...</div>
    <div id="content"><div class="hint">Select a file to view.</div></div>
  </main>
  <script>
    const state = { files: [], selectedPath: null, expandedDirs: new Set(["outputs"]) };

    const treeEl = document.getElementById("tree");
    const contentEl = document.getElementById("content");
    const searchEl = document.getElementById("search");
    const statusEl = document.getElementById("status");

    function fmtSize(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    function normalizeText(text) {
      if (typeof text !== "string") return String(text ?? "");
      return text;
    }

    function buildTree(files) {
      const root = { type: "dir", name: "outputs", path: "outputs", dirs: new Map(), files: [] };
      for (const file of files) {
        const parts = file.path.split("/");
        let node = root;
        for (let i = 1; i < parts.length - 1; i++) {
          const dirName = parts[i];
          const dirPath = parts.slice(0, i + 1).join("/");
          if (!node.dirs.has(dirName)) {
            node.dirs.set(dirName, { type: "dir", name: dirName, path: dirPath, dirs: new Map(), files: [] });
          }
          node = node.dirs.get(dirName);
        }
        node.files.push({ ...file, name: parts[parts.length - 1] });
      }
      return root;
    }

    function hasMatch(node, query) {
      if (!query) return true;
      if (node.type === "file") return node.path.toLowerCase().includes(query);
      for (const child of node.dirs.values()) {
        if (hasMatch(child, query)) return true;
      }
      return node.files.some(file => file.path.toLowerCase().includes(query));
    }

    function renderTreeNode(node, depth, query) {
      if (!hasMatch(node, query)) return [];
      const rows = [];

      if (node.type === "dir") {
        const isExpanded = query ? true : state.expandedDirs.has(node.path);
        const row = document.createElement("div");
        row.className = "tree-row dir";
        row.style.paddingLeft = `${depth * 14 + 4}px`;

        const caret = document.createElement("span");
        caret.className = "caret";
        caret.textContent = isExpanded ? "▾" : "▸";
        caret.addEventListener("click", (e) => {
          e.stopPropagation();
          if (state.expandedDirs.has(node.path)) state.expandedDirs.delete(node.path);
          else state.expandedDirs.add(node.path);
          renderFiles();
        });

        const name = document.createElement("span");
        name.className = "folder-name";
        name.textContent = node.name;

        row.appendChild(caret);
        row.appendChild(name);
        rows.push(row);

        if (!isExpanded) return rows;

        const childDirs = [...node.dirs.values()].sort((a, b) => a.name.localeCompare(b.name));
        const childFiles = [...node.files].sort((a, b) => a.name.localeCompare(b.name));

        for (const dir of childDirs) {
          rows.push(...renderTreeNode(dir, depth + 1, query));
        }
        for (const file of childFiles) {
          if (query && !file.path.toLowerCase().includes(query)) continue;
          const fileRow = document.createElement("div");
          fileRow.className = "tree-row file" + (file.path === state.selectedPath ? " active" : "");
          fileRow.style.paddingLeft = `${(depth + 1) * 14 + 4}px`;

          const placeholder = document.createElement("span");
          placeholder.className = "caret-placeholder";
          fileRow.appendChild(placeholder);

          const nameEl = document.createElement("span");
          nameEl.className = "file-name";
          nameEl.textContent = file.name;
          fileRow.appendChild(nameEl);

          const sizeEl = document.createElement("span");
          sizeEl.className = "file-size";
          sizeEl.textContent = fmtSize(file.size);
          fileRow.appendChild(sizeEl);

          fileRow.addEventListener("click", () => loadFile(file.path));
          rows.push(fileRow);
        }
      }

      return rows;
    }

    function extractMessages(entry) {
      if (entry && Array.isArray(entry.messages)) {
        return entry.messages
          .filter(m => m && typeof m === "object")
          .map(m => ({ role: normalizeText(m.role), content: normalizeText(m.content) }));
      }
      return [];
    }

    function renderGeneric(entry) {
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(entry, null, 2);
      return pre;
    }

    function renderEntry(entry, idx) {
      const card = document.createElement("div");
      card.className = "example";
      const title = document.createElement("h3");
      title.textContent = `Example ${idx + 1}`;
      card.appendChild(title);

      const messages = extractMessages(entry);
      if (messages.length > 0) {
        for (const msg of messages) {
          const role = document.createElement("div");
          role.className = "role";
          role.textContent = msg.role || "unknown";
          const content = document.createElement("div");
          content.className = "content";
          content.textContent = msg.content || "";
          card.appendChild(role);
          card.appendChild(content);
        }
      } else {
        card.appendChild(renderGeneric(entry));
      }

      return card;
    }

    function renderPayload(payload) {
      contentEl.innerHTML = "";
      const data = payload.data;

      if (Array.isArray(data)) {
        if (data.length === 0) {
          contentEl.innerHTML = '<div class="hint">File is empty.</div>';
          return;
        }
        data.forEach((entry, idx) => contentEl.appendChild(renderEntry(entry, idx)));
        return;
      }

      if (data && typeof data === "object") {
        const single = renderEntry(data, 0);
        contentEl.appendChild(single);
        return;
      }

      const pre = document.createElement("pre");
      pre.textContent = normalizeText(data);
      contentEl.appendChild(pre);
    }

    function renderFiles() {
      const q = searchEl.value.trim().toLowerCase();
      const files = state.files.filter(f => !q || f.path.toLowerCase().includes(q));
      treeEl.innerHTML = "";

      const root = buildTree(state.files);
      const rows = renderTreeNode(root, 0, q);
      for (const row of rows) treeEl.appendChild(row);

      statusEl.textContent = q ? `${files.length} matching files` : `${state.files.length} files`;
    }

    async function loadFiles() {
      const res = await fetch("/api/files");
      const files = await res.json();
      state.files = files;
      renderFiles();
    }

    async function loadFile(path) {
      state.selectedPath = path;
      renderFiles();
      statusEl.textContent = `Loading ${path}...`;

      const res = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
      if (!res.ok) {
        contentEl.innerHTML = `<div class="hint">Failed to load ${path}</div>`;
        statusEl.textContent = `Error loading ${path}`;
        return;
      }

      const payload = await res.json();
      renderPayload(payload);
      statusEl.textContent = payload.path;
    }

    searchEl.addEventListener("input", renderFiles);
    loadFiles().catch(err => {
      statusEl.textContent = "Failed to list files";
      contentEl.innerHTML = `<pre>${String(err)}</pre>`;
    });
  </script>
</body>
</html>
"""


class OutputsViewerHandler(BaseHTTPRequestHandler):
    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, payload: str, status: int = HTTPStatus.OK) -> None:
        body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(HTML)
            return

        if parsed.path == "/api/files":
            self._json(list_output_files())
            return

        if parsed.path == "/api/file":
            rel_path = parse_qs(parsed.query).get("path", [""])[0]
            if not rel_path:
                self._json({"error": "Missing ?path="}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = _load_file_payload(rel_path)
            except FileNotFoundError:
                self._json({"error": f"File not found: {rel_path}"}, status=HTTPStatus.NOT_FOUND)
                return
            except PermissionError as exc:
                self._json({"error": str(exc)}, status=HTTPStatus.FORBIDDEN)
                return
            except (json.JSONDecodeError, ValueError) as exc:
                self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            self._json(payload)
            return

        self._json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output clean.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="View outputs/*.json and *.jsonl in browser")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8008, help="Port to bind (default: 8008)")
    args = parser.parse_args()

    if not OUTPUTS_DIR.exists():
        raise SystemExit(f"Missing outputs directory: {OUTPUTS_DIR}")

    server = ThreadingHTTPServer((args.host, args.port), OutputsViewerHandler)
    print(f"Serving outputs viewer at http://{args.host}:{args.port}")
    print(f"Scanning: {OUTPUTS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
