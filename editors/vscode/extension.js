// oxio VS Code companion. Runs a loopback MCP server and injects its port as OXIO_IDE_PORT into
// the integrated terminal, so the oxio CLI connects as an MCP client and receives the active
// selection over the `ide/contextUpdate` notification. Same shape as the Gemini CLI companion,
// so oxio reuses its existing MCP client. Requires `npm install` (the MCP SDK + express).

const vscode = require("vscode");
const express = require("express");
const os = require("node:os");
const path = require("node:path");
const fs = require("node:fs");
const { randomUUID } = require("node:crypto");
const { McpServer } = require("@modelcontextprotocol/sdk/server/mcp.js");
const {
  StreamableHTTPServerTransport,
} = require("@modelcontextprotocol/sdk/server/streamableHttp.js");
const { isInitializeRequest } = require("@modelcontextprotocol/sdk/types.js");
const { z } = require("zod");
// The extension's own version, reported in the MCP handshake so oxio can show which build is the
// LIVE one (a reload may lag the installed version - this is the authoritative "what's running").
const pkg = require("./package.json");

// Scheme for the diff's right pane (the proposed content), served from an in-memory map.
const DIFF_SCHEME = "oxio-diff";

// The active selection, or null when nothing is selected (oxio then sends the prompt as-is).
function currentSelection() {
  const ed = vscode.window.activeTextEditor;
  if (!ed || ed.selection.isEmpty) {
    return null;
  }
  const sel = ed.selection;
  return {
    file: vscode.workspace.asRelativePath(ed.document.uri),
    startLine: sel.start.line + 1, // editor lines are 0-based, humans count from 1
    endLine: sel.end.line + 1,
    text: ed.document.getText(sel),
  };
}

function activate(context) {
  const app = express();
  app.use(express.json({ limit: "10mb" }));
  const transports = {};
  // A FRESH MCP server per client session. The SDK binds one McpServer to ONE transport, so a
  // shared instance rejects the second client with "Already connected to a transport" - the first
  // connection works, every later one hangs. Each session gets its own server with the same tools.
  const makeMcpServer = () => {
    const mcp = new McpServer(
      { name: "oxio-ide", version: pkg.version },
      { capabilities: { logging: {} } },
    );
    registerTools(mcp);
    return mcp;
  };

  // Edit approval: oxio calls the `openDiff` tool, the user accepts or rejects in a native diff,
  // and the tool resolves with `{ accepted }`. The right pane is a virtual doc holding the
  // proposed content; accept writes it to the real file.
  const proposed = new Map(); // right-pane uri string -> proposed content
  const pending = new Map(); // right-pane uri string -> { resolve, fileUri, newContent }
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(DIFF_SCHEME, {
      provideTextDocumentContent: (uri) => proposed.get(uri.toString()) ?? "",
    }),
  );

  async function openDiff(filePath, newContent) {
    const abs = path.isAbsolute(filePath)
      ? filePath
      : path.join(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? "", filePath);
    const fileUri = vscode.Uri.file(abs);
    const rightUri = vscode.Uri.parse(`${DIFF_SCHEME}:${abs}`);
    proposed.set(rightUri.toString(), newContent);
    await vscode.commands.executeCommand(
      "vscode.diff",
      fileUri,
      rightUri,
      `oxio: ${path.basename(abs)} (proposed - accept or reject)`,
    );
    return new Promise((resolve) => {
      pending.set(rightUri.toString(), { resolve });
    });
  }

  async function finishDiff(accepted) {
    const uri = vscode.window.activeTextEditor?.document.uri;
    if (!uri || uri.scheme !== DIFF_SCHEME) return;
    const key = uri.toString();
    const p = pending.get(key);
    if (!p) return;
    pending.delete(key);
    proposed.delete(key);
    // Only report the decision - oxio performs the actual write after approval, so the diff never
    // touches the file itself (writing here would double-apply an apply_patch and break it).
    await vscode.commands.executeCommand("workbench.action.closeActiveEditor");
    p.resolve(accepted);
  }
  context.subscriptions.push(
    vscode.commands.registerCommand("oxio.diff.accept", () => finishDiff(true)),
    vscode.commands.registerCommand("oxio.diff.cancel", () => finishDiff(false)),
  );

  function registerTools(mcp) {
  mcp.registerTool(
    "openDiff",
    {
      description: "Open a diff for the user to accept or reject a file change. Resolves with { accepted }.",
      inputSchema: z.object({ filePath: z.string(), newContent: z.string() }).shape,
    },
    async ({ filePath, newContent }) => {
      const accepted = await openDiff(filePath, newContent);
      return {
        content: [{ type: "text", text: accepted ? "accepted" : "rejected" }],
        structuredContent: { accepted },
      };
    },
  );

  // Pull the current selection at submit (oxio calls this when you send). {file, startLine,
  // endLine, text} or null. Pull, not push - no race with a change event or a late connection.
  mcp.registerTool(
    "getSelection",
    {
      description: "Return the active editor selection right now, or null when nothing is selected.",
      inputSchema: {},
    },
    async () => {
      const sel = currentSelection();
      return {
        content: [
          { type: "text", text: sel ? `${sel.file}:${sel.startLine}-${sel.endLine}` : "no selection" },
        ],
        structuredContent: { selection: sel },
      };
    },
  );

  // Dismiss the open diff when the user answered in the CLI instead. Resolves any pending
  // openDiff so its call unblocks, then closes the editor. A no-op when nothing is open, so it
  // never closes an unrelated editor. This is the CLI-wins half of the two-surface approval.
  mcp.registerTool(
    "closeDiff",
    {
      description: "Close the open oxio diff without a decision (the user answered in the CLI).",
    },
    async () => {
      if (pending.size === 0) {
        return { content: [{ type: "text", text: "no diff" }] };
      }
      for (const [key, p] of pending) {
        pending.delete(key);
        proposed.delete(key);
        p.resolve(false);
      }
      // Close the oxio-diff tab wherever it lives - the user answered in the CLI, so focus is in
      // the terminal and the diff is NOT the active editor (closeActiveEditor would miss it or
      // close the wrong tab). Match the tab by the diff scheme on its modified side.
      try {
        for (const group of vscode.window.tabGroups.all) {
          for (const tab of group.tabs) {
            const modified = tab.input && tab.input.modified;
            if (modified && modified.scheme === DIFF_SCHEME) {
              await vscode.window.tabGroups.close(tab);
            }
          }
        }
      } catch (_e) {
        await vscode.commands.executeCommand("workbench.action.closeActiveEditor");
      }
      return { content: [{ type: "text", text: "closed" }] };
    },
  );
  }

  app.post("/mcp", async (req, res) => {
    const sid = req.headers["mcp-session-id"];
    let transport;
    if (sid && transports[sid]) {
      transport = transports[sid];
    } else if (!sid && isInitializeRequest(req.body)) {
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (id) => {
          transports[id] = transport;
        },
      });
      transport.onclose = () => {
        if (transport.sessionId) delete transports[transport.sessionId];
      };
      await makeMcpServer().connect(transport);
    } else {
      res.status(400).json({ jsonrpc: "2.0", error: { code: -32000, message: "no session" }, id: null });
      return;
    }
    await transport.handleRequest(req, res, req.body);
  });

  const sentInitial = new Set();
  const handleSession = async (req, res) => {
    const sid = req.headers["mcp-session-id"];
    if (!sid || !transports[sid]) {
      res.status(400).send("no session");
      return;
    }
    await transports[sid].handleRequest(req, res);
    // Push the CURRENT selection once when a client's stream is up, so a selection made BEFORE
    // oxio connected is delivered - not only later changes. Without this a static selection never
    // reaches oxio.
    if (!sentInitial.has(sid)) {
      sentInitial.add(sid);
      try {
        transports[sid].send({
          jsonrpc: "2.0",
          method: "ide/contextUpdate",
          params: { selection: currentSelection() },
        });
      } catch (_e) {
        sentInitial.delete(sid);
      }
    }
  };
  app.get("/mcp", handleSession);

  // Push the current selection to every connected oxio on any selection or active-editor change.
  function broadcast() {
    const note = {
      jsonrpc: "2.0",
      method: "ide/contextUpdate",
      params: { selection: currentSelection() },
    };
    for (const t of Object.values(transports)) {
      try {
        t.send(note);
      } catch (_e) {
        // a dropped client is cleaned up on its own onclose
      }
    }
  }
  context.subscriptions.push(
    vscode.window.onDidChangeTextEditorSelection(broadcast),
    vscode.window.onDidChangeActiveTextEditor(broadcast),
  );

  const server = app.listen(0, "127.0.0.1", () => {
    const port = server.address().port;
    // Publish the port to new integrated terminals, and to a port file so a terminal opened
    // before activation can still find it.
    context.environmentVariableCollection.replace("OXIO_IDE_PORT", String(port));
    try {
      fs.writeFileSync(path.join(os.tmpdir(), `oxio-ide-${process.ppid}.json`), JSON.stringify({ port }));
    } catch (_e) {
      // best-effort: the env var is the primary discovery path
    }
  });
  context.subscriptions.push({ dispose: () => server.close() });
}

function deactivate() {}

module.exports = { activate, deactivate };
