# oxio VS Code companion

Sends the active editor selection to the oxio CLI running in VS Code's integrated terminal.
Select code, send a prompt in oxio, and the selection is sent as context, shown as a
`Selected N lines from FILE` chip.

## How it works

The extension runs a loopback MCP server (Streamable HTTP) and injects its port as
`OXIO_IDE_PORT` into the integrated terminal. oxio connects as an MCP client and receives the
active selection over the `ide/contextUpdate` notification, prepending it to the next prompt.
Nothing selected, or no extension, sends the prompt unchanged. Same shape as the Gemini CLI
companion, so oxio reuses its existing MCP client.

Notification: `ide/contextUpdate` with `{ "selection": { "file", "startLine", "endLine",
"text" } }`; `selection` is null when nothing is selected.

## Install (development)

Requires Node dependencies (the MCP SDK and express).

1. Install deps in this folder: `npm install`
2. Symlink or copy this folder into your extensions dir:
   `ln -s "$PWD" ~/.vscode/extensions/oxio`
3. Reload VS Code (Developer: Reload Window).
4. Open a new integrated terminal (existing ones need a reload to pick up `OXIO_IDE_PORT`).
5. Select code, launch `oxio`, send a prompt.

## Scope

The MCP server binds to `127.0.0.1` only and serves the current selection to local clients.
It exposes nothing else.
