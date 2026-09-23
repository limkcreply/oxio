# oxio VS Code companion

Connects the oxio CLI running in VS Code's integrated terminal to the editor, so oxio can read
what you have selected and show file changes as a native diff.

## What it does

**Selection as context.** Select code, send a prompt in oxio, and the selection goes with it,
shown as a `Selected 12 lines from src/main.rs` chip. Nothing selected, or no extension running,
sends the prompt unchanged.

**Diff approval.** When oxio wants to change a file it opens a VS Code diff with the proposed
content on the right. Accept with `cmd+enter`, reject with `escape`, or use the check and cross
buttons in the editor title bar. You can also answer the `y/N` prompt in the terminal instead;
whichever surface you answer first wins and the other one closes.

The extension never writes to your files. It reports your decision and oxio performs the write,
so an approved patch is applied once.

## Install

The extension is bundled inside the oxio binary. Run oxio in VS Code's integrated terminal and it
offers to install it on the first run, then asks you to reload the window. That needs the `code`
command on your PATH.

## How it works

The extension runs a loopback MCP server (Streamable HTTP) and injects its port as
`OXIO_IDE_PORT` into the integrated terminal. oxio connects as an MCP client, so it reuses its
existing MCP client rather than carrying a second protocol. A terminal opened before the
extension activated finds the port in a `oxio-ide-*.json` file in the temp dir instead.

Tools served to oxio:

- `getSelection` returns the selection right now, pulled at submit so there is no race with a
  change event or a late connection.
- `openDiff` opens the diff and resolves with `{ accepted }`.
- `closeDiff` dismisses the diff when the user answered in the CLI.

It also pushes `ide/contextUpdate` with `{ "selection": { "file", "startLine", "endLine",
"text" } }` on every selection change, and once when a client connects so a selection made
before oxio started is still delivered. `selection` is null when nothing is selected.

## Install from source

For working on the extension itself. Requires Node dependencies (the MCP SDK and express).

1. Install deps in this folder: `npm install`
2. Symlink or copy this folder into your extensions dir:
   `ln -s "$PWD" ~/.vscode/extensions/oxio`
3. Reload VS Code (Developer: Reload Window).
4. Open a new integrated terminal (existing ones need a reload to pick up `OXIO_IDE_PORT`).
5. Select code, launch `oxio`, send a prompt.

To build the `.vsix` the release pipeline embeds: `npx @vscode/vsce package -o oxio.vsix`.

## Scope

The MCP server binds to `127.0.0.1` only and serves the current selection and the diff decision
to local clients. It exposes nothing else.
