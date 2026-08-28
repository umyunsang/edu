import { spawn } from "node:child_process";
import { Readable, Writable } from "node:stream";
import * as acp from "/Users/um-yunsang/.npm/_npx/c8b015f66c7988d7/node_modules/@agentclientprotocol/sdk/dist/acp.js";

const child = spawn(
  "/opt/homebrew/bin/npx",
  ["-y", "@agentclientprotocol/codex-acp@1.7.0"],
  {
    cwd: "/Users/um-yunsang/.ok/acp-npx-cwd",
    stdio: ["pipe", "pipe", "pipe"],
  },
);

let stderr = "";
child.stderr.setEncoding("utf8");
child.stderr.on("data", (chunk) => {
  stderr += chunk;
});

const timeout = setTimeout(() => {
  child.kill("SIGTERM");
  process.stderr.write(`ACP initialize timeout\n${stderr}`);
  process.exitCode = 1;
}, 20_000);

try {
  const stream = acp.ndJsonStream(
    Writable.toWeb(child.stdin),
    Readable.toWeb(child.stdout),
  );
  await acp.client({ name: "openknowledge-recovery-check", version: "1.0.0" })
    .connectWith(stream, async (connection) => {
      const result = await connection.request(acp.methods.agent.initialize, {
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
      });
      process.stdout.write(`${JSON.stringify(result)}\n`);
    });
} finally {
  clearTimeout(timeout);
  child.kill("SIGTERM");
}
