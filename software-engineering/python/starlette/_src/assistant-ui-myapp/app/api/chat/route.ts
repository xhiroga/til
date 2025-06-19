import { openai } from "@ai-sdk/openai";
import { experimental_createMCPClient, streamText } from "ai";
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';


export const runtime = "edge";
export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages, system } = await req.json();

  let tools = {};

  // https://ai-sdk.dev/cookbook/next/mcp-tools
  try {
    const transport = new StreamableHTTPClientTransport(
      new URL('http://localhost:8000/mcp/'),
    );
    const client = await experimental_createMCPClient({ transport })
    tools = await client.tools()
  } catch (error) {
    console.error(error);
  }

  const result = streamText({
    model: openai("gpt-4o"),
    messages,
    // forward system prompt and tools from the frontend
    system,
    tools,
  });

  return result.toDataStreamResponse();
}
