#!/usr/bin/env node

/**
 * This is a template MCP server that implements a simple notes system.
 * It demonstrates core MCP concepts like resources and tools by allowing:
 * - Listing notes as resources
 * - Reading individual notes
 * - Creating new notes via a tool
 * - Summarizing all notes via a prompt
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url'; // Import necessary function
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

/**
 * Type alias for a note object.
 */
type Note = { title: string, content: string };

// Get the directory name in ES module scope
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Define the path to the scenes directory relative to the server's execution directory
const scenesDir = path.resolve(__dirname, '../../assets/scenes');

// Note type alias removed as it's duplicated from line 26 and not currently used for scenes.

/**
 * Create an MCP server with capabilities for resources (to list/read notes),
 * tools (to create new notes), and prompts (to summarize notes).
 */
const server = new Server(
  {
    name: "mcp-server",
    version: "0.1.0",
  },
  {
    capabilities: {
      resources: {},
      tools: {},
      prompts: {},
    },
  }
);

/**
 * Handler for listing available scene files as resources.
 * Each .scene file is exposed as a resource with:
 * - A scene:// URI scheme
 * - JSON MIME type (assuming .scene files are JSON-like)
 * - Human readable name and description
 */
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  try {
    const files = await fs.promises.readdir(scenesDir);
    const sceneFiles = files.filter(file => file.endsWith('.scene') && !file.endsWith('.meta'));

    return {
      resources: sceneFiles.map(fileName => ({
        uri: `scene:///${fileName}`,
        // Using application/json as scene files are JSON-based
        mimeType: "application/json",
        name: fileName,
        description: `Cocos Creator scene file: ${fileName}`
      }))
    };
  } catch (error) {
    console.error("Error listing scene resources:", error);
    // Return empty list or throw an error appropriate for MCP
    return { resources: [] };
  }
});

/**
 * Handler for reading the contents of a specific scene file.
 * Takes a scene:// URI and returns the scene content.
 */
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const url = new URL(request.params.uri);
  if (url.protocol !== 'scene:') {
    throw new Error(`Unsupported URI scheme: ${url.protocol}`);
  }
  const fileName = url.pathname.replace(/^\//, '');
  const filePath = path.join(scenesDir, fileName);

  try {
    // Ensure the file is within the intended directory
    if (!filePath.startsWith(scenesDir)) {
        throw new Error(`Access denied to path: ${filePath}`);
    }
    // Check if file exists before reading
    await fs.promises.access(filePath, fs.constants.R_OK);

    const content = await fs.promises.readFile(filePath, 'utf-8');

    return {
      contents: [{
        uri: request.params.uri,
        // Using application/json as scene files are JSON-based
        mimeType: "application/json",
        text: content
      }]
    };
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      throw new Error(`Scene file not found: ${fileName}`);
    } else if (error.code === 'EACCES') {
       throw new Error(`Permission denied reading file: ${fileName}`);
    }
    console.error(`Error reading scene resource ${fileName}:`, error);
    throw new Error(`Failed to read scene file: ${fileName}`);
  }
});

/**
 * Handler that lists available tools.
 * Exposes:
 * - "create_note" tool (kept for reference, could be removed if only scenes are needed)
 * - "update_scene" tool for modifying scene files.
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      // { // Keep or remove the note tool as needed
      //   name: "create_note",
      //   description: "Create a new note",
      //   inputSchema: {
      //     type: "object",
      //     properties: {
      //       title: { type: "string", description: "Title of the note" },
      //       content: { type: "string", description: "Text content of the note" }
      //     },
      //     required: ["title", "content"]
      //   }
      // },
      {
        name: "update_scene",
        description: "Update the content of a scene file",
        inputSchema: {
          type: "object",
          properties: {
            uri: {
              type: "string",
              description: "The scene:// URI of the file to update",
              pattern: "^scene:///.+\\.scene$" // Basic validation for scene URI
            },
            content: {
              type: "string",
              description: "The new JSON content for the scene file"
            }
          },
          required: ["uri", "content"]
        }
      }
    ]
  };
});

/**
 * Handler for tool calls. Handles 'create_note' and 'update_scene'.
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  switch (request.params.name) {
    // case "create_note": { // Keep or remove matching the ListTools handler
    //   const title = String(request.params.arguments?.title);
    //   const content = String(request.params.arguments?.content);
    //   if (!title || !content) {
    //     throw new Error("Title and content are required");
    //   }
    //   // Note: 'notes' variable was removed, this part needs adjustment
    //   // if note creation is kept. For now, assume it's removed or handled differently.
    //   // const id = String(Object.keys(notes).length + 1);
    //   // notes[id] = { title, content };
    //   console.warn("Note creation logic needs update as 'notes' is removed.");
    //   return { content: [{ type: "text", text: `Note creation needs update.` }] };
    // }

    case "update_scene": {
      const uri = String(request.params.arguments?.uri);
      const content = String(request.params.arguments?.content);

      if (!uri || content === undefined) { // Check content for undefined specifically
        throw new Error("URI and content are required for update_scene");
      }

      let url: URL;
      try {
        url = new URL(uri);
      } catch (e) {
        throw new Error(`Invalid URI format: ${uri}`);
      }


      if (url.protocol !== 'scene:') {
        throw new Error(`Unsupported URI scheme for update: ${url.protocol}`);
      }
      const fileName = url.pathname.replace(/^\//, '');
      if (!fileName.endsWith('.scene')) {
         throw new Error(`Invalid file extension in URI: ${fileName}`);
      }
      const filePath = path.join(scenesDir, fileName);

      try {
        // Security check: Ensure the resolved path is still within the scenes directory
        if (!filePath.startsWith(scenesDir)) {
          throw new Error(`Access denied: Attempt to write outside designated directory.`);
        }

        // Validate JSON content before writing (optional but recommended)
        try {
          JSON.parse(content);
        } catch (jsonError) {
          throw new Error(`Invalid JSON content provided for ${fileName}`);
        }


        await fs.promises.writeFile(filePath, content, 'utf-8');

        return {
          content: [{
            type: "text",
            text: `Successfully updated scene file: ${fileName}`
          }]
        };
      } catch (error: any) {
         if (error.code === 'ENOENT') {
           throw new Error(`Scene file not found for update: ${fileName}`);
         } else if (error.code === 'EACCES') {
            throw new Error(`Permission denied writing file: ${fileName}`);
         }
        console.error(`Error updating scene file ${fileName}:`, error);
        throw new Error(`Failed to update scene file: ${fileName}`);
      }
    }

    default:
      throw new Error(`Unknown tool: ${request.params.name}`);
  }
});

/**
 * Handler that lists available prompts.
 * Exposes a single "summarize_notes" prompt that summarizes all notes.
 */
server.setRequestHandler(ListPromptsRequestSchema, async () => {
  return {
    prompts: [
      {
        name: "summarize_notes",
        description: "Summarize all notes",
      }
    ]
  };
});

// Remove or adapt the summarize_notes prompt handler as it relies on the 'notes' object
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  // Example: Keep the handler but return an error or adapt for scenes if needed
  if (request.params.name === "summarize_notes") {
     console.warn("Summarize notes prompt is not functional with scene files.");
     throw new Error("Summarize notes prompt is not available for scene files.");
  }
  throw new Error(`Unknown prompt: ${request.params.name}`);

  // // If adapting for scenes (example - might not be practical):
  // if (request.params.name === "summarize_scenes") { // Hypothetical new prompt
  //   try {
  //     const files = await fs.promises.readdir(scenesDir);
  //     const sceneFiles = files.filter(file => file.endsWith('.scene') && !file.endsWith('.meta'));
  //     const embeddedScenes = await Promise.all(sceneFiles.map(async fileName => {
  //        const filePath = path.join(scenesDir, fileName);
  //        const content = await fs.promises.readFile(filePath, 'utf-8');
  //        return {
  //          type: "resource" as const,
  //          resource: {
  //            uri: `scene:///${fileName}`,
  //            mimeType: "application/json",
  //            text: content // Be mindful of token limits!
  //          }
  //        };
  //     }));
  //     return {
  //       messages: [
  //         { role: "user", content: { type: "text", text: "Analyze the following scene files:" } },
  //         ...embeddedScenes.map(scene => ({ role: "user" as const, content: scene })),
  //         { role: "user", content: { type: "text", text: "Provide insights based on the scene data." } }
  //       ]
  //     };
  //   } catch (error) {
  //      console.error("Error generating summarize_scenes prompt:", error);
  //      throw new Error("Failed to generate scene summary prompt.");
  //   }
  // }
  // throw new Error(`Unknown prompt: ${request.params.name}`);
});

/**
 * Start the server using stdio transport.
 * This allows the server to communicate via standard input/output streams.
 */
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
