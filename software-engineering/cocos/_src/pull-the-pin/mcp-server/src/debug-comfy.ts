import * as fs from 'fs';
import * as path from 'path';
import { WebSocket, RawData } from 'ws';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';

// --- Configuration ---
const comfyuiAddress = process.env.COMFYUI_ADDRESS || 'http://127.0.0.1:11188'; // Allow overriding via env var, default to 11188
const outputDirName = 'out';
// --- End Configuration ---

// Get the directory name in ES module scope
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const outputDir = path.resolve(__dirname, '..', outputDirName); // Place 'out' dir in mcp-server root

// Helper function to ensure output directory exists
async function ensureOutputDir(): Promise<void> {
    try {
        await fs.promises.mkdir(outputDir, { recursive: true });
        console.log(`Output directory ensured: ${outputDir}`);
    } catch (error: any) {
        console.error(`Error creating output directory ${outputDir}:`, error);
        throw error; // Re-throw to stop execution if dir creation fails
    }
}

// Helper function to download and save an image
async function downloadAndSaveImage(imageUrl: string, filename: string): Promise<void> {
    try {
        const response = await fetch(imageUrl);
        if (!response.ok) {
            throw new Error(`Failed to download image ${imageUrl}: ${response.status} ${response.statusText}`);
        }
        if (!response.body) {
            throw new Error(`Response body is null for image ${imageUrl}`);
        }
        const imageBuffer = Buffer.from(await response.arrayBuffer());
        const savePath = path.join(outputDir, filename);
        await fs.promises.writeFile(savePath, imageBuffer);
        console.log(`Image saved to: ${savePath}`);
    } catch (error) {
        console.error(`Error downloading or saving image ${filename}:`, error);
        // Decide if you want to throw or just log the error for individual image failures
        // throw error;
    }
}


// Function to interact with ComfyUI
async function generateImage(promptWorkflow: object, clientId: string): Promise<string[]> {
    const baseUrl = comfyuiAddress.replace(/\/$/, '');
    const promptUrl = `${baseUrl}/prompt`;
    const historyUrl = (promptId: string) => `${baseUrl}/history/${promptId}`;
    const viewUrl = (filename: string, subfolder: string, type: string) =>
        `${baseUrl}/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(subfolder)}&type=${type}`;
    const wsUrl = `${baseUrl.replace(/^http/, 'ws')}/ws?clientId=${clientId}`;

    console.log(`Connecting to ComfyUI at: ${baseUrl}`);
    console.log(`WebSocket URL: ${wsUrl}`);

    // 1. POST to /prompt
    console.log("Sending prompt to ComfyUI...");
    const postResponse = await fetch(promptUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            prompt: promptWorkflow,
            client_id: clientId,
        }),
    });

    if (!postResponse.ok) {
        const errorText = await postResponse.text();
        throw new Error(`ComfyUI /prompt request failed: ${postResponse.status} ${postResponse.statusText} - ${errorText}`);
    }

    const promptResult = await postResponse.json();
    const promptId = promptResult.prompt_id;
    if (!promptId) {
        throw new Error("ComfyUI did not return a prompt_id");
    }
    console.log(`Prompt submitted successfully. Prompt ID: ${promptId}`);

    // 2. Wait for execution via WebSocket
    console.log("Waiting for execution via WebSocket...");
    const imageOutputs = await new Promise<{ filename: string; subfolder: string; type: string }[]>((resolve, reject) => {
        const ws = new WebSocket(wsUrl);
        let executionFinished = false;
        let timeoutHandle: NodeJS.Timeout | null = null;

        const cleanup = (timer: NodeJS.Timeout | null) => {
            if (timer) clearTimeout(timer);
            if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
                ws.close();
            }
        };

        ws.on('open', () => {
            console.log(`WebSocket connected for client ${clientId}`);
             // Start timeout after connection is open
             timeoutHandle = setTimeout(() => {
                if (!executionFinished) {
                    console.error("ComfyUI execution timed out after 120 seconds.");
                    cleanup(timeoutHandle);
                    reject(new Error("ComfyUI execution timed out after 120 seconds."));
                }
             }, 120000); // 2 minutes timeout
        });

        ws.on('message', async (data: RawData) => {
            try {
                const message = JSON.parse(data.toString());
                 // console.log("WS Message:", message); // Verbose logging

                if (message.type === 'status') {
                    const queueRemaining = message.data?.status?.exec_info?.queue_remaining;
                    if (queueRemaining !== undefined) {
                        console.log(`Queue remaining: ${queueRemaining}`);
                    }
                } else if (message.type === 'progress') {
                     console.log(`Progress: ${message.data.value}/${message.data.max}`);
                } else if (message.type === 'executing') {
                    // Useful to know when the specific prompt starts
                    if (message.data.prompt_id === promptId && message.data.node !== null) {
                        console.log(`Executing node: ${message.data.node}`);
                    } else if (message.data.prompt_id === promptId && message.data.node === null) {
                         console.log(`Execution started for prompt ${promptId}`);
                    }
                } else if (message.type === 'executed' && message.data.prompt_id === promptId) {
                    console.log(`Execution finished for prompt ${promptId}. Fetching history...`);
                    executionFinished = true;
                    cleanup(timeoutHandle); // Clear timeout and close WS

                    // 3. GET /history (general history)
                    console.log("Fetching general history...");
                    const generalHistoryUrl = `${baseUrl}/history`;
                    const historyResponse = await fetch(generalHistoryUrl);
                    if (!historyResponse.ok) {
                        throw new Error(`ComfyUI /history (general) request failed: ${historyResponse.status} ${historyResponse.statusText}`);
                    }
                    const historyResult = await historyResponse.json(); // historyResult is { "prompt_id1": {...}, "prompt_id2": {...} }

                    // 4. Extract image info from history outputs
                    const outputs = historyResult[promptId]?.outputs;
                    if (!outputs) {
                        console.warn(`No outputs found in history for prompt_id ${promptId}`);
                        resolve([]); // Resolve with empty array if no outputs
                        return;
                    }

                    const foundImages: { filename: string; subfolder: string; type: string }[] = [];
                    for (const nodeId in outputs) {
                        if (outputs[nodeId].images) {
                            outputs[nodeId].images.forEach((image: { filename: string; subfolder: string; type: string }) => {
                                foundImages.push(image);
                            });
                        }
                    }
                    resolve(foundImages);

                } else if (message.type === 'execution_error' && message.data.prompt_id === promptId) {
                     console.error(`ComfyUI execution error for prompt ${promptId}:`, message.data);
                     cleanup(timeoutHandle);
                     reject(new Error(`ComfyUI execution error: ${JSON.stringify(message.data)}`));
                } else if (message.type === 'execution_cached' && message.data.prompt_id === promptId) {
                    // Handle cached execution similarly to 'executed'
                    console.log(`Execution was cached for prompt ${promptId}. Fetching history...`);
                    executionFinished = true; // Treat as finished
                    cleanup(timeoutHandle);

                    // Also fetch general history for cached prompts
                    console.log("Fetching general history for cached prompt...");
                    const generalHistoryUrlCached = `${baseUrl}/history`;
                    const historyResponse = await fetch(generalHistoryUrlCached);
                     if (!historyResponse.ok) {
                        throw new Error(`ComfyUI /history (general) request failed (cached): ${historyResponse.status} ${historyResponse.statusText}`);
                    }
                    const historyResult = await historyResponse.json();
                    const promptHistoryCached = historyResult[promptId]; // Get the specific prompt's history entry
                    const outputs = promptHistoryCached?.outputs; // Access the outputs for the specific prompt
                     if (!outputs) {
                        console.warn(`No outputs found in history for cached prompt_id ${promptId}`);
                        resolve([]);
                        return;
                    }
                    const foundImages: { filename: string; subfolder: string; type: string }[] = [];
                    for (const nodeId in outputs) {
                        if (outputs[nodeId].images) {
                            outputs[nodeId].images.forEach((image: { filename: string; subfolder: string; type: string }) => {
                                foundImages.push(image);
                            });
                        }
                    }
                    resolve(foundImages);
                }


            } catch (parseError) {
                console.error("Error processing WebSocket message:", parseError, "\nRaw data:", data.toString());
            }
        });

        ws.on('error', (error: Error) => {
            console.error("WebSocket error:", error);
            if (!executionFinished) {
                cleanup(timeoutHandle);
                reject(new Error(`WebSocket connection error: ${error.message}`));
            }
        });

        ws.on('close', (code: number, reason: Buffer) => {
            console.log(`WebSocket closed. Code: ${code}, Reason: ${reason.toString()}`);
            if (!executionFinished) {
                // If closed unexpectedly before finishing and not due to timeout/error handled above
                cleanup(timeoutHandle); // Ensure timer is cleared
                reject(new Error(`WebSocket closed unexpectedly (code: ${code}) before execution finished.`));
            }
        });
    });

    if (imageOutputs.length === 0) {
        console.log("No images found in the execution output.");
        return [];
    }

    // 5. Download and save images
    console.log(`Found ${imageOutputs.length} image(s). Downloading...`);
    const savedFilePaths: string[] = [];
    for (const imageInfo of imageOutputs) {
        const imageUrl = viewUrl(imageInfo.filename, imageInfo.subfolder, imageInfo.type);
        const saveFilename = `${clientId}-${imageInfo.filename}`; // Prefix with client ID for uniqueness
        await downloadAndSaveImage(imageUrl, saveFilename);
        savedFilePaths.push(path.join(outputDir, saveFilename));
    }

    return savedFilePaths;
}

// Main execution logic
async function main() {
    // Get workflow from command line argument
    if (process.argv.length < 3) {
        console.error("Usage: node debug-comfy.js '<prompt_workflow_json_string>'");
        process.exit(1);
    }
    const workflowJsonString = process.argv[2];
    let promptWorkflow: object;
    try {
        promptWorkflow = JSON.parse(workflowJsonString);
    } catch (e) {
        console.error("Invalid JSON provided for prompt workflow:", e);
        process.exit(1);
    }

    const clientId = uuidv4();
    console.log(`Using Client ID: ${clientId}`);

    try {
        await ensureOutputDir();
        const savedFiles = await generateImage(promptWorkflow, clientId);
        if (savedFiles.length > 0) {
            console.log("\nImage generation successful. Saved files:");
            savedFiles.forEach(file => console.log(`- ${file}`));
        } else {
            console.log("\nImage generation process completed, but no images were saved (check workflow/history).");
        }
    } catch (error) {
        console.error("\n--- Image Generation Failed ---");
        if (error instanceof Error) {
            console.error(error.message);
        } else {
            console.error(error);
        }
        process.exit(1);
    }
}

main();