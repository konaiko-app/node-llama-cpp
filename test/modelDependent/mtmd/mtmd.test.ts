import {describe, expect, test} from "vitest";
import {LlamaMmproj} from "../../../src/index.js";
import {getTestLlama} from "../../utils/getTestLlama.js";
import {getModelFile} from "../../utils/modelFiles.js";

describe("mtmd", () => {
    describe("load and query", () => {
        test("loads mmproj and reports vision support", {timeout: 1000 * 60 * 10}, async () => {
            const modelPath = await getModelFile("gemma-4-E2B-it-Q4_K_M.gguf");
            const mmprojPath = await getModelFile("mmproj-gemma-4-E2B-it-Q8_0.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({modelPath});
            const mtmd = await model.loadMmproj(mmprojPath);

            expect(mtmd).toBeInstanceOf(LlamaMmproj);
            expect(mtmd.disposed).toBe(false);
            expect(mtmd.supportsVision).toBe(true);
            expect(mtmd.filePath).toBe(mmprojPath);

            const mtmd2 = await model.loadMmproj(mmprojPath);
            expect(mtmd2).toBe(mtmd);

            await mtmd.dispose();
            expect(mtmd.disposed).toBe(true);
        });
    });

    describe("tokenize", () => {
        test("tokenizes text-only prompt into text chunks", {timeout: 1000 * 60 * 10}, async () => {
            const modelPath = await getModelFile("gemma-4-E2B-it-Q4_K_M.gguf");
            const mmprojPath = await getModelFile("mmproj-gemma-4-E2B-it-Q8_0.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({modelPath});
            const mtmd = await model.loadMmproj(mmprojPath);

            const result = await mtmd.tokenize("Hello, world!", [], {
                addSpecial: false,
                parseSpecial: false
            });

            expect(result.chunks.length).toBe(1);
            expect(result.chunks[0]!.type).toBe("text");
            expect(result.chunks[0]!.tokens).toBeDefined();
            expect(result.chunks[0]!.tokens!.length).toBeGreaterThan(0);

            await mtmd.dispose();
        });

        test("tokenizes prompt with image into mixed chunks", {timeout: 1000 * 60 * 10}, async () => {
            const modelPath = await getModelFile("gemma-4-E2B-it-Q4_K_M.gguf");
            const mmprojPath = await getModelFile("mmproj-gemma-4-E2B-it-Q8_0.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({modelPath});
            const mtmd = await model.loadMmproj(mmprojPath);

            const testImageData = new Uint8Array([
                255, 0, 0, 255, 0, 0,
                255, 0, 0, 255, 0, 0
            ]);

            const result = await mtmd.tokenize(
                "Describe this image: <__media__>",
                [{data: testImageData, width: 2, height: 2}]
            );

            expect(result.chunks.length).toBeGreaterThanOrEqual(2);

            const textChunks = result.chunks.filter(c => c.type === "text");
            const imageChunks = result.chunks.filter(c => c.type === "image");

            expect(textChunks.length).toBeGreaterThanOrEqual(1);
            expect(imageChunks.length).toBe(1);
            expect(imageChunks[0]!.embeddings).toBeDefined();
            expect(imageChunks[0]!.embeddings!.length).toBeGreaterThan(0);
            expect(imageChunks[0]!.nTokens).toBeGreaterThan(0);

            await mtmd.dispose();
        });

        test("tokenizes with fileData input (encoded image bytes)", {timeout: 1000 * 60 * 10}, async () => {
            const modelPath = await getModelFile("gemma-4-E2B-it-Q4_K_M.gguf");
            const mmprojPath = await getModelFile("mmproj-gemma-4-E2B-it-Q8_0.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({modelPath});
            const mtmd = await model.loadMmproj(mmprojPath);

            // Minimal valid 1x1 red PNG (68 bytes)
            const pngBytes = new Uint8Array([
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // 8-bit RGB
                0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
                0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, // compressed data
                0x00, 0x00, 0x03, 0x00, 0x01, 0x36, 0x28, 0x19, // (red pixel)
                0x00, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
                0x44, 0xAE, 0x42, 0x60, 0x82
            ]);

            const result = await mtmd.tokenize(
                "What is in this image? <__media__>",
                [{fileData: pngBytes}]
            );

            const imageChunks = result.chunks.filter(c => c.type === "image");
            expect(imageChunks.length).toBe(1);
            expect(imageChunks[0]!.embeddings).toBeDefined();

            await mtmd.dispose();
        });
    });

    describe("error handling", () => {
        test("rejects when bitmap count does not match markers", {timeout: 1000 * 60 * 10}, async () => {
            const modelPath = await getModelFile("gemma-4-E2B-it-Q4_K_M.gguf");
            const mmprojPath = await getModelFile("mmproj-gemma-4-E2B-it-Q8_0.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({modelPath});
            const mtmd = await model.loadMmproj(mmprojPath);

            await expect(
                mtmd.tokenize("Image: <__media__>", [])
            ).rejects.toThrow();

            await mtmd.dispose();
        });
    });

    describe("disposal", () => {
        test("model disposal cleans up mtmd contexts", {timeout: 1000 * 60 * 10}, async () => {
            const modelPath = await getModelFile("gemma-4-E2B-it-Q4_K_M.gguf");
            const mmprojPath = await getModelFile("mmproj-gemma-4-E2B-it-Q8_0.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({modelPath});
            const mtmd = await model.loadMmproj(mmprojPath);

            expect(mtmd.disposed).toBe(false);

            await model.dispose();

            expect(mtmd.disposed).toBe(true);
        });
    });
});
