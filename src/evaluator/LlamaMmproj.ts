import {AsyncDisposeAggregator, DisposedError, EventRelay} from "lifecycle-utils";
import {AddonMtmd, MtmdBitmapInput, MtmdTokenizeOptions, MtmdTokenizeResult} from "../bindings/AddonTypes.js";
import type {LlamaContext} from "./LlamaContext/LlamaContext.js";

export type LlamaMmprojOptions = {
    /** Use GPU for the multimodal projector. Defaults to `true`. */
    useGpu?: boolean,

    /** Custom media marker string. Defaults to `"<__media__>"`. */
    mediaMarker?: string,

    /** Number of threads for encoding. Defaults to hardware concurrency. */
    threads?: number,

    /** Minimum number of tokens for image input (for models with dynamic resolution). */
    imageMinTokens?: number,

    /** Maximum number of tokens for image input (for models with dynamic resolution). */
    imageMaxTokens?: number
};

export class LlamaMmproj {
    /** @internal */ private readonly _addonMtmd: AddonMtmd;
    /** @internal */ private readonly _disposeAggregator = new AsyncDisposeAggregator();
    /** @internal */ private _disposed = false;

    public readonly onDispose = new EventRelay<void>();

    /** @internal */
    private constructor(addonMtmd: AddonMtmd) {
        this._addonMtmd = addonMtmd;

        this._disposeAggregator.add(this.onDispose.dispatchEvent);
        this._disposeAggregator.add(async () => {
            await this._addonMtmd.dispose();
        });
    }

    get disposed(): boolean {
        return this._disposed;
    }

    /** Whether the loaded projector supports vision (image) input. */
    get supportsVision(): boolean {
        return this._addonMtmd.supportsVision;
    }

    /** Whether the loaded projector supports audio input. */
    get supportsAudio(): boolean {
        return this._addonMtmd.supportsAudio;
    }

    /** The file path of the loaded mmproj file. */
    get filePath(): string {
        return this._addonMtmd.filePath;
    }

    /**
     * Tokenize a prompt containing media markers and bitmaps into chunks.
     *
     * Text chunks contain token IDs (`tokens`). Image/audio chunks contain
     * float embeddings (`embeddings`) produced by encoding through the
     * projector, along with token/position counts.
     *
     * The default media marker is `<__media__>`. Each marker in the prompt
     * must correspond to one bitmap in the array.
     *
     * Bitmaps can be provided as:
     * - `{data: Uint8Array, width: number, height: number}` — raw RGB pixels
     * - `{fileData: Uint8Array}` — encoded file bytes (JPEG, PNG, etc.)
     *
     * Note: text chunks can be fed into `LlamaContext` via its normal batch
     * pipeline. Image/audio chunk embeddings are provided for inspection
     * and custom integration; for end-to-end evaluation use
     * {@link evalChunks} instead.
     */
    async tokenize(
        text: string,
        bitmaps: MtmdBitmapInput[],
        options?: MtmdTokenizeOptions
    ): Promise<MtmdTokenizeResult> {
        this._ensureNotDisposed();
        return await this._addonMtmd.tokenize(text, bitmaps, options);
    }

    /**
     * One-shot: tokenize, encode, and decode all chunks through a context.
     * Returns the updated `n_past` position.
     *
     * **Warning:** This calls `llama_decode` directly, bypassing the JS-side
     * batch queue and sequence management. After calling this, the JS-side
     * KV cache state will be out of sync. Use {@link tokenize} instead for
     * integration with `LlamaContext`'s managed batch pipeline.
     */
    async evalChunks(
        context: LlamaContext,
        text: string,
        bitmaps: MtmdBitmapInput[],
        nPast: number,
        seqId: number,
        options?: MtmdTokenizeOptions
    ): Promise<number> {
        this._ensureNotDisposed();
        return await this._addonMtmd.evalChunks(context._ctx, text, bitmaps, nPast, seqId, options);
    }

    public async dispose() {
        if (this._disposed)
            return;

        this._disposed = true;
        await this._disposeAggregator.dispose();
    }

    /** @hidden */
    public [Symbol.asyncDispose]() {
        return this.dispose();
    }

    /** @internal */
    private _ensureNotDisposed() {
        if (this._disposed)
            throw new DisposedError();
    }

    /** @internal */
    static _create(addonMtmd: AddonMtmd): LlamaMmproj {
        return new LlamaMmproj(addonMtmd);
    }
}
