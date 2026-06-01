import {Token} from "../../../types.js";
import {SequenceEvaluateOptions} from "../types.js";
import {LlamaContextSequence} from "../LlamaContext.js";
import {TokenPredictor} from "../TokenPredictor.js";

const defaultMaxTokens = 4;

/**
 * Predicts the next tokens for **Gemma 4 MTP** (Multi-Token Prediction).
 *
 * Unlike {@link NextNTokenPredictor} (Qwen-style, which runs a separate
 * `LLAMA_CONTEXT_TYPE_MTP` context), the Gemma 4 assistant is a small model
 * loaded *into the target model* via `model.loadMtpAssistant(path)` and it
 * **cross-attends into the target context's live KV cache**. Drafting therefore
 * runs on the *target* context itself via `llama_decode_mtp` — there is no
 * separate draft context.
 *
 * Requirements:
 * - The assistant must already be loaded into the target model:
 *   `await model.loadMtpAssistant(assistantGgufPath)`.
 * - The drafts are the assistant's in-graph greedy argmax, so this predictor
 *   takes no sampler (it is exact: a draft is only kept by the speculative loop
 *   if it matches the target's own sampled token).
 */
export class GemmaMtpTokenPredictor extends TokenPredictor {
    /** @internal */ private readonly _maxTokens: number;
    /** @internal */ private _targetSequence: LlamaContextSequence | null = null;
    /** @internal */ private _disposed: boolean = false;
    /** @internal */ private _embeddingsEnabled: boolean = false;
    /** @internal */ private _consecutiveEmpty: number = 0;
    private static readonly _AUTO_DISABLE_THRESHOLD = 16;

    public constructor(options?: {
        /**
         * Maximum number of tokens to draft per prediction cycle.
         * Defaults to `4`.
         */
        maxTokens?: number
    }) {
        super();
        this._maxTokens = Math.max(1, options?.maxTokens ?? defaultMaxTokens);
    }

    public get maxTokens() {
        return this._maxTokens;
    }

    public reset({targetSequence}: {
        targetSequence: LlamaContextSequence,
        stateTokens: Token[],
        evaluateOptions: Readonly<SequenceEvaluateOptions>
    }): void {
        this._targetSequence = targetSequence;
        this._consecutiveEmpty = 0;

        const targetCtx = targetSequence.context._ctx;
        if (typeof (targetCtx as any).predictGemma4Mtp !== "function") {
            // Addon was built without the Gemma 4 MTP methods — disable gracefully.
            this._targetSequence = null;
            console.warn("[GemmaMtpTokenPredictor] addon does not expose predictGemma4Mtp — rebuild required; speculative decoding disabled");
            return;
        }

        // The assistant cross-attention reads the target's post-norm hidden state
        // as h_prev, which requires regular embeddings output on the target context.
        if (!this._embeddingsEnabled) {
            (targetCtx as any).setEmbeddings(true);
            this._embeddingsEnabled = true;
        }
    }

    public pushTokens(_tokens: Token[]): void {
        // h_prev is read fresh from the target context inside predictTokens(),
        // so there is no carried state to update here.
    }

    public async predictTokens(): Promise<Token[]> {
        if (this._targetSequence == null || this._disposed)
            return [];

        if (this._consecutiveEmpty >= GemmaMtpTokenPredictor._AUTO_DISABLE_THRESHOLD)
            return [];

        const sequence = this._targetSequence;
        const targetCtx = sequence.context._ctx as any;

        const ctxTokens = sequence.contextTokens;
        if (ctxTokens.length === 0)
            return [];

        const lastToken = ctxTokens[ctxTokens.length - 1]!;
        const attnPos = sequence.nextTokenIndex - 1;
        const seqId = sequence._internalSequenceId;

        try {
            // node-llama-cpp's speculative verify batch is [pendingToken, ...drafts], where
            // pendingToken is the token already sampled after lastToken. The assistant, seeded
            // with lastToken, drafts predictions starting AT pendingToken — so its first draft
            // duplicates the token node-llama-cpp already has. Draft one extra and drop it so the
            // remaining drafts align with the positions the verify decode actually checks.
            // h_prev must come from the seed (last accepted) token's output row in the most
            // recent verify decode — not the last (possibly rejected) draft row. The speculative
            // loop records that row index per round; -1 falls back to the last output (prompt).
            const hiddenIndex = sequence._mtpSeedHiddenIndex ?? -1;

            const drafted: Int32Array = await targetCtx.predictGemma4Mtp(
                seqId, attnPos, lastToken, this._maxTokens + 1, hiddenIndex
            );

            const predictions = Array.from(drafted).slice(1) as Token[];
            if (predictions.length > 0)
                this._consecutiveEmpty = 0;
            else
                this._consecutiveEmpty++;

            return predictions;
        } catch (err) {
            this._consecutiveEmpty++;
            if (this._consecutiveEmpty <= 3)
                console.warn("[GemmaMtpTokenPredictor] prediction failed:", (err as Error).message);

            return [];
        }
    }

    public override stop(_untilPredictionsExhausted?: boolean): void {
        // Single-shot synchronous drafts per call — nothing to abort.
    }

    public override dispose(): void {
        this._disposed = true;
        this._targetSequence = null;
    }
}
