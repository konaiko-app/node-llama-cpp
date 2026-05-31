import {Token} from "../../../types.js";
import {SequenceEvaluateOptions} from "../types.js";
import {LlamaContextSequence} from "../LlamaContext.js";
import {LlamaSampler} from "../LlamaSampler.js";
import {TokenPredictor} from "../TokenPredictor.js";
import {removeNullFields} from "../../../utils/removeNullFields.js";
import type {AddonContext, BatchLogitIndex} from "../../../bindings/AddonTypes.js";

const defaultMaxTokens = 3;
const defaultMinConfidence = 0;

/**
 * Predicts the next tokens using built-in NextN MTP (Multi-Token Prediction)
 * layers in the model (e.g. Qwen 3.5/3.6). Creates a second context from the
 * same model with LLAMA_CONTEXT_TYPE_MTP and uses the target context's
 * pre-norm hidden states to draft tokens speculatively.
 *
 * Unlike DraftSequenceTokenPredictor, this does NOT need a separate draft model —
 * the MTP layers are already embedded in the GGUF file.
 */
export class NextNTokenPredictor extends TokenPredictor {
    /** @internal */ private readonly _maxTokens: number;
    /** @internal */ private readonly _minConfidence: number;
    /** @internal */ private _targetSequence: LlamaContextSequence | null = null;
    /** @internal */ private _mtpCtx: AddonContext | null = null;
    /** @internal */ private _nEmbd: number = 0;
    /** @internal */ private _pendingH: Float32Array | null = null;
    /** @internal */ private _disposed: boolean = false;
    /** @internal */ private _consecutiveEmpty: number = 0;
    private static readonly _AUTO_DISABLE_THRESHOLD = 10;
    /** @internal */ private _initialized: boolean = false;
    /** @internal */ private _lastBatchLogitIndex: number = -1;
    /** @internal */ private _sampler: LlamaSampler | null = null;
    /** @internal */ private _evaluateOptions: Readonly<SequenceEvaluateOptions> = {};

    public constructor(options?: {
        /**
         * Maximum number of tokens to draft per prediction cycle.
         * Defaults to `6`.
         */
        maxTokens?: number,

        /**
         * Minimum confidence threshold for draft tokens (0-1).
         * Set to `0` to disable confidence filtering.
         * Defaults to `0`.
         */
        minConfidence?: number,
    }) {
        super();
        this._maxTokens = options?.maxTokens ?? defaultMaxTokens;
        this._minConfidence = options?.minConfidence ?? defaultMinConfidence;
    }

    public get maxTokens() { return this._maxTokens; }
    public get minConfidence() { return this._minConfidence; }

    public async reset({targetSequence, evaluateOptions}: {
        targetSequence: LlamaContextSequence,
        stateTokens: Token[],
        evaluateOptions: Readonly<SequenceEvaluateOptions>
    }): Promise<void> {
        this._targetSequence = targetSequence;
        this._evaluateOptions = evaluateOptions ?? {};

        if (this._sampler) {
            this._sampler.dispose();
            this._sampler = null;
        }
        this._sampler = new LlamaSampler(targetSequence.model);
        this._sampler.applyConfig(removeNullFields({
            temperature: this._evaluateOptions.temperature ?? 0,
            minP: this._evaluateOptions.minP ?? 0,
            topK: this._evaluateOptions.topK ?? 40,
            topP: this._evaluateOptions.topP ?? 0.95,
            seed: this._evaluateOptions.seed != null
                ? Math.max(0, Math.floor(this._evaluateOptions.seed))
                : undefined,
        }) as Parameters<LlamaSampler['applyConfig']>[0]);

        if (!this._initialized && !this._disposed) {
            await this._initMtpContext();
            this._initialized = true;
        }
    }

    public pushTokens(_tokens: Token[]): void {
        // The pending hidden state is updated in predictTokens() after
        // reading from the target context, so pushTokens is a no-op.
    }

    public async predictTokens(): Promise<Token[]> {
        if (!this._mtpCtx || !this._targetSequence || this._disposed) {
            return [];
        }

        if (this._consecutiveEmpty >= NextNTokenPredictor._AUTO_DISABLE_THRESHOLD) {
            return [];
        }

        const targetCtx = this._targetSequence.context._ctx;

        try {
            const h = this._pendingH ?? targetCtx.getEmbeddingsPreNormIth(-1);
            if (h == null) {
                this._consecutiveEmpty++;
                if (this._consecutiveEmpty === NextNTokenPredictor._AUTO_DISABLE_THRESHOLD) {
                    console.warn('[NextNTokenPredictor] embeddings readback failed 10 consecutive times — disabling speculative decoding for this session');
                }
                return [];
            }

            let currentToken = -1;
            const ctxTokens = this._targetSequence.contextTokens;
            if (ctxTokens.length > 0) {
                currentToken = ctxTokens[ctxTokens.length - 1]!;
            }
            if (currentToken === -1) return [];

            const currentPos = this._targetSequence.nextTokenIndex - 1;

            const drafted = await this._mtpCtx.predictMtpTokens(
                0, currentPos, currentToken, h, this._maxTokens, this._sampler!._sampler,
            );

            this._pendingH = null;

            const predictions = Array.from(drafted) as Token[];
            if (predictions.length > 0) this._consecutiveEmpty = 0;
            else this._consecutiveEmpty++;
            return predictions;
        } catch (err) {
            this._consecutiveEmpty++;
            if (this._consecutiveEmpty <= 3) {
                console.warn('[NextNTokenPredictor] prediction failed:', (err as Error).message);
            }
            return [];
        }
    }

    public override stop(_untilPredictionsExhausted?: boolean): void {
        // Single-shot predictions, nothing to abort
    }

    public override async dispose(): Promise<void> {
        this._disposed = true;
        if (this._sampler) {
            this._sampler.dispose();
            this._sampler = null;
        }
        if (this._mtpCtx) {
            await this._mtpCtx.dispose();
            this._mtpCtx = null;
        }
    }

    /** @internal */
    private async _initMtpContext(): Promise<void> {
        if (!this._targetSequence) throw new Error("Target sequence not set");

        const targetContext = this._targetSequence.context;
        const bindings = targetContext.model._llama._bindings;
        const model = targetContext.model._model;

        // Verify the addon has MTP methods before creating the context
        const targetAddonCtx = targetContext._ctx;
        if (typeof targetAddonCtx.getModelNEmbd !== 'function') {
            console.warn('[NextNTokenPredictor] Addon does not support MTP methods — rebuild required');
            return;
        }

        this._nEmbd = targetAddonCtx.getModelNEmbd();

        this._mtpCtx = new bindings.AddonContext(model, {
            contextSize: targetContext.contextSize,
            batchSize: (targetContext as any)._batchSize ?? 512,
            sequences: 1,
            flashAttention: true,
            ctxType: "mtp",
        });

        const loaded = await this._mtpCtx.init();
        if (!loaded) {
            this._mtpCtx = null;
            console.warn('[NextNTokenPredictor] Failed to initialize MTP context (model may not support ctx_type=MTP)');
            return;
        }

        // Allocate MTP batch with both token and embd slots
        this._mtpCtx.initMtpBatch(this._maxTokens + 1, this._nEmbd);

        // Enable pre-norm embedding capture on both contexts.
        // Use unmasked mode (second arg false) so embeddings are stored
        // densely by position — masked mode requires tokens to be marked
        // as output in the batch, which the normal chat generation path
        // doesn't guarantee for every token.
        this._mtpCtx.setEmbeddingsPreNorm(true, true);
        targetContext._ctx.setEmbeddingsPreNorm(true, false);

        console.log(`[NextNTokenPredictor] MTP context initialized (nEmbd=${this._nEmbd}, ctxSize=${targetContext.contextSize}, maxTokens=${this._maxTokens})`);
    }
}
