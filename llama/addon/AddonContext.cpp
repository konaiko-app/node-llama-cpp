#include <thread>
#include <algorithm>
#include <cmath>
#include "common/common.h"
#include "llama-context.h"
#include "llama-vocab.h"
#include "llama.h"

#include "llama-ext.h"

#include "addonGlobals.h"
#include "AddonModel.h"
#include "AddonModelLora.h"
#include "AddonGrammarEvaluationState.h"
#include "AddonContext.h"

static uint64_t calculateBatchMemorySize(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    uint64_t totalSize = 0;

    if (embd) {
        totalSize += sizeof(float) * n_tokens_alloc * embd;
    } else {
        totalSize += sizeof(llama_token) * n_tokens_alloc;
    }

    totalSize += sizeof(llama_pos) * n_tokens_alloc;
    totalSize += sizeof(int32_t) * n_tokens_alloc;
    totalSize += sizeof(llama_seq_id *) * (n_tokens_alloc + 1);

    totalSize += sizeof(llama_seq_id) * n_seq_max * n_tokens_alloc;

    totalSize += sizeof(int8_t) * n_tokens_alloc;

    return totalSize;
}

class AddonContextDecodeBatchWorker : public Napi::AsyncWorker {
    public:
        AddonContext* ctx;

        AddonContextDecodeBatchWorker(const Napi::Env& env, AddonContext* ctx)
            : Napi::AsyncWorker(env, "AddonContextDecodeBatchWorker"),
              ctx(ctx),
              deferred(Napi::Promise::Deferred::New(env)) {
            ctx->Ref();
        }
        ~AddonContextDecodeBatchWorker() {
            ctx->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                // Perform the evaluation using llama_decode.
                int r = llama_decode(ctx->ctx, ctx->batch);

                if (r != 0) {
                    if (r == 1) {
                        SetError("could not find a KV slot for the batch (try reducing the size of the batch or increase the context)");
                    } else {
                        SetError("Eval has failed");
                    }

                    return;
                }

                llama_synchronize(ctx->ctx);
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch(...) {
                SetError("Unknown error when calling \"llama_decode\"");
            }
        }
        void OnOK() {
            deferred.Resolve(Env().Undefined());
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};

class AddonContextLoadContextWorker : public Napi::AsyncWorker {
    public:
        AddonContext* context;

        AddonContextLoadContextWorker(const Napi::Env& env, AddonContext* context)
            : Napi::AsyncWorker(env, "AddonContextLoadContextWorker"),
              context(context),
              deferred(Napi::Promise::Deferred::New(env)) {
            context->Ref();
        }
        ~AddonContextLoadContextWorker() {
            context->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                context->ctx = llama_init_from_model(context->model->model, context->context_params);

                context->contextLoaded = context->ctx != nullptr;
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch(...) {
                SetError("Unknown error when calling \"llama_init_from_model\"");
            }
        }
        void OnOK() {
            if (context->contextLoaded && !context->model->model_params.no_alloc) {
                uint64_t contextMemorySize = llama_state_get_size(context->ctx);
                adjustNapiExternalMemoryAdd(Env(), contextMemorySize);
                context->loadedContextMemorySize = contextMemorySize;
            }

            deferred.Resolve(Napi::Boolean::New(Env(), context->contextLoaded));
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};
class AddonContextUnloadContextWorker : public Napi::AsyncWorker {
    public:
        AddonContext* context;

        AddonContextUnloadContextWorker(const Napi::Env& env, AddonContext* context)
            : Napi::AsyncWorker(env, "AddonContextUnloadContextWorker"),
              context(context),
              deferred(Napi::Promise::Deferred::New(env)) {
            context->Ref();
        }
        ~AddonContextUnloadContextWorker() {
            context->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                llama_free(context->ctx);
                context->contextLoaded = false;

                try {
                    if (context->has_batch) {
                        llama_batch_free(context->batch);
                        context->has_batch = false;
                        context->batch_n_tokens = 0;
                    }

                    context->dispose();
                } catch (const std::exception& e) {
                    SetError(e.what());
                } catch(...) {
                    SetError("Unknown error when calling \"llama_batch_free\"");
                }
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch(...) {
                SetError("Unknown error when calling \"llama_free\"");
            }
        }
        void OnOK() {
            if (!context->model->model_params.no_alloc) {
                adjustNapiExternalMemorySubtract(Env(), context->loadedContextMemorySize);
                context->loadedContextMemorySize = 0;
            }

            adjustNapiExternalMemorySubtract(Env(), context->batchMemorySize);
            context->batchMemorySize = 0;

            deferred.Resolve(Env().Undefined());
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};


class AddonContextSampleTokenWorker : public Napi::AsyncWorker {
    public:
        AddonContext* ctx;
        AddonSampler* sampler;
        bool arrayResult = false;
        bool returnProbabilities = false;
        bool returnConfidence = false;
        float tokenConfidence = -1;
        bool has_probabilities = false;
        size_t probabilities_size;
        llama_token * probabilities_tokens;
        float * probabilities_probs;
        int32_t batchLogitIndex;
        llama_token result;
        bool no_output = false;

        AddonContextSampleTokenWorker(const Napi::CallbackInfo& info, AddonContext* ctx)
            : Napi::AsyncWorker(info.Env(), "AddonContextSampleTokenWorker"),
              ctx(ctx),
              deferred(Napi::Promise::Deferred::New(info.Env())) {
            ctx->Ref();

            batchLogitIndex = info[0].As<Napi::Number>().Int32Value();
            sampler = Napi::ObjectWrap<AddonSampler>::Unwrap(info[1].As<Napi::Object>());
            arrayResult = info.Length() > 2 && info[2].IsBoolean();
            returnProbabilities = arrayResult ? info[2].As<Napi::Boolean>().Value() : false;
            returnConfidence = arrayResult && info.Length() > 3 && info[3].IsBoolean() ? info[3].As<Napi::Boolean>().Value() : false;
            sampler->Ref();
        }
        ~AddonContextSampleTokenWorker() {
            ctx->Unref();
            sampler->Unref();

            if (has_probabilities) {
                delete[] probabilities_tokens;
                delete[] probabilities_probs;
            }
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                SampleToken();
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch(...) {
                SetError("Unknown error when calling \"SampleToken\"");
            }
        }

        void SampleToken() {
            if (llama_get_logits(ctx->ctx) == nullptr) {
                SetError("This model does not support token generation");
                return;
            }

            sampler->rebuildChainIfNeeded();

            llama_token_data_array cur_p;
            sampler->sample(ctx->ctx, batchLogitIndex, cur_p, returnProbabilities || returnConfidence);

            if (!(cur_p.selected >= 0 && cur_p.selected < (int32_t)cur_p.size)) {
                no_output = true;
                return;
            }

            auto new_token_id = cur_p.data[cur_p.selected].id;

            if (returnProbabilities || returnConfidence) {
                if (!cur_p.sorted) {
                    std::sort(cur_p.data, cur_p.data + cur_p.size, [](const llama_token_data & a, const llama_token_data & b) {
                        return a.logit > b.logit;
                    });
                    cur_p.sorted = true;

                    for (size_t i = 0; i < cur_p.size; i++) {
                        if (cur_p.data[i].id == new_token_id) {
                            cur_p.selected = i;
                            break;
                        }
                    }
                }
            }

            if (returnProbabilities) {
                probabilities_size = cur_p.size;
                probabilities_tokens = new llama_token[probabilities_size];
                probabilities_probs = new float[probabilities_size];
                float maxLogit = cur_p.size > 0 ? cur_p.data[0].logit : -INFINITY;

                for (size_t i = 0; i < cur_p.size; i++) {
                    auto logit = cur_p.data[i].logit;

                    probabilities_tokens[i] = cur_p.data[i].id;
                    probabilities_probs[i] = logit;

                    if (logit > maxLogit) {
                        maxLogit = logit;
                    }
                }

                if (probabilities_size > 0 && maxLogit != -INFINITY) {
                    float sum = 0.0f;
                    for (size_t i = 0; i < probabilities_size; i++) {
                        float prob = expf(probabilities_probs[i] - maxLogit);
                        probabilities_probs[i] = prob;
                        sum += prob;
                    }

                    for (size_t i = 0; i < probabilities_size; i++) {
                        probabilities_probs[i] /= sum;
                    }
                }

                has_probabilities = true;
            }

            if (returnConfidence) {
                if (has_probabilities && cur_p.selected < probabilities_size) {
                    tokenConfidence = probabilities_probs[cur_p.selected];
                } else {
                    float maxLogit = cur_p.data[0].logit;
                    float sum = 0.0f;
                    for (size_t i = 0; i < cur_p.size; i++) {
                        auto logit = cur_p.data[i].logit;

                        if (logit > maxLogit) {
                            maxLogit = logit;
                        }
                    }

                    for (size_t i = 0; i < cur_p.size; i++) {
                        sum += expf(cur_p.data[i].logit - maxLogit);
                    }

                    tokenConfidence = expf(cur_p.data[cur_p.selected].logit - maxLogit) / sum;
                }
            }

            try {
                sampler->acceptToken(new_token_id);
                result = new_token_id;
            } catch (const std::exception& e) {
                SetError(std::string("Failed to accept token in sampler: ") + e.what());
            } catch(...) {
                SetError("Unknown error when calling \"acceptToken\"");
            }
        }
        void OnOK() {
            Napi::Number resultToken;
            if (no_output) {
                resultToken = Napi::Number::New(Env(), -1);
            } else {
                resultToken = Napi::Number::New(Env(), static_cast<uint32_t>(result));
            }

            if (!arrayResult) {
                deferred.Resolve(resultToken);
                return;
            }

            Napi::Array resultArray = Napi::Array::New(Env(), 2);
            resultArray.Set(Napi::Number::New(Env(), 0), resultToken);

            if (has_probabilities) {
                Napi::Array probabilities = Napi::Array::New(Env(), probabilities_size * 2);
                for (size_t i = 0; i < probabilities_size; i++) {
                    probabilities.Set(i * 2, Napi::Number::New(Env(), probabilities_tokens[i]));
                    probabilities.Set(i * 2 + 1, Napi::Number::New(Env(), probabilities_probs[i]));
                }
                resultArray.Set(1, probabilities);
            }

            if (returnConfidence && tokenConfidence != -1) {
                resultArray.Set(2, Napi::Number::New(Env(), tokenConfidence));
            }

            deferred.Resolve(resultArray);
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};

AddonContext::AddonContext(const Napi::CallbackInfo& info) : Napi::ObjectWrap<AddonContext>(info) {
    model = Napi::ObjectWrap<AddonModel>::Unwrap(info[0].As<Napi::Object>());
    model->Ref();

    context_params = llama_context_default_params();
    context_params.n_ctx = 4096;
    context_params.n_threads = std::max(common_cpu_get_num_math(), 1);
    context_params.n_threads_batch = context_params.n_threads;
    context_params.no_perf = true;
    context_params.swa_full = false;

    if (info.Length() > 1 && info[1].IsObject()) {
        const auto options = info[1].As<Napi::Object>();

        if (options.Has("contextSize")) {
            context_params.n_ctx = options.Get("contextSize").As<Napi::Number>().Uint32Value();
        }

        if (options.Has("batchSize")) {
            context_params.n_batch = options.Get("batchSize").As<Napi::Number>().Uint32Value();
            context_params.n_ubatch = context_params.n_batch; // the batch queue is managed in the JS side, so there's no need for managing it on the C++ side
        }

        if (options.Has("sequences")) {
            context_params.n_seq_max = options.Get("sequences").As<Napi::Number>().Uint32Value();
        }

        if (options.Has("embeddings")) {
            context_params.embeddings = options.Get("embeddings").As<Napi::Boolean>().Value();
        }

        if (options.Has("ranking") && options.Get("ranking").As<Napi::Boolean>().Value()) {
            context_params.pooling_type = LLAMA_POOLING_TYPE_RANK;
        }

        if (options.Has("flashAttention")) {
            const auto flashAttention = options.Get("flashAttention");

            if (flashAttention.IsString() && flashAttention.As<Napi::String>().Utf8Value() == "auto") {
                context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
            } else {
                const bool flashAttentionEnabled = flashAttention.As<Napi::Boolean>().Value();
                context_params.flash_attn_type = flashAttentionEnabled
                    ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                    : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            }
        }

        if (options.Has("threads")) {
            const auto threads = options.Get("threads").As<Napi::Number>().Int32Value();
            const auto resolvedThreads = threads == 0
                ? std::max((int32_t)std::thread::hardware_concurrency(), context_params.n_threads)
                : threads;

            context_params.n_threads = resolvedThreads;
            context_params.n_threads_batch = resolvedThreads;
        }

        if (options.Has("performanceTracking")) {
            context_params.no_perf = !(options.Get("performanceTracking").As<Napi::Boolean>().Value());
        }

        if (options.Has("kvCacheKeyType") && options.Get("kvCacheKeyType").IsNumber()) {
            const auto keyType = options.Get("kvCacheKeyType").As<Napi::Number>().Int32Value();
            if (keyType >= 0 && keyType < GGML_TYPE_COUNT) {
                context_params.type_k = static_cast<ggml_type>(keyType);
            }
        }

        if (options.Has("kvCacheValueType") && options.Get("kvCacheValueType").IsNumber()) {
            const auto valueType = options.Get("kvCacheValueType").As<Napi::Number>().Int32Value();
            if (valueType >= 0 && valueType < GGML_TYPE_COUNT) {
                context_params.type_v = static_cast<ggml_type>(valueType);
            }
        }

        if (options.Has("swaFullCache")) {
            context_params.swa_full = options.Get("swaFullCache").As<Napi::Boolean>().Value();
        }

        if (options.Has("ctxType")) {
            auto ctxTypeStr = options.Get("ctxType").As<Napi::String>().Utf8Value();
            if (ctxTypeStr == "mtp") {
                context_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            }
        }
    }
}
AddonContext::~AddonContext() {
    dispose();
}

void AddonContext::dispose() {
    if (disposed) {
        return;
    }

    disposed = true;
    if (contextLoaded) {
        contextLoaded = false;
        llama_free(ctx);

        if (!model->model_params.no_alloc) {
            adjustNapiExternalMemorySubtract(Env(), loadedContextMemorySize);
            loadedContextMemorySize = 0;
        }
    }

    model->Unref();

    disposeBatch();
}
void AddonContext::disposeBatch() {
    if (!has_batch) {
        return;
    }

    if (mtp_batch_token_owned && batch.token != nullptr) {
        free(batch.token);
        batch.token = nullptr;
        mtp_batch_token_owned = false;
    }
    llama_batch_free(batch);
    has_batch = false;
    batch_n_tokens = 0;

    adjustNapiExternalMemorySubtract(Env(), batchMemorySize);
    batchMemorySize = 0;
}

Napi::Value AddonContext::Init(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    AddonContextLoadContextWorker* worker = new AddonContextLoadContextWorker(this->Env(), this);
    worker->Queue();
    return worker->GetPromise();
}
Napi::Value AddonContext::Dispose(const Napi::CallbackInfo& info) {
    if (disposed) {
        return info.Env().Undefined();
    }

    if (contextLoaded) {
        contextLoaded = false;

        AddonContextUnloadContextWorker* worker = new AddonContextUnloadContextWorker(this->Env(), this);
        worker->Queue();
        return worker->GetPromise();
    } else {
        dispose();

        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(info.Env());
        deferred.Resolve(info.Env().Undefined());
        return deferred.Promise();
    }
}

Napi::Value AddonContext::GetContextSize(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_n_ctx(ctx));
}
Napi::Value AddonContext::InitBatch(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    if (has_batch) {
        llama_batch_free(batch);
    }

    int32_t n_tokens = info[0].As<Napi::Number>().Int32Value();

    batch = llama_batch_init(n_tokens, 0, 1);
    has_batch = true;
    batch_n_tokens = n_tokens;

    uint64_t newBatchMemorySize = calculateBatchMemorySize(n_tokens, llama_model_n_embd(model->model), context_params.n_batch);
    if (newBatchMemorySize > batchMemorySize) {
        adjustNapiExternalMemoryAdd(Env(), newBatchMemorySize - batchMemorySize);
        batchMemorySize = newBatchMemorySize;
    } else if (newBatchMemorySize < batchMemorySize) {
        adjustNapiExternalMemorySubtract(Env(), batchMemorySize - newBatchMemorySize);
        batchMemorySize = newBatchMemorySize;
    }

    return info.Env().Undefined();
}
Napi::Value AddonContext::DisposeBatch(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    disposeBatch();

    return info.Env().Undefined();
}
Napi::Value AddonContext::AddToBatch(const Napi::CallbackInfo& info) {
    if (!has_batch) {
        Napi::Error::New(info.Env(), "No batch is initialized").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    int32_t sequenceId = info[0].As<Napi::Number>().Int32Value();
    int32_t firstTokenContextIndex = info[1].As<Napi::Number>().Int32Value();
    Napi::Uint32Array tokens = info[2].As<Napi::Uint32Array>();
    Napi::Uint32Array tokenLogitIndexes = info[3].As<Napi::Uint32Array>();

    auto tokensLength = tokens.ElementLength();
    auto tokenLogitIndexesLength = tokenLogitIndexes.ElementLength();
    GGML_ASSERT(batch.n_tokens + tokensLength <= batch_n_tokens);

    Napi::Uint32Array resLogitIndexes = Napi::Uint32Array::New(info.Env(), tokenLogitIndexesLength);

    for (size_t i = 0, l = 0; i < tokensLength; i++) {
        if (l < tokenLogitIndexesLength && l < tokenLogitIndexesLength && tokenLogitIndexes[l] == i) {
            common_batch_add(batch, static_cast<llama_token>(tokens[i]), firstTokenContextIndex + i, { sequenceId }, true);
            resLogitIndexes[l] = batch.n_tokens - 1;
            l++;
        } else {
            common_batch_add(batch, static_cast<llama_token>(tokens[i]), firstTokenContextIndex + i, { sequenceId }, false);
        }
    }

    return resLogitIndexes;
}
Napi::Value AddonContext::DisposeSequence(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    int32_t sequenceId = info[0].As<Napi::Number>().Int32Value();

    bool result = llama_memory_seq_rm(llama_get_memory(ctx), sequenceId, -1, -1);

    if (!result) {
        Napi::Error::New(info.Env(), "Failed to dispose sequence").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    return info.Env().Undefined();
}
Napi::Value AddonContext::RemoveTokenCellsFromSequence(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    int32_t sequenceId = info[0].As<Napi::Number>().Int32Value();
    int32_t startPos = info[1].As<Napi::Number>().Int32Value();
    int32_t endPos = info[2].As<Napi::Number>().Int32Value();

    bool result = llama_memory_seq_rm(llama_get_memory(ctx), sequenceId, startPos, endPos);

    return Napi::Boolean::New(info.Env(), result);
}
Napi::Value AddonContext::ShiftSequenceTokenCells(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    int32_t sequenceId = info[0].As<Napi::Number>().Int32Value();
    int32_t startPos = info[1].As<Napi::Number>().Int32Value();
    int32_t endPos = info[2].As<Napi::Number>().Int32Value();
    int32_t shiftDelta = info[3].As<Napi::Number>().Int32Value();

    llama_memory_seq_add(llama_get_memory(ctx), sequenceId, startPos, endPos, shiftDelta);

    return info.Env().Undefined();
}
Napi::Value AddonContext::GetSequenceKvCacheMinPosition(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    int32_t sequenceId = info[0].As<Napi::Number>().Int32Value();


    const auto minPosition = llama_memory_seq_pos_min(llama_get_memory(ctx), sequenceId);

    return Napi::Number::New(info.Env(), minPosition);
}
Napi::Value AddonContext::GetSequenceKvCacheMaxPosition(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    int32_t sequenceId = info[0].As<Napi::Number>().Int32Value();


    const auto maxPosition = llama_memory_seq_pos_max(llama_get_memory(ctx), sequenceId);

    return Napi::Number::New(info.Env(), maxPosition);
}
Napi::Value AddonContext::DecodeBatch(const Napi::CallbackInfo& info) {
    AddonContextDecodeBatchWorker* worker = new AddonContextDecodeBatchWorker(info.Env(), this);
    worker->Queue();
    return worker->GetPromise();
}
Napi::Value AddonContext::SampleToken(const Napi::CallbackInfo& info) {
    AddonContextSampleTokenWorker* worker = new AddonContextSampleTokenWorker(info, this);
    worker->Queue();
    return worker->GetPromise();
}

Napi::Value AddonContext::GetEmbedding(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    int32_t inputTokensLength = info[0].As<Napi::Number>().Int32Value();
    int32_t maxVectorSize = (info.Length() > 1 && info[1].IsNumber()) ? info[1].As<Napi::Number>().Int32Value() : 0;

    if (inputTokensLength <= 0) {
        Napi::Error::New(info.Env(), "Invalid input tokens length").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    const int n_embd = llama_model_n_embd(model->model);
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const auto* embeddings = pooling_type == LLAMA_POOLING_TYPE_NONE ? NULL : llama_get_embeddings_seq(ctx, 0);
    if (embeddings == NULL) {
        embeddings = llama_get_embeddings_ith(ctx, inputTokensLength - 1);
    }

    if (embeddings == NULL) {
        Napi::Error::New(info.Env(), std::string("Failed to get embeddings for token ") + std::to_string(inputTokensLength - 1)).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    size_t resultSize = maxVectorSize == 0 ? n_embd : std::min(n_embd, maxVectorSize);
    Napi::Float64Array result = Napi::Float64Array::New(info.Env(), resultSize);
    for (size_t i = 0; i < resultSize; i++) {
        result[i] = embeddings[i];
    }

    return result;
}

Napi::Value AddonContext::GetStateSize(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_state_get_size(ctx));
}

Napi::Value AddonContext::GetMemoryBreakdown(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    if (!contextLoaded || ctx == nullptr) {
        Napi::Error::New(info.Env(), "Context is not loaded").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    std::size_t cpuRam = 0;
    std::size_t gpuVram = 0;

    for (const auto& [bufferType, memoryBreakdown] : ctx->memory_breakdown()) {
        const std::size_t size = memoryBreakdown.context + memoryBreakdown.compute;
        if (size == 0) {
            continue;
        }

        if (ggml_backend_buft_is_host(bufferType)) {
            cpuRam += size;
        } else {
            ggml_backend_dev_t device = ggml_backend_buft_get_device(bufferType);
            if (device != nullptr) {
                auto deviceType = ggml_backend_dev_type(device);
                if (deviceType == GGML_BACKEND_DEVICE_TYPE_GPU || deviceType == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    gpuVram += size;
                } else {
                    cpuRam += size;
                }
            } else {
                cpuRam += size;
            }
        }
    }

    Napi::Object result = Napi::Object::New(info.Env());
    result.Set("cpuRam", Napi::Number::New(info.Env(), cpuRam));
    result.Set("gpuVram", Napi::Number::New(info.Env(), gpuVram));
    return result;
}

Napi::Value AddonContext::GetThreads(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_n_threads(ctx));
}

Napi::Value AddonContext::SetThreads(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    const auto threads = info[0].As<Napi::Number>().Int32Value();
    const auto resolvedThreads = threads == 0
        ? std::max((int32_t)std::thread::hardware_concurrency(), std::max(common_cpu_get_num_math(), 1))
        : threads;

    if (llama_n_threads(ctx) != resolvedThreads) {
        llama_set_n_threads(ctx, resolvedThreads, resolvedThreads);
    }

    return info.Env().Undefined();
}

class AddonContextSaveSequenceStateToFileWorker : public Napi::AsyncWorker {
    public:
        AddonContext* context;
        std::string filepath;
        llama_seq_id sequenceId;
        std::vector<llama_token> tokens;
        size_t savedFileSize = 0;

        AddonContextSaveSequenceStateToFileWorker(const Napi::CallbackInfo& info, AddonContext* context)
            : Napi::AsyncWorker(info.Env(), "AddonContextSaveSequenceStateToFileWorker"),
              context(context),
              deferred(Napi::Promise::Deferred::New(info.Env())) {
            context->Ref();

            filepath = info[0].As<Napi::String>().Utf8Value();
            sequenceId = info[1].As<Napi::Number>().Int32Value();
            Napi::Uint32Array inputTokens = info[2].As<Napi::Uint32Array>();

            tokens.resize(inputTokens.ElementLength());
            for (size_t i = 0; i < tokens.size(); i++) {
                tokens[i] = inputTokens[i];
            }
        }
        ~AddonContextSaveSequenceStateToFileWorker() {
            context->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                savedFileSize = llama_state_seq_save_file(context->ctx, filepath.c_str(), sequenceId, tokens.data(), tokens.size());
                if (savedFileSize == 0) {
                    SetError("Failed to save state to file");
                    return;
                }
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch(...) {
                SetError("Unknown error when calling \"llama_state_seq_save_file\"");
            }
        }
        void OnOK() {
            deferred.Resolve(Napi::Number::New(Env(), savedFileSize));
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};
Napi::Value AddonContext::SaveSequenceStateToFile(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    AddonContextSaveSequenceStateToFileWorker* worker = new AddonContextSaveSequenceStateToFileWorker(info, this);
    worker->Queue();
    return worker->GetPromise();
}

class AddonContextLoadSequenceStateFromFileWorker : public Napi::AsyncWorker {
    public:
        AddonContext* context;
        std::string filepath;
        llama_seq_id sequenceId;
        size_t maxContextSize;
        std::vector<llama_token> tokens;

        AddonContextLoadSequenceStateFromFileWorker(const Napi::CallbackInfo& info, AddonContext* context)
            : Napi::AsyncWorker(info.Env(), "AddonContextLoadSequenceStateFromFileWorker"),
              context(context),
              deferred(Napi::Promise::Deferred::New(info.Env())) {
            context->Ref();

            filepath = info[0].As<Napi::String>().Utf8Value();
            sequenceId = info[1].As<Napi::Number>().Int32Value();
            maxContextSize = info[2].As<Napi::Number>().Uint32Value();

            tokens.resize(maxContextSize);
        }
        ~AddonContextLoadSequenceStateFromFileWorker() {
            context->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                size_t tokenCount = 0;
                const size_t fileSize = llama_state_seq_load_file(context->ctx, filepath.c_str(), sequenceId, tokens.data(), tokens.size(), &tokenCount);
                if (fileSize == 0) {
                    SetError("Failed to load state from file. Current context sequence size may be smaller that the state of the file");
                    return;
                }

                tokens.resize(tokenCount);
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch(...) {
                SetError("Unknown error when calling \"llama_state_seq_load_file\"");
            }
        }
        void OnOK() {
            size_t tokenCount = tokens.size();
            Napi::Uint32Array result = Napi::Uint32Array::New(Env(), tokenCount);

            for (size_t i = 0; i < tokenCount; i++) {
                result[i] = tokens[i];
            }

            deferred.Resolve(result);
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};
Napi::Value AddonContext::LoadSequenceStateFromFile(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    AddonContextLoadSequenceStateFromFileWorker* worker = new AddonContextLoadSequenceStateFromFileWorker(info, this);
    worker->Queue();
    return worker->GetPromise();
}

Napi::Value AddonContext::PrintTimings(const Napi::CallbackInfo& info) {
    llama_perf_context_print(ctx);
    llama_perf_context_reset(ctx);
    return info.Env().Undefined();
}

Napi::Value AddonContext::EnsureDraftContextIsCompatibleForSpeculative(const Napi::CallbackInfo& info) {
    constexpr auto vocabSizeMaxDifference = 128; // SPEC_VOCAB_MAX_SIZE_DIFFERENCE
    constexpr auto vocabCheckStartTokenId = 5; // SPEC_VOCAB_CHECK_START_TOKEN_ID

    const AddonContext * draftContext = Napi::ObjectWrap<AddonContext>::Unwrap(info[0].As<Napi::Object>());
    const auto currentCtx = ctx;
    const auto draftCtx = draftContext->ctx;
    const auto currentModel = model->model;
    const auto draftModel = draftContext->model->model;
    const auto currentVocab = model->vocab;
    const auto draftVocab = draftContext->model->vocab;

    if (llama_vocab_type(currentVocab) != llama_vocab_type(draftVocab)) {
        Napi::Error::New(info.Env(), "Speculative draft model vocabulary type must match the target model vocabulary type").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    if (llama_vocab_get_add_bos(currentVocab) != llama_vocab_get_add_bos(draftVocab) ||
        llama_vocab_get_add_eos(currentVocab) != llama_vocab_get_add_eos(draftVocab) ||
        llama_vocab_bos(currentVocab) != llama_vocab_bos(draftVocab) ||
        llama_vocab_eos(currentVocab) != llama_vocab_eos(draftVocab)
    ) {
        Napi::Error::New(info.Env(), "Speculative draft model special tokens must match the target model special tokens").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    const int currentModelVocabSize = llama_vocab_n_tokens(currentVocab);
    const int draftModelVocabSize = llama_vocab_n_tokens(draftVocab);

    const int vocabDiff = std::abs(currentModelVocabSize - draftModelVocabSize);

    if (vocabDiff > vocabSizeMaxDifference) {
        Napi::Error::New(
            info.Env(),
            std::string("Speculative draft model vocabulary must closely match the target model vocabulary size (vocabulary size difference: ") +
            std::to_string(vocabDiff) + std::string(", max allowed: ") + std::to_string(vocabSizeMaxDifference) + std::string(")")
        ).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    const int minVocabSize = std::min(currentModelVocabSize, draftModelVocabSize);
    for (int i = vocabCheckStartTokenId; i < minVocabSize; ++i) {
        const char * currentTokenText = llama_vocab_get_text(currentVocab, i);
        const char * draftTokenText = llama_vocab_get_text(draftVocab, i);
        if (std::strcmp(currentTokenText, draftTokenText) != 0) {
            Napi::Error::New(
                info.Env(),
                std::string("Speculative draft model vocabulary must match the target model vocabulary, but token ") +
                std::to_string(i) + std::string(" content differs. Target: \"") + std::string(currentTokenText) +
                std::string("\", Draft: \"") + std::string(draftTokenText) + std::string("")
            ).ThrowAsJavaScriptException();
            return info.Env().Undefined();
        }
    }

    return info.Env().Undefined();
}

Napi::Value AddonContext::SetLoras(const Napi::CallbackInfo& info) {
    Napi::Array loraArray = info[0].As<Napi::Array>();
    Napi::Array scaleArray = info[1].As<Napi::Array>();

    std::vector<llama_adapter_lora *> loras;
    std::vector<float> scales;

    loras.reserve(loraArray.Length());
    scales.reserve(scaleArray.Length());

    for (size_t i = 0; i < loraArray.Length() && i < scaleArray.Length(); i++) {
        AddonModelLora* lora = Napi::ObjectWrap<AddonModelLora>::Unwrap(loraArray.Get(i).As<Napi::Object>());
        float scale = scaleArray.Get(i).As<Napi::Number>().FloatValue();

        loras.push_back(lora->lora_adapter);
        scales.push_back(scale);
    }

    llama_set_adapters_lora(ctx, loras.data(), loras.size(), scales.data());

    return info.Env().Undefined();
}

class RestoreCheckpointWorker : public Napi::AsyncWorker {
    public:
        AddonContext* context;
        AddonContextSequenceCheckpoint* checkpoint;
        std::size_t maxPosIndex;
        bool restoreSuccess = false;

        RestoreCheckpointWorker(const Napi::CallbackInfo& info, AddonContext* context, AddonContextSequenceCheckpoint* checkpoint, std::size_t maxPosIndex)
            : Napi::AsyncWorker(info.Env(), "RestoreCheckpointWorker"),
              context(context),
              checkpoint(checkpoint),
              maxPosIndex(maxPosIndex),
              deferred(Napi::Promise::Deferred::New(info.Env())) {
            context->Ref();
            checkpoint->Ref();
        }
        ~RestoreCheckpointWorker() {
            context->Unref();
            checkpoint->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                std::lock_guard<std::mutex> lock(checkpoint->dataMutex);

                std::size_t dataSize = checkpoint->data.size();
                std::size_t restoreSize = llama_state_seq_set_data_ext(context->ctx, checkpoint->data.data(), dataSize, checkpoint->sequenceId, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (restoreSize == dataSize) {
                    restoreSuccess = (
                        llama_memory_seq_rm(llama_get_memory(context->ctx), checkpoint->sequenceId, maxPosIndex + 1, -1) &&
                        llama_memory_seq_pos_max(llama_get_memory(context->ctx), checkpoint->sequenceId) == maxPosIndex
                    );
                }
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch(...) {
                SetError("Unknown error when calling \"llama_state_seq_set_data_ext\"");
            }
        }
        void OnOK() {
            deferred.Resolve(Napi::Boolean::New(Env(), restoreSuccess));
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};

Napi::Value AddonContext::RestoreCheckpoint(const Napi::CallbackInfo& info) {
    AddonContextSequenceCheckpoint* checkpoint = Napi::ObjectWrap<AddonContextSequenceCheckpoint>::Unwrap(info[0].As<Napi::Object>());
    std::size_t maxPosIndex = info[1].As<Napi::Number>().Int32Value();

    RestoreCheckpointWorker* worker = new RestoreCheckpointWorker(info, this, checkpoint, maxPosIndex);
    worker->Queue();
    return worker->GetPromise();
}

Napi::Value AddonContext::SetEmbeddingsPreNorm(const Napi::CallbackInfo& info) {
    if (disposed || !contextLoaded) {
        Napi::Error::New(info.Env(), "Context is disposed or not loaded").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    bool enabled = info[0].As<Napi::Boolean>().Value();
    bool masked = info[1].As<Napi::Boolean>().Value();
    llama_set_embeddings_pre_norm(ctx, enabled, masked);
    return info.Env().Undefined();
}

Napi::Value AddonContext::GetEmbeddingsPreNormIth(const Napi::CallbackInfo& info) {
    if (disposed || !contextLoaded) {
        Napi::Error::New(info.Env(), "Context is disposed or not loaded").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    int32_t i = info[0].As<Napi::Number>().Int32Value();
    const int n_embd = llama_model_n_embd(model->model);
    float* embd = llama_get_embeddings_pre_norm_ith(ctx, i);
    if (embd == nullptr) {
        return info.Env().Null();
    }
    auto buf = Napi::ArrayBuffer::New(info.Env(), n_embd * sizeof(float));
    std::memcpy(buf.Data(), embd, n_embd * sizeof(float));
    return Napi::Float32Array::New(info.Env(), n_embd, buf, 0);
}

Napi::Value AddonContext::InitMtpBatch(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    disposeBatch();

    int32_t n_tokens = info[0].As<Napi::Number>().Int32Value();
    int32_t n_embd = info[1].As<Napi::Number>().Int32Value();

    batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
    mtp_batch_token_owned = true;
    has_batch = true;
    batch_n_tokens = n_tokens;
    return info.Env().Undefined();
}

Napi::Value AddonContext::AddToMtpBatch(const Napi::CallbackInfo& info) {
    if (!has_batch) {
        Napi::Error::New(info.Env(), "No batch initialized").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    int32_t sequenceId = info[0].As<Napi::Number>().Int32Value();
    int32_t pos = info[1].As<Napi::Number>().Int32Value();
    llama_token token = info[2].As<Napi::Number>().Int32Value();
    Napi::Float32Array embd = info[3].As<Napi::Float32Array>();
    bool logits = info[4].As<Napi::Boolean>().Value();

    int n_embd = (int) embd.ElementLength();
    int idx = batch.n_tokens;

    batch.token[idx] = token;
    std::memcpy(batch.embd + (size_t)idx * n_embd, embd.Data(), n_embd * sizeof(float));
    batch.pos[idx] = pos;
    batch.n_seq_id[idx] = 1;
    batch.seq_id[idx][0] = sequenceId;
    batch.logits[idx] = logits ? 1 : 0;
    batch.n_tokens++;

    return Napi::Number::New(info.Env(), idx);
}

Napi::Value AddonContext::ClearBatch(const Napi::CallbackInfo& info) {
    if (!has_batch) {
        Napi::Error::New(info.Env(), "No batch initialized").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    batch.n_tokens = 0;
    return info.Env().Undefined();
}

Napi::Value AddonContext::GetModelNEmbd(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), llama_model_n_embd(model->model));
}

// ── PredictMtpTokens: entire MTP draft loop in one native call ──────
// Args: (sequenceId: number, startPos: number, startToken: number,
//        initialH: Float32Array, maxTokens: number, sampler: AddonSampler)
// Returns: Promise<Int32Array> of draft token ids.
class PredictMtpTokensWorker : public Napi::AsyncWorker {
public:
    AddonContext* ctx;
    AddonSampler* sampler;
    int32_t sequenceId;
    int32_t startPos;
    llama_token startToken;
    int32_t maxTokens;
    int32_t n_embd;
    std::vector<float> initialH;
    std::vector<llama_token> results;
    Napi::Promise::Deferred deferred;

    PredictMtpTokensWorker(const Napi::CallbackInfo& info, AddonContext* ctx)
        : Napi::AsyncWorker(info.Env(), "PredictMtpTokensWorker"),
          ctx(ctx),
          deferred(Napi::Promise::Deferred::New(info.Env())) {
        ctx->Ref();

        sequenceId = info[0].As<Napi::Number>().Int32Value();
        startPos   = info[1].As<Napi::Number>().Int32Value();
        startToken = info[2].As<Napi::Number>().Int32Value();
        auto embdArr = info[3].As<Napi::Float32Array>();
        maxTokens  = info[4].As<Napi::Number>().Int32Value();
        sampler    = Napi::ObjectWrap<AddonSampler>::Unwrap(info[5].As<Napi::Object>());
        sampler->Ref();
        // Must rebuild sampler chain on the main thread — NAPI calls are
        // not safe from the async worker thread.
        sampler->rebuildChainIfNeeded();

        n_embd = (int32_t) embdArr.ElementLength();
        initialH.resize(n_embd);
        std::memcpy(initialH.data(), embdArr.Data(), n_embd * sizeof(float));
    }
    ~PredictMtpTokensWorker() {
        ctx->Unref();
        sampler->Unref();
    }

    Napi::Promise GetPromise() { return deferred.Promise(); }

protected:
    void Execute() {
        try {
            auto & batch = ctx->batch;
            std::vector<float> currentH = initialH;
            llama_token currentToken = startToken;

            llama_memory_seq_rm(llama_get_memory(ctx->ctx), sequenceId, 0, -1);
            sampler->rebuildChainIfNeeded();

            for (int32_t i = 0; i < maxTokens; i++) {
                batch.n_tokens = 0;
                batch.token[0] = currentToken;
                std::memcpy(batch.embd, currentH.data(), n_embd * sizeof(float));
                batch.pos[0] = startPos + i + 1;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = sequenceId;
                batch.logits[0] = 1;
                batch.n_tokens = 1;

                int r = llama_decode(ctx->ctx, batch);
                if (r != 0) break;

                llama_token_data_array cur_p;
                sampler->sample(ctx->ctx, 0, cur_p, false);
                llama_token draftToken = (cur_p.selected >= 0 && cur_p.selected < (int32_t)cur_p.size)
                    ? cur_p.data[cur_p.selected].id : -1;
                if (draftToken == -1) break;

                results.push_back(draftToken);

                float* nextH = llama_get_embeddings_pre_norm_ith(ctx->ctx, 0);
                if (nextH == nullptr) break;

                currentH.assign(nextH, nextH + n_embd);
                currentToken = draftToken;
            }
        } catch (const std::exception& e) {
            SetError(e.what());
        } catch (...) {
            SetError("Unknown error in PredictMtpTokens");
        }
    }

    void OnOK() {
        auto env = Env();
        auto buf = Napi::ArrayBuffer::New(env, results.size() * sizeof(int32_t));
        if (!results.empty()) {
            std::memcpy(buf.Data(), results.data(), results.size() * sizeof(int32_t));
        }
        deferred.Resolve(Napi::Int32Array::New(env, results.size(), buf, 0));
    }
    void OnError(const Napi::Error& err) {
        deferred.Reject(err.Value());
    }
};

Napi::Value AddonContext::PredictMtpTokens(const Napi::CallbackInfo& info) {
    if (disposed || !contextLoaded || !has_batch) {
        Napi::Error::New(info.Env(), "Context is disposed, not loaded, or no MTP batch").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    auto* worker = new PredictMtpTokensWorker(info, this);
    worker->Queue();
    return worker->GetPromise();
}

void AddonContext::init(Napi::Object exports) {
    exports.Set(
        "AddonContext",
        DefineClass(
            exports.Env(),
            "AddonContext",
            {
                InstanceMethod("init", &AddonContext::Init),
                InstanceMethod("getContextSize", &AddonContext::GetContextSize),
                InstanceMethod("initBatch", &AddonContext::InitBatch),
                InstanceMethod("addToBatch", &AddonContext::AddToBatch),
                InstanceMethod("disposeSequence", &AddonContext::DisposeSequence),
                InstanceMethod("removeTokenCellsFromSequence", &AddonContext::RemoveTokenCellsFromSequence),
                InstanceMethod("shiftSequenceTokenCells", &AddonContext::ShiftSequenceTokenCells),
                InstanceMethod("getSequenceKvCacheMinPosition", &AddonContext::GetSequenceKvCacheMinPosition),
                InstanceMethod("getSequenceKvCacheMaxPosition", &AddonContext::GetSequenceKvCacheMaxPosition),
                InstanceMethod("decodeBatch", &AddonContext::DecodeBatch),
                InstanceMethod("sampleToken", &AddonContext::SampleToken),
                InstanceMethod("getEmbedding", &AddonContext::GetEmbedding),
                InstanceMethod("getStateSize", &AddonContext::GetStateSize),
                InstanceMethod("getMemoryBreakdown", &AddonContext::GetMemoryBreakdown),
                InstanceMethod("getThreads", &AddonContext::GetThreads),
                InstanceMethod("setThreads", &AddonContext::SetThreads),
                InstanceMethod("printTimings", &AddonContext::PrintTimings),
                InstanceMethod("ensureDraftContextIsCompatibleForSpeculative", &AddonContext::EnsureDraftContextIsCompatibleForSpeculative),
                InstanceMethod("saveSequenceStateToFile", &AddonContext::SaveSequenceStateToFile),
                InstanceMethod("loadSequenceStateFromFile", &AddonContext::LoadSequenceStateFromFile),
                InstanceMethod("setLoras", &AddonContext::SetLoras),
                InstanceMethod("restoreCheckpoint", &AddonContext::RestoreCheckpoint),
                InstanceMethod("setEmbeddingsPreNorm", &AddonContext::SetEmbeddingsPreNorm),
                InstanceMethod("getEmbeddingsPreNormIth", &AddonContext::GetEmbeddingsPreNormIth),
                InstanceMethod("initMtpBatch", &AddonContext::InitMtpBatch),
                InstanceMethod("addToMtpBatch", &AddonContext::AddToMtpBatch),
                InstanceMethod("clearBatch", &AddonContext::ClearBatch),
                InstanceMethod("getModelNEmbd", &AddonContext::GetModelNEmbd),
                InstanceMethod("predictMtpTokens", &AddonContext::PredictMtpTokens),
                InstanceMethod("dispose", &AddonContext::Dispose),
            }
        )
    );
}

AddonContextSequenceCheckpoint::AddonContextSequenceCheckpoint(const Napi::CallbackInfo& info) : Napi::ObjectWrap<AddonContextSequenceCheckpoint>(info) {

}
AddonContextSequenceCheckpoint::~AddonContextSequenceCheckpoint() {
    dispose();
}

class AddonContextSequenceCheckpointInitWorker : public Napi::AsyncWorker {
    public:
    AddonContextSequenceCheckpoint* checkpoint;
        AddonContext* context;

        AddonContextSequenceCheckpointInitWorker(const Napi::CallbackInfo& info, AddonContextSequenceCheckpoint* checkpoint, AddonContext* context)
            : Napi::AsyncWorker(info.Env(), "AddonContextSequenceCheckpointInitWorker"),
            checkpoint(checkpoint),
              context(context),
              deferred(Napi::Promise::Deferred::New(info.Env())) {
            checkpoint->Ref();
            context->Ref();
        }
        ~AddonContextSequenceCheckpointInitWorker() {
            checkpoint->Unref();
            context->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                checkpoint->minPos = llama_memory_seq_pos_min(llama_get_memory(context->ctx), checkpoint->sequenceId);
                checkpoint->maxPos = llama_memory_seq_pos_max(llama_get_memory(context->ctx), checkpoint->sequenceId);
                const size_t checkpointSize = llama_state_seq_get_size_ext(context->ctx, checkpoint->sequenceId, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                checkpoint->data.resize(checkpointSize, 0);
                llama_state_seq_get_data_ext(context->ctx, checkpoint->data.data(), checkpointSize, checkpoint->sequenceId, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch(...) {
                SetError("Unknown error when calling \"llama_state_seq_get_data_ext\"");
            }
        }
        void OnOK() {
            deferred.Resolve(Env().Undefined());
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};

Napi::Value AddonContextSequenceCheckpoint::Init(const Napi::CallbackInfo& info) {
    AddonContext * context = Napi::ObjectWrap<AddonContext>::Unwrap(info[0].As<Napi::Object>());
    sequenceId = info[1].As<Napi::Number>().Int32Value();

    AddonContextSequenceCheckpointInitWorker* worker = new AddonContextSequenceCheckpointInitWorker(info, this, context);
    worker->Queue();
    return worker->GetPromise();
}

Napi::Value AddonContextSequenceCheckpoint::Dispose(const Napi::CallbackInfo& info) {
    dispose();
    return info.Env().Undefined();
}

void AddonContextSequenceCheckpoint::dispose() {
    std::lock_guard<std::mutex> lock(dataMutex);
    data.clear();
    data.resize(0);
}

Napi::Value AddonContextSequenceCheckpoint::GetSize(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), data.size());
}

Napi::Value AddonContextSequenceCheckpoint::GetMinPos(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), minPos);
}

Napi::Value AddonContextSequenceCheckpoint::GetMaxPos(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), maxPos);
}

void AddonContextSequenceCheckpoint::init(Napi::Object exports) {
    exports.Set(
        "AddonContextSequenceCheckpoint",
        DefineClass(
            exports.Env(),
            "AddonContextSequenceCheckpoint",
            {
                InstanceMethod("init", &AddonContextSequenceCheckpoint::Init),
                InstanceMethod("dispose", &AddonContextSequenceCheckpoint::Dispose),

                InstanceAccessor("size", &AddonContextSequenceCheckpoint::GetSize, nullptr),
                InstanceAccessor("minPos", &AddonContextSequenceCheckpoint::GetMinPos, nullptr),
                InstanceAccessor("maxPos", &AddonContextSequenceCheckpoint::GetMaxPos, nullptr),
            }
        )
    );
}
