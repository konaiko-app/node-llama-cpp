#include "AddonMtmd.h"

#ifdef NLC_HAS_MTMD

#include <fstream>
#include "addonGlobals.h"
#include "AddonModel.h"
#include "AddonContext.h"

static uint64_t getFileSize(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) return 0;
    return static_cast<uint64_t>(f.tellg());
}


class AddonMtmdInitWorker : public Napi::AsyncWorker {
    public:
        AddonMtmd* addon;

        AddonMtmdInitWorker(const Napi::Env& env, AddonMtmd* addon)
            : Napi::AsyncWorker(env, "AddonMtmdInitWorker"),
              addon(addon),
              deferred(Napi::Promise::Deferred::New(env)) {
            addon->Ref();
            addon->model->Ref();
        }
        ~AddonMtmdInitWorker() {
            addon->model->Unref();
            addon->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                mtmd_context_params params = mtmd_context_params_default();
                params.use_gpu = addon->useGpu;
                params.print_timings = false;

                if (!addon->mediaMarker.empty()) {
                    params.media_marker = addon->mediaMarker.c_str();
                }
                if (addon->nThreads > 0) {
                    params.n_threads = addon->nThreads;
                }
                if (addon->imageMinTokens > 0) {
                    params.image_min_tokens = addon->imageMinTokens;
                }
                if (addon->imageMaxTokens > 0) {
                    params.image_max_tokens = addon->imageMaxTokens;
                }

                addon->mtmdCtx = mtmd_init_from_file(
                    addon->mmprojPath.c_str(),
                    addon->model->model,
                    params
                );

                if (addon->mtmdCtx == nullptr) {
                    SetError(
                        std::string("Failed to initialize multimodal projector \"")
                        + addon->mmprojPath + "\""
                    );
                } else {
                    addon->model->Ref();
                    addon->modelRefHeld = true;
                }
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch (...) {
                SetError("Unknown error when initializing mtmd context");
            }
        }
        void OnOK() {
            if (addon->mtmdCtx != nullptr) {
                addon->loadedMmprojSize = getFileSize(addon->mmprojPath);
                if (addon->loadedMmprojSize > 0) {
                    adjustNapiExternalMemoryAdd(Env(), addon->loadedMmprojSize);
                }
            }
            deferred.Resolve(Env().Undefined());
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};

class AddonMtmdDisposeWorker : public Napi::AsyncWorker {
    public:
        AddonMtmd* addon;
        uint64_t mmprojSizeToFree = 0;

        AddonMtmdDisposeWorker(const Napi::Env& env, AddonMtmd* addon)
            : Napi::AsyncWorker(env, "AddonMtmdDisposeWorker"),
              addon(addon),
              deferred(Napi::Promise::Deferred::New(env)) {
            addon->Ref();
        }
        ~AddonMtmdDisposeWorker() {
            addon->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                if (!addon->disposed && addon->mtmdCtx != nullptr) {
                    mtmd_free(addon->mtmdCtx);
                    addon->mtmdCtx = nullptr;
                    mmprojSizeToFree = addon->loadedMmprojSize;
                    addon->loadedMmprojSize = 0;
                }

                addon->dispose();
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch (...) {
                SetError("Unknown error when disposing mtmd context");
            }
        }
        void OnOK() {
            if (mmprojSizeToFree > 0) {
                adjustNapiExternalMemorySubtract(Env(), mmprojSizeToFree);
            }
            deferred.Resolve(Env().Undefined());
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};

struct MtmdBitmapInput {
    uint32_t width;
    uint32_t height;
    std::vector<unsigned char> data;
    bool isFile;
};

static void parseBitmapInputs(const Napi::Array& bitmapArray, std::vector<MtmdBitmapInput>& out) {
    out.resize(bitmapArray.Length());
    for (uint32_t i = 0; i < bitmapArray.Length(); i++) {
        Napi::Object bmpObj = bitmapArray.Get(i).As<Napi::Object>();
        if (bmpObj.Has("fileData")) {
            Napi::Uint8Array fileData = bmpObj.Get("fileData").As<Napi::Uint8Array>();
            out[i].data.assign(fileData.Data(), fileData.Data() + fileData.ElementLength());
            out[i].width = 0;
            out[i].height = 0;
            out[i].isFile = true;
        } else {
            out[i].width = bmpObj.Get("width").As<Napi::Number>().Uint32Value();
            out[i].height = bmpObj.Get("height").As<Napi::Number>().Uint32Value();
            Napi::Uint8Array data = bmpObj.Get("data").As<Napi::Uint8Array>();
            out[i].data.assign(data.Data(), data.Data() + data.ElementLength());
            out[i].isFile = false;
        }
    }
}

static bool buildBitmaps(
    mtmd_context* ctx,
    const std::vector<MtmdBitmapInput>& inputs,
    std::vector<mtmd_bitmap*>& out,
    std::string& errorMsg
) {
    out.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i].isFile) {
            out[i] = mtmd_helper_bitmap_init_from_buf(
                ctx,
                inputs[i].data.data(),
                inputs[i].data.size()
            );
        } else {
            size_t expectedSize = (size_t)inputs[i].width * inputs[i].height * 3;
            if (inputs[i].data.size() != expectedSize) {
                for (size_t j = 0; j < i; j++) mtmd_bitmap_free(out[j]);
                errorMsg = "Bitmap " + std::to_string(i) + " data size mismatch: expected "
                    + std::to_string(expectedSize) + " bytes (width * height * 3), got "
                    + std::to_string(inputs[i].data.size());
                return false;
            }
            out[i] = mtmd_bitmap_init(
                inputs[i].width,
                inputs[i].height,
                inputs[i].data.data()
            );
        }
        if (out[i] == nullptr) {
            for (size_t j = 0; j < i; j++) mtmd_bitmap_free(out[j]);
            errorMsg = "Failed to create bitmap " + std::to_string(i);
            return false;
        }
    }
    return true;
}

static void freeBitmaps(std::vector<mtmd_bitmap*>& bitmaps) {
    for (auto* bmp : bitmaps) mtmd_bitmap_free(bmp);
}

struct TokenizeChunkResult {
    mtmd_input_chunk_type type;
    std::vector<llama_token> tokens;
    std::vector<float> embeddings;
    size_t n_tokens;
    llama_pos n_pos;
};

class AddonMtmdTokenizeWorker : public Napi::AsyncWorker {
    public:
        AddonMtmd* addon;
        std::string text;
        bool addSpecial;
        bool parseSpecial;

        std::vector<MtmdBitmapInput> bitmaps;

        std::vector<TokenizeChunkResult> results;
        bool usesMRope = false;

        AddonMtmdTokenizeWorker(const Napi::CallbackInfo& info, AddonMtmd* addon)
            : Napi::AsyncWorker(info.Env(), "AddonMtmdTokenizeWorker"),
              addon(addon),
              deferred(Napi::Promise::Deferred::New(info.Env())) {
            addon->Ref();

            text = info[0].As<Napi::String>().Utf8Value();
            parseBitmapInputs(info[1].As<Napi::Array>(), bitmaps);

            if (info.Length() > 2 && info[2].IsObject()) {
                Napi::Object opts = info[2].As<Napi::Object>();
                addSpecial = opts.Has("addSpecial") ? opts.Get("addSpecial").As<Napi::Boolean>().Value() : true;
                parseSpecial = opts.Has("parseSpecial") ? opts.Get("parseSpecial").As<Napi::Boolean>().Value() : true;
            } else {
                addSpecial = true;
                parseSpecial = true;
            }
        }
        ~AddonMtmdTokenizeWorker() {
            addon->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                std::vector<mtmd_bitmap*> rawBitmaps;
                std::string bmpError;
                if (!buildBitmaps(addon->mtmdCtx, bitmaps, rawBitmaps, bmpError)) {
                    SetError(bmpError);
                    return;
                }

                mtmd_input_chunks* chunks = mtmd_input_chunks_init();
                mtmd_input_text inputText;
                inputText.text = text.c_str();
                inputText.add_special = addSpecial;
                inputText.parse_special = parseSpecial;

                int32_t res = mtmd_tokenize(
                    addon->mtmdCtx,
                    chunks,
                    &inputText,
                    (const mtmd_bitmap**)rawBitmaps.data(),
                    rawBitmaps.size()
                );

                freeBitmaps(rawBitmaps);

                if (res != 0) {
                    mtmd_input_chunks_free(chunks);
                    if (res == 1) {
                        SetError("Number of bitmaps does not match the number of media markers in the prompt");
                    } else if (res == 2) {
                        SetError("Image preprocessing error");
                    } else {
                        SetError("mtmd_tokenize failed with code " + std::to_string(res));
                    }
                    return;
                }

                usesMRope = mtmd_decode_use_mrope(addon->mtmdCtx);

                size_t nChunks = mtmd_input_chunks_size(chunks);
                results.resize(nChunks);

                for (size_t i = 0; i < nChunks; i++) {
                    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
                    results[i].type = mtmd_input_chunk_get_type(chunk);
                    results[i].n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
                    results[i].n_pos = mtmd_input_chunk_get_n_pos(chunk);

                    if (results[i].type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        size_t nTokens = 0;
                        const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &nTokens);
                        results[i].tokens.assign(tokens, tokens + nTokens);
                    } else {
                        // mtmd_encode_chunk is NOT thread-safe
                        std::lock_guard<std::mutex> lock(addon->encodeMutex);

                        int32_t encRes = mtmd_encode_chunk(addon->mtmdCtx, chunk);
                        if (encRes != 0) {
                            mtmd_input_chunks_free(chunks);
                            SetError("mtmd_encode_chunk failed for chunk " + std::to_string(i));
                            return;
                        }

                        float* embd = mtmd_get_output_embd(addon->mtmdCtx);
                        int n_embd = llama_model_n_embd_inp(addon->model->model);
                        size_t totalFloats = (size_t)n_embd * results[i].n_tokens;
                        results[i].embeddings.assign(embd, embd + totalFloats);
                    }
                }

                mtmd_input_chunks_free(chunks);
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch (...) {
                SetError("Unknown error in mtmd tokenize");
            }
        }

        void OnOK() {
            Napi::Env env = Env();
            Napi::Object result = Napi::Object::New(env);

            Napi::Array chunksArray = Napi::Array::New(env, results.size());
            for (size_t i = 0; i < results.size(); i++) {
                Napi::Object chunkObj = Napi::Object::New(env);

                switch (results[i].type) {
                    case MTMD_INPUT_CHUNK_TYPE_TEXT:
                        chunkObj.Set("type", Napi::String::New(env, "text"));
                        break;
                    case MTMD_INPUT_CHUNK_TYPE_IMAGE:
                        chunkObj.Set("type", Napi::String::New(env, "image"));
                        break;
                    case MTMD_INPUT_CHUNK_TYPE_AUDIO:
                        chunkObj.Set("type", Napi::String::New(env, "audio"));
                        break;
                    default:
                        chunkObj.Set("type", Napi::String::New(env, "unknown"));
                        break;
                }

                chunkObj.Set("nTokens", Napi::Number::New(env, results[i].n_tokens));
                chunkObj.Set("nPos", Napi::Number::New(env, results[i].n_pos));

                if (results[i].type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    Napi::Uint32Array tokens = Napi::Uint32Array::New(env, results[i].tokens.size());
                    for (size_t j = 0; j < results[i].tokens.size(); j++) {
                        tokens[j] = static_cast<uint32_t>(results[i].tokens[j]);
                    }
                    chunkObj.Set("tokens", tokens);
                } else {
                    Napi::Float32Array embeddings = Napi::Float32Array::New(env, results[i].embeddings.size());
                    for (size_t j = 0; j < results[i].embeddings.size(); j++) {
                        embeddings[j] = results[i].embeddings[j];
                    }
                    chunkObj.Set("embeddings", embeddings);
                }

                chunksArray.Set(i, chunkObj);
            }

            result.Set("chunks", chunksArray);
            result.Set("usesMRope", Napi::Boolean::New(env, usesMRope));

            deferred.Resolve(result);
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};


class AddonMtmdEvalChunksWorker : public Napi::AsyncWorker {
    public:
        AddonMtmd* addon;
        AddonContext* addonCtx;
        std::string text;
        bool addSpecial;
        bool parseSpecial;

        std::vector<MtmdBitmapInput> bitmaps;

        llama_pos n_past;
        llama_seq_id seq_id;
        llama_pos new_n_past = 0;

        AddonMtmdEvalChunksWorker(const Napi::CallbackInfo& info, AddonMtmd* addon)
            : Napi::AsyncWorker(info.Env(), "AddonMtmdEvalChunksWorker"),
              addon(addon),
              deferred(Napi::Promise::Deferred::New(info.Env())) {
            addon->Ref();

            addonCtx = Napi::ObjectWrap<AddonContext>::Unwrap(info[0].As<Napi::Object>());
            addonCtx->Ref();

            text = info[1].As<Napi::String>().Utf8Value();
            parseBitmapInputs(info[2].As<Napi::Array>(), bitmaps);

            n_past = info[3].As<Napi::Number>().Int32Value();
            seq_id = info[4].As<Napi::Number>().Int32Value();

            if (info.Length() > 5 && info[5].IsObject()) {
                Napi::Object opts = info[5].As<Napi::Object>();
                addSpecial = opts.Has("addSpecial") ? opts.Get("addSpecial").As<Napi::Boolean>().Value() : true;
                parseSpecial = opts.Has("parseSpecial") ? opts.Get("parseSpecial").As<Napi::Boolean>().Value() : true;
            } else {
                addSpecial = true;
                parseSpecial = true;
            }
        }
        ~AddonMtmdEvalChunksWorker() {
            addonCtx->Unref();
            addon->Unref();
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                std::vector<mtmd_bitmap*> rawBitmaps;
                std::string bmpError;
                if (!buildBitmaps(addon->mtmdCtx, bitmaps, rawBitmaps, bmpError)) {
                    SetError(bmpError);
                    return;
                }

                mtmd_input_chunks* chunks = mtmd_input_chunks_init();
                mtmd_input_text inputText;
                inputText.text = text.c_str();
                inputText.add_special = addSpecial;
                inputText.parse_special = parseSpecial;

                int32_t tokRes = mtmd_tokenize(
                    addon->mtmdCtx,
                    chunks,
                    &inputText,
                    (const mtmd_bitmap**)rawBitmaps.data(),
                    rawBitmaps.size()
                );

                freeBitmaps(rawBitmaps);

                if (tokRes != 0) {
                    mtmd_input_chunks_free(chunks);
                    SetError("mtmd_tokenize failed with code " + std::to_string(tokRes));
                    return;
                }

                // mtmd_helper_eval_chunks is NOT thread-safe
                std::lock_guard<std::mutex> lock(addon->encodeMutex);

                int32_t n_batch = addonCtx->context_params.n_batch;
                llama_pos updatedPast = n_past;
                int32_t evalRes = mtmd_helper_eval_chunks(
                    addon->mtmdCtx,
                    addonCtx->ctx,
                    chunks,
                    n_past,
                    seq_id,
                    n_batch,
                    true,  // logits_last
                    &updatedPast
                );

                mtmd_input_chunks_free(chunks);

                if (evalRes != 0) {
                    SetError("mtmd_helper_eval_chunks failed with code " + std::to_string(evalRes));
                    return;
                }

                new_n_past = updatedPast;
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch (...) {
                SetError("Unknown error in mtmd evalChunks");
            }
        }
        void OnOK() {
            deferred.Resolve(Napi::Number::New(Env(), new_n_past));
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};


AddonMtmd::AddonMtmd(const Napi::CallbackInfo& info) : Napi::ObjectWrap<AddonMtmd>(info) {
    model = Napi::ObjectWrap<AddonModel>::Unwrap(info[0].As<Napi::Object>());
    mmprojPath = info[1].As<Napi::String>().Utf8Value();
    mtmdCtx = nullptr;

    useGpu = true;
    nThreads = 0;
    imageMinTokens = 0;
    imageMaxTokens = 0;

    if (info.Length() > 2 && info[2].IsObject()) {
        Napi::Object opts = info[2].As<Napi::Object>();
        if (opts.Has("useGpu")) {
            useGpu = opts.Get("useGpu").As<Napi::Boolean>().Value();
        }
        if (opts.Has("mediaMarker")) {
            mediaMarker = opts.Get("mediaMarker").As<Napi::String>().Utf8Value();
        }
        if (opts.Has("threads")) {
            nThreads = opts.Get("threads").As<Napi::Number>().Int32Value();
        }
        if (opts.Has("imageMinTokens")) {
            imageMinTokens = opts.Get("imageMinTokens").As<Napi::Number>().Int32Value();
        }
        if (opts.Has("imageMaxTokens")) {
            imageMaxTokens = opts.Get("imageMaxTokens").As<Napi::Number>().Int32Value();
        }
    }
}

AddonMtmd::~AddonMtmd() {
    dispose();
}

void AddonMtmd::dispose() {
    if (disposed) return;
    disposed = true;

    if (mtmdCtx != nullptr) {
        mtmd_free(mtmdCtx);
        mtmdCtx = nullptr;
    }

    if (loadedMmprojSize > 0) {
        adjustNapiExternalMemorySubtract(Env(), loadedMmprojSize);
        loadedMmprojSize = 0;
    }

    if (modelRefHeld) {
        modelRefHeld = false;
        model->Unref();
    }
}

Napi::Value AddonMtmd::Init(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Mtmd context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    AddonMtmdInitWorker* worker = new AddonMtmdInitWorker(this->Env(), this);
    worker->Queue();
    return worker->GetPromise();
}

Napi::Value AddonMtmd::Dispose(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(info.Env());
        deferred.Resolve(info.Env().Undefined());
        return deferred.Promise();
    }

    AddonMtmdDisposeWorker* worker = new AddonMtmdDisposeWorker(this->Env(), this);
    worker->Queue();
    return worker->GetPromise();
}

Napi::Value AddonMtmd::GetFilePath(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), mmprojPath);
}

Napi::Value AddonMtmd::GetDisposed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), disposed || mtmdCtx == nullptr);
}

Napi::Value AddonMtmd::GetSupportsVision(const Napi::CallbackInfo& info) {
    if (mtmdCtx == nullptr) return Napi::Boolean::New(info.Env(), false);
    return Napi::Boolean::New(info.Env(), mtmd_support_vision(mtmdCtx));
}

Napi::Value AddonMtmd::GetSupportsAudio(const Napi::CallbackInfo& info) {
    if (mtmdCtx == nullptr) return Napi::Boolean::New(info.Env(), false);
    return Napi::Boolean::New(info.Env(), mtmd_support_audio(mtmdCtx));
}

Napi::Value AddonMtmd::GetUsages(const Napi::CallbackInfo& info) {
    return Napi::Number::From(info.Env(), usages);
}

void AddonMtmd::SetUsages(const Napi::CallbackInfo& info, const Napi::Value &value) {
    usages = value.As<Napi::Number>().Uint32Value();
}

Napi::Value AddonMtmd::Tokenize(const Napi::CallbackInfo& info) {
    if (disposed || mtmdCtx == nullptr) {
        Napi::Error::New(info.Env(), "Mtmd context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    AddonMtmdTokenizeWorker* worker = new AddonMtmdTokenizeWorker(info, this);
    worker->Queue();
    return worker->GetPromise();
}

Napi::Value AddonMtmd::EvalChunks(const Napi::CallbackInfo& info) {
    if (disposed || mtmdCtx == nullptr) {
        Napi::Error::New(info.Env(), "Mtmd context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    AddonMtmdEvalChunksWorker* worker = new AddonMtmdEvalChunksWorker(info, this);
    worker->Queue();
    return worker->GetPromise();
}

void AddonMtmd::init(Napi::Object exports) {
    exports.Set(
        "AddonMtmd",
        DefineClass(
            exports.Env(),
            "AddonMtmd",
            {
                InstanceMethod("init", &AddonMtmd::Init),
                InstanceMethod("tokenize", &AddonMtmd::Tokenize),
                InstanceMethod("evalChunks", &AddonMtmd::EvalChunks),
                InstanceAccessor("usages", &AddonMtmd::GetUsages, &AddonMtmd::SetUsages),
                InstanceAccessor("filePath", &AddonMtmd::GetFilePath, nullptr),
                InstanceAccessor("disposed", &AddonMtmd::GetDisposed, nullptr),
                InstanceAccessor("supportsVision", &AddonMtmd::GetSupportsVision, nullptr),
                InstanceAccessor("supportsAudio", &AddonMtmd::GetSupportsAudio, nullptr),
                InstanceMethod("dispose", &AddonMtmd::Dispose),
            }
        )
    );
}

#endif
