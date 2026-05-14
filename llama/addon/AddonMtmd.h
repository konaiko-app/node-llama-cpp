#pragma once

#include <string>
#include <mutex>
#include <cstdint>
#include "llama.h"
#include "napi.h"
#include "addonGlobals.h"

#ifdef NLC_HAS_MTMD
#include "mtmd.h"
#include "mtmd-helper.h"

class AddonMtmd : public Napi::ObjectWrap<AddonMtmd> {
    public:
        AddonModel* model;
        mtmd_context* mtmdCtx;
        std::string mmprojPath;
        bool useGpu;
        std::string mediaMarker;
        int nThreads;
        int imageMinTokens;
        int imageMaxTokens;
        uint32_t usages = 0;
        uint64_t loadedMmprojSize = 0;

        std::mutex encodeMutex;

        bool modelRefHeld = false;
        bool disposed = false;

        AddonMtmd(const Napi::CallbackInfo& info);
        ~AddonMtmd();

        void dispose();

        Napi::Value Init(const Napi::CallbackInfo& info);
        Napi::Value Dispose(const Napi::CallbackInfo& info);

        Napi::Value GetFilePath(const Napi::CallbackInfo& info);
        Napi::Value GetDisposed(const Napi::CallbackInfo& info);
        Napi::Value GetSupportsVision(const Napi::CallbackInfo& info);
        Napi::Value GetSupportsAudio(const Napi::CallbackInfo& info);

        Napi::Value GetUsages(const Napi::CallbackInfo& info);
        void SetUsages(const Napi::CallbackInfo& info, const Napi::Value &value);

        Napi::Value Tokenize(const Napi::CallbackInfo& info);
        Napi::Value EvalChunks(const Napi::CallbackInfo& info);

        static void init(Napi::Object exports);
};

#endif
