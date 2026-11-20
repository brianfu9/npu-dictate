// asr_decode — C11 LLM decode driver for an exported LiteRT-LM decoder.
//
// Runs a three-section .litertlm decoder (prefill_decode / embedder / aux
// .tflite files) from precomputed prefill embeddings and greedy-decodes,
// mirroring litert-torch's Python sampling Executor exactly
// (litert_torch/generative/export_hf/experimental/calib/sampling_executor.py):
//   * prefill_chunk bucket choice (smallest bucket >= remaining, else the
//     largest), right-padding embeddings and token ids with zeros,
//   * valid_mask (true for the real tokens of the chunk),
//   * time_step / input_pos progression (positions = arange per chunk; after
//     the last prefill chunk time_step is rolled back to T-1 so decode
//     re-embeds the last prompt token first),
//   * the mask / rope aux signatures (the *_cache_update aux signatures are
//     discovered but never run: the KV merge happens on the host),
//   * logits output + greedy argmax (sample_logits), next-token embedding
//     via the decode_embedder signature.
//
// KV cache (LiteRT-LM NPU executor style): one persistent LiteRT tensor
// buffer per kv_cache_k_<i>/kv_cache_v_<i>, created once in the model's
// native dtype and quantization (as declared by the decode signature input)
// and bound via LiteRtDuplicateTensorBuffer to every prefill_<N> / decode
// input that takes it, including signatures living in the second
// prefill_decode file. It is initialized the way the oracle's init_cache
// produces (zeros in float -> the quantized value of 0.0, i.e. the zero
// point for int16). After each prefill/decode run the newly produced
// kv_slice outputs are merged into the persistent buffers in place on the
// host: the exported *_cache_update subgraph copies the ENTIRE slice to
// [input_pos[0], input_pos[0] + S) (verified against the model: its
// valid_mask input does not gate the copy), so the host writes all S
// positions of the chunk, which keeps the cache contents identical to the
// oracle's cache_update round trip. A raw element copy is used when the
// slice and the cache have the same dtype and the same scale/zero_point
// (checked per tensor at startup); otherwise the elements are requantized
// (dequant with the slice params, quant with the cache params, same
// rounding as the quantize helper below). A prefill/decode cache input
// whose dtype differs from the persistent buffer's (e.g. a float32
// kv_cache input on a quantized model) keeps its own buffer and is filled
// by dequantizing the persistent buffer just before the run; no other host
// quantize/dequantize of KV happens.
//
// Quantized models (the oracle's try_run_signature_with_quant_dequant): at
// every other signature boundary (rope/mask/embeddings into prefill/decode),
// float32 values are quantized into non-float inputs with that input
// tensor's own per-tensor scale/zero_point (round-half-even, narrow-range
// clip for symmetric >= 8-bit types, matching ai_edge_quantizer's
// uniform_quantize) and non-float outputs are dequantized to float32
// (uniform_dequantize). Float32 inputs/outputs pass through; integer inputs
// without quantization params (token ids, positions, time_step, bool masks)
// are passed as is. Mask and rope outputs are passed to the runner
// bit-for-bit: the oracle calls those runners without quant/dequant
// wrapping. Embeddings inputs stay float. A value produced by one
// signature's output that feeds a signature input with the same element
// type AND the same per-tensor quantization params (scale, zero_point) is
// copied as raw bytes instead of dequantize -> float32 -> quantize (with
// matching params that round trip is the identity for in-range values).
//
// Logits: when the decode logits output is int16 per-tensor quantized with
// a positive scale, greedy argmax runs directly on the raw int16 values
// (monotonic in the dequantized value, same index) and the row is
// dequantized to float32 only when --dump-logits is given. Ties always go
// to the last index among equal maxima (np.argsort(...)[..., -1]).
// Per-channel quantized inputs/outputs are rejected with an error.
//
// Each signature's input/output LiteRT tensor buffers are created once and
// reused for every run (no per-step create/destroy).
//
// Usage:
//   asr_decode --model-dir D
//              (--embeds F --ids F
//               | --audio-features F --prompt-ids F [--audio-pad-id ID=151676]
//                 [--dump-embeds F])
//              [--max-new N=64] [--cache-length L]
//              [--eos 151643,151645] [--accel cpu|npu] [--dump-logits F]
//              [--profile]
//   asr_decode --model-dir D --server
//              [--max-new N=64] [--cache-length L]
//              [--eos 151643,151645] [--accel cpu|npu] [--profile]
//
// --embeds/--ids feed precomputed prefill embeddings. --audio-features/
// --prompt-ids build them on-device instead: the decode_embedder runs for
// every prompt id, the rows whose id equals --audio-pad-id are replaced by
// the audio feature rows in order (mirroring make_feat_fixture.py), and the
// prefill/decode then proceed exactly as with --embeds.
//
// --server runs the setup (environment, model load, compile, buffer
// allocation) once, prints `READY setup_ms=<x>` on stdout and then serves
// transcription requests from stdin, one per line, until EOF or a line
// `QUIT`:
//   <audio_features_path> <prompt_ids_path> [max_new] [dump_logits_path]
// (space separated; max_new defaults to --max-new; with dump_logits_path
// the logits rows are written like --dump-logits). Each request resets all
// per-request state to what a fresh process starts with (the KV cache back
// to its initial value, time step 0, sampled tokens, logits/argmax state,
// the prefill bookkeeping) and then runs exactly the one-shot
// --audio-features/--prompt-ids pipeline, printing the one-shot output
// lines (the TOKENS line and the timing line; with --profile the per-stage
// profile, reset per request) followed by a line `DONE`. A malformed or
// failing request (missing file, pad-count mismatch, ...) prints
// `ERROR <message>` then `DONE` and the server keeps running; only the
// fatal LiteRT errors exit.
//
// D must contain exactly one file whose name contains each of the substrings
// "prefill_decode", "embedder" and "aux". Prefill buckets are discovered from
// the prefill_decode model's signature names prefill_<N>; the cache length is
// discovered from the kv_cache input shapes of the decode signature.
//
// The NPU prefill mishandles a non-empty cache (a second prefill chunk gives
// garbage on the MediaTek NPU; the CPU reference is correct), so a prompt
// longer than the largest prefill bucket runs ONE prefill chunk over the
// first positions and then one decode-signature step per remaining prompt
// position (no sampling; the step's `embeddings` input is the pre-built row
// of the assembled embeddings, audio rows included) before the normal decode
// starts. The tail time counts into prefill_ms. ASR_PREFILL_CHUNK (diagnostic)
// keeps the old multi-chunk prefill; ASR_PREFILL_SPLIT=N (diagnostic, N >= 1)
// splits after position min(N, max bucket) even when the prompt fits one
// bucket. A prompt fails (recoverable) when T + max_new exceeds the cache
// length.
//
// Output (stdout):
//   TOKENS <space separated ids>
//   prefill_ms=<total> chunks=<n> [tail_steps=<n> tail_ms=<x>] decode_tokens=<k> decode_ms_per_tok=<avg> setup_ms=<model load+compile>
//   (tail_steps/tail_ms are printed only when the prompt tail ran; the tail
//   time is inside prefill_ms)
//   (with --profile) one line per stage:
//     profile <stage> total_ms=<x> calls=<n> avg_ms=<y>
//   for the assemble (the on-device embedding assembly: the
//   decode_embedder runs + the feature splice), the embedder / aux mask /
//   aux rope / aux cache_update / prefill /
//   decode runs (the LiteRtRunCompiledModel call only; the aux cache_update
//   runs now total 0 calls, the merge is the host cache_update stage), the
//   host cache_update (the in-place merge into the persistent KV buffers),
//   the input quantize+copy into tensor buffers, the output
//   dequantize+copy out of tensor buffers, the argmax, the
//   per-decode-step residual ("other"), the prompt-tail steps (their total
//   wall time in the "tail" stage), plus the total wall time of the
//   first 3 and last 3 decode steps.
//
// --dump-logits F writes one float32 row of vocab size per decode step (same
// layout as ref_logits.f32).
//
// Accelerator: the dispatcher/compiler plugin directory is $DISPATCH_DIR, or
// the executable's directory when unset (exactly like litert_run.c).
// --accel cpu selects kLiteRtHwAcceleratorCpu (AOT-compiled NPU sections
// dispatch automatically); --accel npu also sets the NPU bit.

#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

#define APP "asr_decode"

// Not every libLiteRt.so build exports LiteRtGetStatusString (e.g. the one
// shipped in the Python wheel), so link it weakly and fall back to a plain
// status code.
extern const char *LiteRtGetStatusString(LiteRtStatus status)
    __attribute__((weak));

static const char *status_str(LiteRtStatus s) {
    if (LiteRtGetStatusString != NULL) {
        return LiteRtGetStatusString(s);
    }
    return "<status>";
}

// Check a LiteRt call: print the call text and the status on failure, then
// exit nonzero.
#define CHECK_LITERT(expr)                                              \
    do {                                                                \
        LiteRtStatus s_ = (expr);                                       \
        if (s_ != kLiteRtStatusOk) {                                    \
            fprintf(stderr, APP ": FAILED: %s -> %d (%s)\n", #expr,     \
                    (int)s_, status_str(s_));                           \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// ---------------------------------------------------------------------------
// Failure handling. Outside a --server request, fail() prints the message
// and exits, exactly like the original die() (exit 1) and
// fprintf+exit(2) error sites. Inside a server request the same failure sets
// the request error (reported on the `ERROR <message>` stdout line) and
// longjmps back to the request boundary so the server keeps serving. Only
// the fatal LiteRT paths (CHECK_LITERT, the compiled-model run) still exit.
// ---------------------------------------------------------------------------

static jmp_buf g_req_jmp;
static int g_in_request = 0;
static char g_req_err[512];

static void fail(int code, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0) {
        n = 0;
    }
    if (n >= (int)sizeof buf) {
        n = (int)sizeof buf - 1;
    }
    buf[n] = '\0';
    fprintf(stderr, APP ": %s\n", buf);
    if (g_in_request) {
        snprintf(g_req_err, sizeof g_req_err, "%s", buf);
        longjmp(g_req_jmp, 1);
    }
    exit(code);
}

static void die(const char *msg) {
    fail(1, "%s", msg);
}

static void die_usage(const char *msg) {
    fprintf(stderr, APP ": %s\n", msg);
    fprintf(stderr,
            "Usage: " APP " --model-dir D\n"
            "                 --embeds F --ids F\n"
            "                 | --audio-features F --prompt-ids F\n"
            "                   [--audio-pad-id ID=151676] [--dump-embeds F]\n"
            "                 [--max-new N=64] [--cache-length L]\n"
            "                 [--eos 151643,151645] [--accel cpu|npu]\n"
            "                 [--dump-logits F] [--profile]\n"
            "       " APP " --model-dir D --server\n"
            "                 [--max-new N=64] [--cache-length L]\n"
            "                 [--eos 151643,151645] [--accel cpu|npu]\n"
            "                 [--profile]\n"
            "  --server: READY on stdout, then one request per stdin line\n"
            "  until EOF or QUIT: <audio_features_path> <prompt_ids_path>\n"
            "  [max_new] [dump_logits_path]; each prints the one-shot\n"
            "  output lines (ERROR <msg> when it fails) then DONE.\n");
    exit(2);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

// ---------------------------------------------------------------------------
// Profiling (--profile): per-stage wall time (CLOCK_MONOTONIC) plus the
// total wall time of each decode step (the first 3 and last 3 are printed).
// With --profile off, tnow() takes no clock and every tick adds 0.0.
// ---------------------------------------------------------------------------

typedef enum {
    STG_EMBEDDER = 0,  // decode_embedder run
    STG_ASSEMBLE,      // on-device embedding assembly (embedder runs + splice)
    STG_AUX_MASK,      // aux mask runs (prefill_mask_*, decode_mask)
    STG_AUX_ROPE,      // aux rope runs
    STG_AUX_CACHE_UPDATE,  // aux *_cache_update runs (0 calls: host update)
    STG_CACHE_UPDATE,  // host in-place merge of kv_slice into the persistent KV
    STG_PREFILL,       // prefill_<N> run
    STG_DECODE,        // decode run
    STG_INPUT_QUANT,   // quantize+copy of inputs into the tensor buffers
    STG_OUTPUT_DEQUANT,// dequantize+copy of outputs out of the tensor buffers
    STG_ARGMAX,        // greedy argmax
    STG_OTHER,         // per decode step: everything not in the stages above
    STG_TAIL,          // prompt-tail steps: total wall time per step
    STG_COUNT
} StageId;

static const char *const g_stage_names[STG_COUNT] = {
    "embedder", "assemble", "aux_mask", "aux_rope", "aux_cache_update",
    "cache_update", "prefill", "decode", "input_quant", "output_dequant",
    "argmax", "other", "tail",
};

typedef struct {
    double total_ms;
    long long calls;
} StageStat;

static int g_profile = 0;
static StageStat g_stage[STG_COUNT];
static double *g_step_ms = NULL;  // total wall time per decode step
static size_t g_num_steps = 0;
static size_t g_step_cap = 0;

static inline double tnow(void) {
    return g_profile ? now_ms() : 0.0;
}

static void tick(int stg, double ms) {
    g_stage[stg].total_ms += ms;
    g_stage[stg].calls++;
}

static void record_step(double total_ms) {
    if (g_num_steps >= g_step_cap) {
        size_t ncap = g_step_cap ? g_step_cap * 2 : 16;
        double *p = (double *)realloc(g_step_ms, ncap * sizeof(double));
        if (!p) {
            die("out of memory");
        }
        g_step_ms = p;
        g_step_cap = ncap;
    }
    g_step_ms[g_num_steps++] = total_ms;
}

static void print_profile(void) {
    for (int i = 0; i < STG_COUNT; i++) {
        double avg = g_stage[i].calls > 0
                         ? g_stage[i].total_ms / (double)g_stage[i].calls
                         : 0.0;
        printf("profile %s total_ms=%.2f calls=%lld avg_ms=%.3f\n",
               g_stage_names[i], g_stage[i].total_ms, g_stage[i].calls, avg);
    }
    for (size_t k = 0; k < g_num_steps && k < 3; k++) {
        printf("profile decode_step%zu total_ms=%.2f\n", k, g_step_ms[k]);
    }
    for (long k = (long)g_num_steps - 3; k < (long)g_num_steps; k++) {
        if (k < 0 || k < 3) {
            continue;
        }
        printf("profile decode_step%ld total_ms=%.2f\n", k, g_step_ms[(size_t)k]);
    }
}

// ---------------------------------------------------------------------------
// Element-type helpers (raw storage format; quantization handled separately).
// ---------------------------------------------------------------------------

static size_t elem_size(LiteRtElementType t) {
    switch (t) {
        case kLiteRtElementTypeFloat32:
        case kLiteRtElementTypeInt32:
        case kLiteRtElementTypeUInt32:
            return 4;
        case kLiteRtElementTypeFloat16:
        case kLiteRtElementTypeInt16:
        case kLiteRtElementTypeUInt16:
            return 2;
        case kLiteRtElementTypeInt8:
        case kLiteRtElementTypeUInt8:
        case kLiteRtElementTypeBool:
            return 1;
        case kLiteRtElementTypeInt64:
        case kLiteRtElementTypeUInt64:
            return 8;
        default:
            return 0;
    }
}

// IEEE 754 half -> single, bit-level (no libm dependency).
static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            int e = 127 - 15 + 1;  // normalize the subnormal
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3FFu;
            bits = sign | ((uint32_t)e << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / NaN
    } else {
        bits = sign | (((exp - 15u) + 127u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

// Single -> IEEE 754 half, round-to-nearest-even.
static uint16_t float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) {  // inf / NaN
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);  // overflow -> inf
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;  // underflow to zero
        }
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t q = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (q & 1u))) {
            q++;
        }
        return (uint16_t)(sign | q);
    }
    uint32_t q = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) {
        q++;
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | q);
}

// ---------------------------------------------------------------------------
// Tensor types with quantization parameters.
// ---------------------------------------------------------------------------

typedef enum { Q_NONE = 0, Q_PER_TENSOR = 1, Q_PER_CHANNEL = 2 } QKind;

typedef struct {
    LiteRtElementType elem;
    LiteRtLayout layout;
    size_t n;  // number of elements
    QKind qkind;
    float scale;
    int64_t zp;
    int32_t quant_dim;
    uint64_t num_channels;
    float *ch_scale;  // owned
    int64_t *ch_zp;   // owned
} TType;

static void ttype_free(TType *t) {
    free(t->ch_scale);
    free(t->ch_zp);
    t->ch_scale = NULL;
    t->ch_zp = NULL;
}

// Deep copy (per-channel parameter arrays are copied too).
static void ttype_copy(TType *dst, const TType *src) {
    *dst = *src;
    if (src->ch_scale != NULL) {
        dst->ch_scale = (float *)malloc((size_t)src->num_channels *
                                        sizeof(float));
        dst->ch_zp = (int64_t *)malloc((size_t)src->num_channels *
                                       sizeof(int64_t));
        if (!dst->ch_scale || !dst->ch_zp) {
            die("out of memory");
        }
        memcpy(dst->ch_scale, src->ch_scale,
               (size_t)src->num_channels * sizeof(float));
        memcpy(dst->ch_zp, src->ch_zp, (size_t)src->num_channels *
                                       sizeof(int64_t));
    }
}

// Fill the TType from a LiteRT tensor handle. Only integer element types can
// carry quantization; float32/float16 slots are always handled as plain
// floats (mirroring the oracle's is_quantized(), which skips float32).
static void ttype_from_tensor(LiteRtTensor tensor, TType *t) {
    memset(t, 0, sizeof *t);
    LiteRtTensorTypeId type_id;
    CHECK_LITERT(LiteRtGetTensorTypeId(tensor, &type_id));
    if (type_id != kLiteRtRankedTensorType) {
        die("tensor is unranked; only ranked tensors are supported");
    }
    LiteRtRankedTensorType rt;
    CHECK_LITERT(LiteRtGetRankedTensorType(tensor, &rt));
    t->elem = rt.element_type;
    t->layout = rt.layout;
    if (elem_size(t->elem) == 0) {
        die("unsupported element type");
    }
    t->n = 1;
    for (unsigned d = 0; d < t->layout.rank; d++) {
        int32_t dim = t->layout.dimensions[d];
        if (dim < 0) {
            die("tensor has a dynamic dimension; cannot determine buffer "
                "size");
        }
        t->n *= (size_t)dim;
    }
    t->qkind = Q_NONE;
    LiteRtQuantizationTypeId qid;
    CHECK_LITERT(LiteRtGetQuantizationTypeId(tensor, &qid));
    if (qid == kLiteRtQuantizationPerTensor) {
        LiteRtQuantizationPerTensor q;
        CHECK_LITERT(LiteRtGetPerTensorQuantization(tensor, &q));
        t->qkind = Q_PER_TENSOR;
        t->scale = q.scale;
        t->zp = q.zero_point;
    } else if (qid == kLiteRtQuantizationPerChannel) {
        LiteRtQuantizationPerChannel q;
        CHECK_LITERT(LiteRtGetPerChannelQuantization(tensor, &q));
        t->qkind = Q_PER_CHANNEL;
        t->quant_dim = q.quantized_dimension;
        t->num_channels = q.num_channels;
        if (q.num_channels > 0) {
            t->ch_scale = (float *)malloc((size_t)q.num_channels *
                                          sizeof(float));
            t->ch_zp = (int64_t *)malloc((size_t)q.num_channels *
                                         sizeof(int64_t));
            if (!t->ch_scale || !t->ch_zp) {
                die("out of memory");
            }
            memcpy(t->ch_scale, q.scales, (size_t)q.num_channels *
                                           sizeof(float));
            memcpy(t->ch_zp, q.zero_points, (size_t)q.num_channels *
                                             sizeof(int64_t));
        }
    } else if (qid == kLiteRtQuantizationBlockWise) {
        die("blockwise quantization is not supported");
    }
    // Float element types are stored as plain floats.
    if (t->elem == kLiteRtElementTypeFloat32 ||
        t->elem == kLiteRtElementTypeFloat16) {
        ttype_free(t);
        t->qkind = Q_NONE;
    }
}

// Check that two occurrences of the same logical tensor have the same shape.
// Element type and quantization params are intentionally NOT compared: the
// same KV cache tensor may be a float32 input in one signature and a
// quantized int16 input/output in another (the oracle keeps the state as
// float32 and converts at each boundary with the tensor's own params).
static void ttype_check_shape(const TType *a, const TType *b, const char *na,
                              const char *nb) {
    if (a->n != b->n || a->layout.rank != b->layout.rank) {
        fail(2, "tensor '%s' and '%s' have mismatched shapes", na, nb);
    }
    for (unsigned d = 0; d < a->layout.rank; d++) {
        if (a->layout.dimensions[d] != b->layout.dimensions[d]) {
            fail(2, "tensor '%s' and '%s' have mismatched dimension %u", na,
                 nb, d);
        }
    }
}

// Bit compatibility: two tensors whose raw bytes can be handed to each
// other without conversion - same element type and the same per-tensor
// quantization params (scale and zero_point). The oracle's dequantize ->
// quantize round trip with matching params is the identity for in-range
// values (integer (q - zp) scaled and scaled back, re-rounded), so a raw
// copy between such a producer and a consumer is bit-identical.
static int ttype_bitwise_same(const TType *a, const TType *b) {
    if (a->elem != b->elem || a->qkind != b->qkind) {
        return 0;
    }
    if (a->qkind == Q_PER_TENSOR) {
        uint32_t sa, sb;
        memcpy(&sa, &a->scale, sizeof sa);
        memcpy(&sb, &b->scale, sizeof sb);
        return sa == sb && a->zp == b->zp;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Signatures and slots.
// ---------------------------------------------------------------------------

typedef struct {
    char name[128];
    TType t;
    LiteRtTensorBuffer buf;
    int owns_buf;
    int cache_direct;  // buf is a duplicate of the persistent KV cache buffer
    float *f32;  // owned float32 staging of the output values (outputs only)
} Slot;

typedef struct {
    char key[128];
    int section_idx;
    LiteRtParamIndex idx;
    Slot *in;
    size_t num_in;
    Slot *out;
    size_t num_out;
    LiteRtTensorBuffer *in_bufs;
    LiteRtTensorBuffer *out_bufs;
    int chain;  // 0: a real signature; k > 0: composite of g_chains[k - 1] (layer chunks)
} Sig;

typedef struct {
    char path[1024];
    const char *role;
    LiteRtModel model;
    LiteRtCompiledModel compiled;
    Sig *sigs;
    size_t num_sigs;
    int loaded;  // set by load_section; loops over all sections skip empty slots
} Section;

#define SEC_PD 0  // prefill_decode
#define SEC_ED 1  // embedder
#define SEC_AU 2  // aux
#define SEC_PD2 3  // optional 2nd prefill_decode file: a model compiled one signature per file
#define SEC_PLE 4  // optional per_layer_embedder (Gemma 3n/4 per-layer embeddings), CPU
#define NUM_SEC_SLOTS 5

static LiteRtEnvironment g_env = NULL;
static Section g_sections[NUM_SEC_SLOTS];
static int g_num_sections = 3;

static Sig *find_sig(int section_idx, const char *key) {
    Section *sec = &g_sections[section_idx];
    for (size_t i = 0; i < sec->num_sigs; i++) {
        if (strcmp(sec->sigs[i].key, key) == 0) {
            return &sec->sigs[i];
        }
    }
    // prefill_decode may be split across two files (one compiled network each).
    if (section_idx == SEC_PD && g_num_sections > SEC_PD2) {
        return find_sig(SEC_PD2, key);
    }
    return NULL;
}

static Slot *sig_in_by_name(Sig *sig, const char *name) {
    for (size_t i = 0; i < sig->num_in; i++) {
        if (strcmp(sig->in[i].name, name) == 0) {
            return &sig->in[i];
        }
    }
    return NULL;
}

static const Slot *sig_out_by_name(const Sig *sig, const char *name) {
    for (size_t i = 0; i < sig->num_out; i++) {
        if (strcmp(sig->out[i].name, name) == 0) {
            return &sig->out[i];
        }
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// KV cache set: the decode runner's inputs whose names contain "cache"
// (the oracle's init_cache). Each cache tensor has ONE persistent LiteRT
// tensor buffer in the model's native dtype and quantization (as declared
// by the decode signature input), created once, initialized the way the
// oracle's init_cache produces (zeros in float -> the quantized value of
// 0.0, i.e. the zero point for int16) and bound via
// LiteRtDuplicateTensorBuffer to every prefill_<N> / decode input that
// takes it, including signatures in the second prefill_decode file. After
// each prefill/decode run the fresh kv_slice outputs are merged into it
// in place on the host (host_cache_update); the *_cache_update aux
// signatures are never run.
// ---------------------------------------------------------------------------

typedef struct {
    char name[128];
    char slice_name[128];  // the matching runner output (kv_slice_*)
    TType t;  // native type (the decode signature input declaration)
    LiteRtTensorBuffer buf;  // persistent buffer, shared via duplicates
    int time_dim;  // index of the dimension sized to the cache length
    int raw_copy;  // slice type == cache type (verified at startup)
} CacheT;

#define MAX_CACHES 256
static CacheT g_caches[MAX_CACHES];
static int g_num_caches = 0;

// Logits sampling state: raw int16 argmax when the decode logits output is
// int16 per-tensor quantized with a positive scale; the float32 scratch is
// used to dequantize a row on demand (--dump-logits) and as the fallback
// argmax path.
static int g_logits_raw_argmax = 0;
static float *g_logits_scratch = NULL;

static int cache_index(const char *name) {
    for (int i = 0; i < g_num_caches; i++) {
        if (strcmp(g_caches[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Slot element access with quantize/dequantize, mirroring the oracle's
// try_get_quantized_input / try_get_dequantized_output (ai_edge_quantizer
// uniform_quantize / uniform_dequantize), in float32 exactly like numpy:
//   quantize:    q = clip(rintf(x / scale + zp), qmin, qmax)
//                round-half-even (np.rint), narrow-range clip [qmin+1, qmax]
//                for symmetric (zp == 0) signed types (num_bits >= 8)
//   dequantize:  x = (q - zp) * scale
//                integer subtraction first, then the float32 multiply
// ---------------------------------------------------------------------------

static int elem_is_signed(LiteRtElementType e) {
    return e == kLiteRtElementTypeInt8 || e == kLiteRtElementTypeInt16 ||
           e == kLiteRtElementTypeInt32 || e == kLiteRtElementTypeInt64;
}

static void int_range(LiteRtElementType e, int64_t *lo, int64_t *hi) {
    switch (e) {
        case kLiteRtElementTypeInt8:
            *lo = -128;
            *hi = 127;
            break;
        case kLiteRtElementTypeUInt8:
            *lo = 0;
            *hi = 255;
            break;
        case kLiteRtElementTypeInt16:
            *lo = -32768;
            *hi = 32767;
            break;
        case kLiteRtElementTypeUInt16:
            *lo = 0;
            *hi = 65535;
            break;
        case kLiteRtElementTypeInt32:
            *lo = -2147483648LL;
            *hi = 2147483647LL;
            break;
        case kLiteRtElementTypeUInt32:
            *lo = 0;
            *hi = 4294967295LL;
            break;
        default:
            *lo = INT64_MIN;
            *hi = INT64_MAX;
            break;
    }
}

static float read_elem(const TType *t, const uint8_t *p) {
    switch (t->elem) {
        case kLiteRtElementTypeFloat32: {
            float f;
            memcpy(&f, p, 4);
            return f;
        }
        case kLiteRtElementTypeFloat16: {
            uint16_t h;
            memcpy(&h, p, 2);
            return half_to_float(h);
        }
        case kLiteRtElementTypeBool:
        case kLiteRtElementTypeUInt8:
            return (float)*(const uint8_t *)p;
        case kLiteRtElementTypeInt8:
            return (float)*(const int8_t *)p;
        case kLiteRtElementTypeUInt16:
            return (float)*(const uint16_t *)p;
        case kLiteRtElementTypeInt16:
            return (float)*(const int16_t *)p;
        case kLiteRtElementTypeUInt32:
            return (float)*(const uint32_t *)p;
        case kLiteRtElementTypeInt32:
            return (float)*(const int32_t *)p;
        case kLiteRtElementTypeUInt64:
            return (double)*(const uint64_t *)p;
        case kLiteRtElementTypeInt64:
            return (double)*(const int64_t *)p;
        default:
            die("unsupported element type");
    }
    return 0.0f;
}

// Read element i from a locked buffer as a plain float32, dequantizing when
// the slot is quantized (the oracle's uniform_dequantize: integer q - zp
// first, then the float32 multiply by scale).
static float slot_read(const Slot *s, const uint8_t *base, size_t i) {
    const TType *t = &s->t;
    float v = read_elem(t, base + i * elem_size(t->elem));
    if (t->qkind == Q_NONE) {
        return v;
    }
    return (float)((int32_t)(int64_t)v - (int32_t)t->zp) * t->scale;
}

// Store a raw integer value (no conversion) into an integer element.
static void store_int(const TType *t, uint8_t *p, int64_t q) {
    switch (t->elem) {
        case kLiteRtElementTypeBool:
        case kLiteRtElementTypeUInt8:
            *p = (uint8_t)q;
            break;
        case kLiteRtElementTypeInt8:
            *(int8_t *)p = (int8_t)q;
            break;
        case kLiteRtElementTypeUInt16:
            *(uint16_t *)p = (uint16_t)q;
            break;
        case kLiteRtElementTypeInt16:
            *(int16_t *)p = (int16_t)q;
            break;
        case kLiteRtElementTypeUInt32:
            *(uint32_t *)p = (uint32_t)q;
            break;
        case kLiteRtElementTypeInt32:
            *(int32_t *)p = (int32_t)q;
            break;
        case kLiteRtElementTypeUInt64:
            *(uint64_t *)p = (uint64_t)q;
            break;
        case kLiteRtElementTypeInt64:
            *(int64_t *)p = q;
            break;
        default:
            die("unsupported element type");
            break;
    }
}

// Write an integer value into element i as is: the oracle passes non-float
// inputs to the runner without conversion, so no quantization is applied
// (even to quantized slots).
static void slot_write_int(const Slot *s, uint8_t *base, size_t i, int64_t q) {
    const TType *t = &s->t;
    uint8_t *p = base + i * elem_size(t->elem);
    if (t->elem == kLiteRtElementTypeFloat32) {
        float f = (float)q;
        memcpy(p, &f, 4);
        return;
    }
    if (t->elem == kLiteRtElementTypeFloat16) {
        uint16_t h = float_to_half((float)q);
        memcpy(p, &h, 2);
        return;
    }
    store_int(t, p, q);
}

// Write a plain float32 value into element i of a locked buffer. For a
// quantized (non-float) slot this mirrors the oracle's uniform_quantize:
// q = clip(rintf(x / scale + zp), qmin, qmax), in float32, round-half-even,
// with the narrow-range clip [qmin+1, qmax] for symmetric (zp == 0) signed
// types (num_bits >= 8 always holds for the integer element types here).
static void slot_write(const Slot *s, uint8_t *base, size_t i, float v) {
    const TType *t = &s->t;
    uint8_t *p = base + i * elem_size(t->elem);
    if (t->elem == kLiteRtElementTypeFloat32) {
        memcpy(p, &v, 4);
        return;
    }
    if (t->elem == kLiteRtElementTypeFloat16) {
        uint16_t h = float_to_half(v);
        memcpy(p, &h, 2);
        return;
    }
    int64_t q;
    if (t->qkind == Q_NONE) {
        q = (int64_t)(double)v;
    } else {
        float qf = v / t->scale;
        qf = qf + (float)t->zp;  // float32 arithmetic, like numpy
        qf = rintf(qf);          // round-half-even, like np.rint
        int64_t lo, hi;
        int_range(t->elem, &lo, &hi);
        if (t->zp == 0 && elem_is_signed(t->elem)) {
            lo += 1;  // narrow-range symmetric quantization (oracle)
        }
        if (qf < (float)lo) {
            q = lo;
        } else if (qf > (float)hi) {
            q = hi;
        } else {
            q = (int64_t)qf;
        }
    }
    store_int(t, p, q);
}

static void *buf_lock(LiteRtTensorBuffer buf, LiteRtTensorBufferLockMode mode,
                      const char *what) {
    void *addr = NULL;
    LiteRtStatus st = LiteRtLockTensorBuffer(buf, &addr, mode);
    if (st != kLiteRtStatusOk || !addr) {
        fail(1, "FAILED: lock %s -> %d (%s)", what, (int)st, status_str(st));
    }
    return addr;
}

static void buf_unlock(LiteRtTensorBuffer buf) {
    CHECK_LITERT(LiteRtUnlockTensorBuffer(buf));
}

static void *slot_lock(const Slot *s, LiteRtTensorBufferLockMode mode,
                       const char *what) {
    return buf_lock(s->buf, mode, what);
}

static void slot_unlock(const Slot *s) {
    buf_unlock(s->buf);
}

// Fill the whole slot from float32 values (src must have t->n elements).
static void fill_f32(Slot *s, const float *src) {
    uint8_t *base = (uint8_t *)slot_lock(s, kLiteRtTensorBufferLockModeWrite,
                                         s->name);
    for (size_t i = 0; i < s->t.n; i++) {
        slot_write(s, base, i, src[i]);
    }
    slot_unlock(s);
}

// Fill the whole slot from float32 values starting at src[off].
static void fill_f32_off(Slot *s, const float *src, size_t off) {
    fill_f32(s, src + off);
}

// Fill the whole slot from int32 values (src must have t->n elements).
// Integer inputs pass to the runner as is (no quantization).
static void fill_i32(Slot *s, const int32_t *src) {
    uint8_t *base = (uint8_t *)slot_lock(s, kLiteRtTensorBufferLockModeWrite,
                                         s->name);
    for (size_t i = 0; i < s->t.n; i++) {
        slot_write_int(s, base, i, (int64_t)src[i]);
    }
    slot_unlock(s);
}

// Fill the first `real` elements from src, the rest with zeros.
static void fill_i32_zeros_tail(Slot *s, const int32_t *src, size_t real) {
    if (real > s->t.n) {
        real = s->t.n;
    }
    uint8_t *base = (uint8_t *)slot_lock(s, kLiteRtTensorBufferLockModeWrite,
                                         s->name);
    for (size_t i = 0; i < s->t.n; i++) {
        slot_write_int(s, base, i, i < real ? (int64_t)src[i] : 0);
    }
    slot_unlock(s);
}

// Copy an output slot into an input slot (same element count). When the two
// tensors are bit-compatible (same element type and same per-tensor
// quantization params), the raw bytes are copied as is: with matching params
// the oracle's dequantize -> quantize round trip is the identity, so this
// is bit-identical and skips the float32 round trip. When they differ,
// convert via the float32 staging: dequantize the output, quantize into the
// input with the input tensor's own params.
static void copy_out_to_in(Slot *dst, const Slot *src) {
    if (dst->t.n != src->t.n) {
        fail(2, "copy_out_to_in '%s' -> '%s': element count mismatch "
                "(%zu vs %zu)",
             dst->name, src->name, dst->t.n, src->t.n);
    }
    if (!src->f32) {
        fail(2, "copy_out_to_in '%s' -> '%s': source has no float32 "
                "staging",
             src->name, dst->name);
    }
    const uint8_t *sbase =
        (const uint8_t *)slot_lock(src, kLiteRtTensorBufferLockModeRead,
                                   src->name);
    uint8_t *dbase = (uint8_t *)slot_lock(dst, kLiteRtTensorBufferLockModeWrite,
                                          dst->name);
    if (ttype_bitwise_same(&dst->t, &src->t)) {
        memcpy(dbase, sbase, dst->t.n * elem_size(dst->t.elem));
    } else {
        for (size_t i = 0; i < dst->t.n; i++) {
            slot_write(dst, dbase, i, src->f32[i]);
        }
    }
    slot_unlock(dst);
    slot_unlock(src);
}

// Greedy argmax over float32 logits: index of the largest element, the
// last one among equal maxima (the oracle's np.argsort(...)[..., -1]).
static size_t argmax_f32(const float *v, size_t n) {
    size_t best = 0;
    float best_v = v[0];
    for (size_t i = 1; i < n; i++) {
        if (v[i] >= best_v) {
            best_v = v[i];
            best = i;
        }
    }
    return best;
}

// Greedy argmax over raw int16 logits (valid when the tensor is per-tensor
// quantized with a positive scale: the dequantized values (q - zp) * scale
// are monotonic in q, so the maximum index is the same). Ties go to the
// last index among equal maxima, exactly like argmax_f32.
static size_t argmax_i16(const int16_t *v, size_t n) {
    size_t best = 0;
    int16_t best_v = v[0];
    for (size_t i = 1; i < n; i++) {
        if (v[i] >= best_v) {
            best_v = v[i];
            best = i;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Loading.
// ---------------------------------------------------------------------------

static void exe_dir(char *buf, size_t cap) {
    ssize_t n = readlink("/proc/self/exe", buf, cap - 1);
    if (n <= 0 || (size_t)n >= cap) {
        fprintf(stderr, APP ": cannot resolve executable path: %s\n",
                strerror(errno));
        exit(2);
    }
    buf[n] = '\0';
    char *slash = strrchr(buf, '/');
    if (!slash) {
        buf[0] = '.';
        buf[1] = '\0';
        return;
    }
    if (slash == buf) {  // root dir
        buf[1] = '\0';
        return;
    }
    *slash = '\0';
}

// Find the files in dir whose names contain `substr` (at least 1, at most `max`),
// sorted by name into out[0..max-1]. Returns the count.
static int find_model_files(const char *dir, const char *substr, char out[][1024],
                            int max) {
    DIR *d = opendir(dir);
    if (!d) {
        fail(2, "cannot open model dir '%s': %s", dir, strerror(errno));
    }
    int found = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strstr(e->d_name, substr) == NULL) {
            continue;
        }
        if (found >= max) {
            fail(2, "more than %d files matching '%s' in '%s'", max, substr,
                 dir);
        }
        int w = snprintf(out[found], 1024, "%s/%s", dir, e->d_name);
        if (w < 0 || w >= 1024) {
            fail(2, "model path too long for '%s'", dir);
        }
        found++;
    }
    closedir(d);
    if (found == 0) {
        fail(2, "no file matching '%s' in '%s'", substr, dir);
    }
    if (found == 2 && strcmp(out[0], out[1]) > 0) {
        char tmp[1024];
        memcpy(tmp, out[0], sizeof tmp);
        memcpy(out[0], out[1], sizeof tmp);
        memcpy(out[1], tmp, sizeof tmp);
    }
    return found;
}

// Find the one file in dir whose name contains `substr` (and not `exclude`,
// if given). With optional=1, zero matches returns 0 instead of failing.
static int find_model_file_x(const char *dir, const char *substr,
                             const char *exclude, int optional, char *out,
                             size_t cap) {
    DIR *d = opendir(dir);
    if (!d) {
        fail(2, "cannot open model dir '%s': %s", dir, strerror(errno));
    }
    int found = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strstr(e->d_name, substr) == NULL ||
            (exclude && strstr(e->d_name, exclude) != NULL)) {
            continue;
        }
        found++;
        int w = snprintf(out, cap, "%s/%s", dir, e->d_name);
        if (w < 0 || (size_t)w >= cap) {
            fail(2, "model path too long for '%s'", dir);
        }
    }
    closedir(d);
    if (optional && found == 0) {
        return 0;
    }
    if (found != 1) {
        fail(2, "expected exactly one file matching '%s' in '%s', found %d",
             substr, dir, found);
    }
    return 1;
}

static void find_model_file(const char *dir, const char *substr, char *out,
                            size_t cap) {
    find_model_file_x(dir, substr, NULL, 0, out, cap);
}

// Derive the signature-level tensor name from the raw tensor name. The
// exporter names tensors "<sigkey>_<name>" for inputs and
// "<sigkey>_<name>_output" for outputs, while the oracle (and the litertlm
// state-buffer metadata) addresses tensors by their signature names
// ("embeddings", "kv_cache_k_0", "logits", ...). Fall back to the raw name
// when the pattern is absent.
static void derive_sig_name(const char *key, bool is_input, const char *raw,
                            char *out, size_t cap) {
    size_t klen = strlen(key);
    const char *p = raw;
    if (strncmp(raw, key, klen) == 0 && raw[klen] == '_') {
        p = raw + klen + 1;
    }
    if (!is_input) {
        static const char suf[] = "_output";
        size_t slen = sizeof suf - 1;
        size_t len = strlen(p);
        if (len > slen && strcmp(p + len - slen, suf) == 0) {
            len -= slen;
        }
        if (len >= cap) {
            len = cap - 1;
        }
        memcpy(out, p, len);
        out[len] = '\0';
        return;
    }
    snprintf(out, cap, "%s", p);
}

static void collect_slot(LiteRtSignature sig, const char *key, bool is_input,
                         LiteRtParamIndex i, Slot **arr, size_t *count) {
    Slot *tmp = (Slot *)realloc(*arr, (*count + 1) * sizeof(Slot));
    if (!tmp) {
        die("out of memory");
    }
    *arr = tmp;
    Slot *s = &(*arr)[(*count)++];
    memset(s, 0, sizeof *s);
    LiteRtTensor tensor;
    if (is_input) {
        CHECK_LITERT(LiteRtGetSignatureInputTensorByIndex(sig, i, &tensor));
    } else {
        CHECK_LITERT(LiteRtGetSignatureOutputTensorByIndex(sig, i, &tensor));
    }
    // Prefer the signature's own I/O name (Google's exports name tensors e.g.
    // "StatefulPartitionedCall:30" while the signature calls it "logits");
    // fall back to the tensor-name rule when the signature has no name.
    const char *sname = NULL;
    LiteRtStatus nst = is_input ? LiteRtGetSignatureInputName(sig, i, &sname)
                                : LiteRtGetSignatureOutputName(sig, i, &sname);
    if (nst == kLiteRtStatusOk && sname != NULL && strlen(sname) > 0 &&
        strlen(sname) < sizeof(s->name)) {
        strcpy(s->name, sname);
    } else {
        const char *name = NULL;
        CHECK_LITERT(LiteRtGetTensorName(tensor, &name));
        if (name == NULL || strlen(name) == 0) {
            die("tensor name missing");
        }
        derive_sig_name(key, is_input, name, s->name, sizeof(s->name));
    }
    if (strlen(s->name) == 0 || strlen(s->name) >= sizeof(s->name)) {
        die("signature tensor name empty or too long");
    }
    ttype_from_tensor(tensor, &s->t);
}

static void load_section(int idx, const char *path, const char *role,
                         LiteRtOptions options) {
    Section *sec = &g_sections[idx];
    sec->role = role;
    sec->loaded = 1;
    snprintf(sec->path, sizeof sec->path, "%s", path);
    fprintf(stderr, APP ": loading %s model: %s\n", role, path);
    CHECK_LITERT(LiteRtCreateModelFromFile(g_env, path, &sec->model));
    LiteRtParamIndex n = 0;
    CHECK_LITERT(LiteRtGetNumModelSignatures(sec->model, &n));
    sec->sigs = (Sig *)calloc((size_t)(n > 0 ? n : 1), sizeof(Sig));
    if (!sec->sigs) {
        die("out of memory");
    }
    sec->num_sigs = 0;
    for (LiteRtParamIndex i = 0; i < n; i++) {
        Sig *sig = &sec->sigs[sec->num_sigs++];
        memset(sig, 0, sizeof *sig);
        sig->section_idx = idx;
        sig->idx = i;
        LiteRtSignature rsig;
        CHECK_LITERT(LiteRtGetModelSignature(sec->model, i, &rsig));
        const char *key = NULL;
        CHECK_LITERT(LiteRtGetSignatureKey(rsig, &key));
        if (key == NULL || strlen(key) == 0 ||
            strlen(key) >= sizeof(sig->key)) {
            die("signature key missing or too long");
        }
        strcpy(sig->key, key);
        LiteRtParamIndex num_in = 0, num_out = 0;
        CHECK_LITERT(LiteRtGetNumSignatureInputs(rsig, &num_in));
        CHECK_LITERT(LiteRtGetNumSignatureOutputs(rsig, &num_out));
        for (LiteRtParamIndex j = 0; j < num_in; j++) {
            collect_slot(rsig, sig->key, true, j, &sig->in, &sig->num_in);
        }
        for (LiteRtParamIndex j = 0; j < num_out; j++) {
            collect_slot(rsig, sig->key, false, j, &sig->out, &sig->num_out);
        }
    }
    CHECK_LITERT(
        LiteRtCreateCompiledModel(g_env, sec->model, options, &sec->compiled));
}

// ---------------------------------------------------------------------------
// Layer-chunk chains (2026-11-11): a signature too large for one compiled
// network (~1 GiB on MT6993) is cut by npu/llm17/survey/chunk/split_layers.py
// into files <...prefill_decode...><sig>.c<k>.<...>tflite. Each chunk is its
// own model + compiled model. A composite Sig (in g_sections[SEC_PD]) exposes
// the union of the chunks' graph inputs/outputs, so the rest of the driver is
// unchanged; tensors crossing a cut are named x<N> and live in one shared
// buffer each. Every chunk's buffer arrays point at the composite slots or the
// cross buffers, so running the chain needs no copies.
// ---------------------------------------------------------------------------
#define MAX_CHUNKS 16
#define MAX_CHAINS 8
typedef struct {
    LiteRtModel model;
    LiteRtCompiledModel compiled;
    Sig sig;  // the chunk's own signature (slots carry names/types only)
} Chunk;
typedef struct {
    char key[128];
    int n;
    Chunk ch[MAX_CHUNKS];
} Chain;
static Chain g_chains[MAX_CHAINS];
static int earlier_chunk_outputs(const Chain *c, int k, const char *name);
static int g_num_chains = 0;
// ASR_CHAIN_SEQUENTIAL=1: a chain's compiled networks exist only while that chain is in use; switching chains
// (prefill -> decode) first destroys the other chain's networks. Peak NPU memory = the largest chain, not the sum
// (Gemma 4 E4B: prefill 1.32 GB + decode 2.68 GB of networks is too close to the 2026-11-11 G1 hang).
static int g_chain_seq = 0;
static LiteRtOptions g_chain_opts = NULL;
static int g_active_chain = -1;
static double g_chain_create_ms = 0.0;
typedef struct {
    char name[128];
    LiteRtTensorBuffer buf;
} CrossBuf;
static CrossBuf *g_cross = NULL;
static size_t g_num_cross = 0;

static int is_cross_name(const char *n) {
    if (n[0] != 'x' || n[1] == '\0') {
        return 0;
    }
    for (const char *c = n + 1; *c; c++) {
        if (*c < '0' || *c > '9') {
            return 0;
        }
    }
    return 1;
}

// Chunk index from a file name: the digits after ".c" followed by '.'
// (e.g. "decode.c2.compiled.tflite" -> 2), else -1.
static int chunk_index_of(const char *fname) {
    const char *p = fname;
    while ((p = strstr(p, ".c")) != NULL) {
        p += 2;
        if (*p >= '0' && *p <= '9') {
            int v = 0;
            while (*p >= '0' && *p <= '9') {
                v = v * 10 + (*p - '0');
                p++;
            }
            if (*p == '.') {
                return v;
            }
        }
    }
    return -1;
}

static void load_chunk(const char *path, int cidx, LiteRtOptions options) {
    LiteRtModel model;
    fprintf(stderr, APP ": loading chunk %d: %s\n", cidx, path);
    CHECK_LITERT(LiteRtCreateModelFromFile(g_env, path, &model));
    LiteRtParamIndex n = 0;
    CHECK_LITERT(LiteRtGetNumModelSignatures(model, &n));
    if (n != 1) {
        fail(2, "chunk '%s' must have exactly one signature (has %d)", path, (int)n);
    }
    LiteRtSignature rsig;
    CHECK_LITERT(LiteRtGetModelSignature(model, 0, &rsig));
    const char *key = NULL;
    CHECK_LITERT(LiteRtGetSignatureKey(rsig, &key));
    if (!key || strlen(key) == 0 || strlen(key) >= 128) {
        die("chunk signature key missing or too long");
    }
    Chain *c = NULL;
    for (int i = 0; i < g_num_chains; i++) {
        if (strcmp(g_chains[i].key, key) == 0) {
            c = &g_chains[i];
        }
    }
    if (!c) {
        if (g_num_chains >= MAX_CHAINS) {
            die("too many chunked signatures");
        }
        c = &g_chains[g_num_chains++];
        memset(c, 0, sizeof *c);
        strcpy(c->key, key);
    }
    if (cidx < 0 || cidx >= MAX_CHUNKS || c->ch[cidx].model) {
        fail(2, "bad or duplicate chunk index %d for '%s'", cidx, key);
    }
    Chunk *ch = &c->ch[cidx];
    if (cidx + 1 > c->n) {
        c->n = cidx + 1;
    }
    ch->model = model;
    memset(&ch->sig, 0, sizeof ch->sig);
    strcpy(ch->sig.key, key);
    LiteRtParamIndex num_in = 0, num_out = 0;
    CHECK_LITERT(LiteRtGetNumSignatureInputs(rsig, &num_in));
    CHECK_LITERT(LiteRtGetNumSignatureOutputs(rsig, &num_out));
    for (LiteRtParamIndex j = 0; j < num_in; j++) {
        collect_slot(rsig, key, true, j, &ch->sig.in, &ch->sig.num_in);
    }
    for (LiteRtParamIndex j = 0; j < num_out; j++) {
        collect_slot(rsig, key, false, j, &ch->sig.out, &ch->sig.num_out);
    }
    if (g_chain_seq) {
        g_chain_opts = options;   // compiled on first use (see activate_chain)
    } else {
        CHECK_LITERT(LiteRtCreateCompiledModel(g_env, model, options, &ch->compiled));
    }
}

// Sequential mode: make chain ci the only one with compiled networks.
static void activate_chain(int ci) {
    if (!g_chain_seq || g_active_chain == ci) {
        return;
    }
    double t0 = now_ms();
    for (int i = 0; i < g_num_chains; i++) {
        if (i == ci) {
            continue;
        }
        for (int k = 0; k < g_chains[i].n; k++) {
            if (g_chains[i].ch[k].compiled) {
                LiteRtDestroyCompiledModel(g_chains[i].ch[k].compiled);
                g_chains[i].ch[k].compiled = NULL;
            }
        }
    }
    Chain *c = &g_chains[ci];
    for (int k = 0; k < c->n; k++) {
        if (!c->ch[k].compiled) {
            CHECK_LITERT(LiteRtCreateCompiledModel(g_env, c->ch[k].model, g_chain_opts, &c->ch[k].compiled));
        }
    }
    g_active_chain = ci;
    double dt = now_ms() - t0;
    g_chain_create_ms += dt;
    fprintf(stderr, APP ": activated chain '%s' (%d chunks) in %.0f ms\n", c->key, c->n, dt);
}

// Append a copy of slot `src` (name + type) to arr unless the name is there
// already (then the types must be bit-compatible).
static void union_slot(Slot **arr, size_t *count, const Slot *src) {
    for (size_t i = 0; i < *count; i++) {
        if (strcmp((*arr)[i].name, src->name) == 0) {
            if (!ttype_bitwise_same(&(*arr)[i].t, &src->t)) {
                fail(2, "chunk tensor '%s' has different types in two chunks", src->name);
            }
            return;
        }
    }
    Slot *tmp = (Slot *)realloc(*arr, (*count + 1) * sizeof(Slot));
    if (!tmp) {
        die("out of memory");
    }
    *arr = tmp;
    Slot *d = &(*arr)[(*count)++];
    memset(d, 0, sizeof *d);
    strcpy(d->name, src->name);
    d->t = src->t;  // shallow copy of the type: the chunk slots are never freed
}

// Build the composite Sigs into g_sections[SEC_PD] (a synthetic section).
static void build_chain_section(void) {
    Section *sec = &g_sections[SEC_PD];
    memset(sec, 0, sizeof *sec);
    sec->role = "prefill_decode(chunks)";
    sec->loaded = 1;
    sec->sigs = (Sig *)calloc((size_t)g_num_chains, sizeof(Sig));
    if (!sec->sigs) {
        die("out of memory");
    }
    for (int i = 0; i < g_num_chains; i++) {
        Chain *c = &g_chains[i];
        for (int k = 0; k < c->n; k++) {
            if (!c->ch[k].model) {
                fail(2, "chunk %d of '%s' is missing", k, c->key);
            }
        }
        Sig *sig = &sec->sigs[sec->num_sigs++];
        strcpy(sig->key, c->key);
        sig->section_idx = SEC_PD;
        sig->chain = i + 1;
        for (int k = 0; k < c->n; k++) {
            Sig *cs = &c->ch[k].sig;
            for (size_t j = 0; j < cs->num_in; j++) {
                if (!is_cross_name(cs->in[j].name) &&
                    !earlier_chunk_outputs(c, k, cs->in[j].name)) {
                    union_slot(&sig->in, &sig->num_in, &cs->in[j]);
                }
            }
            for (size_t j = 0; j < cs->num_out; j++) {
                if (!is_cross_name(cs->out[j].name)) {
                    union_slot(&sig->out, &sig->num_out, &cs->out[j]);
                }
            }
        }
        fprintf(stderr, APP ": chain '%s': %d chunks, %zu inputs, %zu outputs\n",
                sig->key, c->n, sig->num_in, sig->num_out);
    }
}

// 1 if an earlier chunk (index < k) of chain c has an output named `name`
// (a graph output another chunk also reads, e.g. a kv_slice that a Gemma
// KV-sharing layer consumes).
static int earlier_chunk_outputs(const Chain *c, int k, const char *name) {
    for (int e = 0; e < k; e++) {
        const Sig *cs = &c->ch[e].sig;
        for (size_t j = 0; j < cs->num_out; j++) {
            if (strcmp(cs->out[j].name, name) == 0) {
                return 1;
            }
        }
    }
    return 0;
}

static LiteRtTensorBuffer cross_buf(const Slot *s) {
    for (size_t i = 0; i < g_num_cross; i++) {
        if (strcmp(g_cross[i].name, s->name) == 0) {
            return g_cross[i].buf;
        }
    }
    CrossBuf *tmp = (CrossBuf *)realloc(g_cross, (g_num_cross + 1) * sizeof(CrossBuf));
    if (!tmp) {
        die("out of memory");
    }
    g_cross = tmp;
    CrossBuf *cb = &g_cross[g_num_cross++];
    strcpy(cb->name, s->name);
    LiteRtRankedTensorType rt;
    rt.element_type = s->t.elem;
    rt.layout = s->t.layout;
    CHECK_LITERT(LiteRtCreateManagedTensorBuffer(
        g_env, kLiteRtTensorBufferTypeHostMemory, &rt, s->t.n * elem_size(s->t.elem), &cb->buf));
    return cb->buf;
}

// After allocate_buffers: point every chunk's buffer arrays at the composite
// slot buffers (graph tensors) or the shared cross buffers.
static void bind_chains(void) {
    Section *sec = &g_sections[SEC_PD];
    for (size_t si = 0; si < sec->num_sigs; si++) {
        Sig *sig = &sec->sigs[si];
        if (!sig->chain) {
            continue;
        }
        Chain *c = &g_chains[sig->chain - 1];
        for (int k = 0; k < c->n; k++) {
            Sig *cs = &c->ch[k].sig;
            free(cs->in_bufs);
            free(cs->out_bufs);
            cs->in_bufs = (LiteRtTensorBuffer *)calloc(cs->num_in + 1, sizeof(LiteRtTensorBuffer));
            cs->out_bufs = (LiteRtTensorBuffer *)calloc(cs->num_out + 1, sizeof(LiteRtTensorBuffer));
            if (!cs->in_bufs || !cs->out_bufs) {
                die("out of memory");
            }
            for (size_t j = 0; j < cs->num_in; j++) {
                const Slot *s = &cs->in[j];
                if (is_cross_name(s->name)) {
                    cs->in_bufs[j] = cross_buf(s);
                } else if (earlier_chunk_outputs(c, k, s->name)) {
                    cs->in_bufs[j] = sig_out_by_name(sig, s->name)->buf;  // read an earlier chunk's graph output
                } else {
                    cs->in_bufs[j] = sig_in_by_name(sig, s->name)->buf;
                }
            }
            for (size_t j = 0; j < cs->num_out; j++) {
                const Slot *s = &cs->out[j];
                cs->out_bufs[j] = is_cross_name(s->name) ? cross_buf(s) : sig_out_by_name(sig, s->name)->buf;
            }
        }
    }
}

static void run_chain(Sig *sig) {
    activate_chain(sig->chain - 1);
    Chain *c = &g_chains[sig->chain - 1];
    for (int k = 0; k < c->n; k++) {
        Sig *cs = &c->ch[k].sig;
        LiteRtStatus st = LiteRtRunCompiledModel(c->ch[k].compiled, 0, cs->num_in, cs->in_bufs,
                                                 cs->num_out, cs->out_bufs);
        if (st != kLiteRtStatusOk) {
            fprintf(stderr, APP ": FAILED: run chunk %d of '%s' -> %d (%s)\n", k, sig->key, (int)st,
                    status_str(st));
            exit(1);
        }
    }
}

static LiteRtStatus make_managed(const TType *t, const char *what,
                                 LiteRtTensorBuffer *out) {
    LiteRtRankedTensorType rt;
    rt.element_type = t->elem;
    rt.layout = t->layout;
    LiteRtStatus st = LiteRtCreateManagedTensorBuffer(
        g_env, kLiteRtTensorBufferTypeHostMemory, &rt,
        t->n * elem_size(t->elem), out);
    if (st != kLiteRtStatusOk) {
        fail(1, "FAILED: create managed buffer for %s -> %d (%s)", what,
             (int)st, status_str(st));
    }
    return st;
}

// Create every tensor buffer once (no per-step create/destroy):
//   * the persistent KV cache buffers: one per cache tensor, in the
//     model's native dtype and quantization, initialized the way the
//     oracle's init_cache produces (the quantized value of 0.0); the
//     prefill/decode cache inputs that are bit-compatible with it are
//     bound to it via LiteRtDuplicateTensorBuffer (shared storage),
//   * every other slot's own managed buffer in the tensor's native format,
//   * the float32 output staging: every output slot keeps its dequantized
//     float32 values except the runners' kv_slice outputs (merged into the
//     persistent buffers on the host from their raw bytes) and the decode
//     logits (raw argmax; dequantized to g_logits_scratch on demand).
static void allocate_buffers(void) {
    for (int i = 0; i < g_num_caches; i++) {
        CacheT *c = &g_caches[i];
        LiteRtRankedTensorType rt;
        rt.element_type = c->t.elem;
        rt.layout = c->t.layout;
        CHECK_LITERT(LiteRtCreateManagedTensorBuffer(
            g_env, kLiteRtTensorBufferTypeHostMemory, &rt,
            c->t.n * elem_size(c->t.elem), &c->buf));
        uint8_t *base = (uint8_t *)buf_lock(c->buf,
                                            kLiteRtTensorBufferLockModeWrite,
                                            c->name);
        Slot tmp;
        memset(&tmp, 0, sizeof tmp);
        tmp.t = c->t;
        for (size_t j = 0; j < c->t.n; j++) {
            slot_write(&tmp, base, j, 0.0f);
        }
        buf_unlock(c->buf);
    }
    // Logits scratch: one vocab-sized float32 row, dequantized on demand.
    {
        const Sig *decode = find_sig(SEC_PD, "decode");
        const Slot *logits = sig_out_by_name(decode, "logits");
        g_logits_scratch = (float *)malloc(logits->t.n * sizeof(float));
        if (!g_logits_scratch) {
            die("out of memory");
        }
    }
    for (int si = 0; si < NUM_SEC_SLOTS; si++) {
        Section *sec = &g_sections[si];
        if (!sec->loaded) {
            continue;
        }
        for (size_t k = 0; k < sec->num_sigs; k++) {
            Sig *sig = &sec->sigs[k];
            sig->in_bufs = (LiteRtTensorBuffer *)calloc(
                (size_t)(sig->num_in > 0 ? sig->num_in : 1),
                sizeof(LiteRtTensorBuffer));
            sig->out_bufs = (LiteRtTensorBuffer *)calloc(
                (size_t)(sig->num_out > 0 ? sig->num_out : 1),
                sizeof(LiteRtTensorBuffer));
            if (!sig->in_bufs || !sig->out_bufs) {
                die("out of memory");
            }
            for (size_t i = 0; i < sig->num_in; i++) {
                Slot *s = &sig->in[i];
                int ci = cache_index(s->name);
                if (ci >= 0 && (si == SEC_PD || si == SEC_PD2) &&
                    ttype_bitwise_same(&g_caches[ci].t, &s->t)) {
                    // Bind this prefill/decode cache input to the persistent
                    // KV cache buffer: same storage, shared (the duplicate
                    // raises the reference count so each later destroy only
                    // drops one reference).
                    CHECK_LITERT(LiteRtDuplicateTensorBuffer(g_caches[ci].buf));
                    s->buf = g_caches[ci].buf;
                    s->cache_direct = 1;
                } else {
                    make_managed(&s->t, s->name, &s->buf);
                }
                s->owns_buf = 1;
                sig->in_bufs[i] = s->buf;
            }
            for (size_t i = 0; i < sig->num_out; i++) {
                Slot *s = &sig->out[i];
                make_managed(&s->t, s->name, &s->buf);
                s->owns_buf = 1;
                if (strncmp(s->name, "kv_slice", 8) != 0 &&
                    strcmp(s->name, "logits") != 0) {
                    s->f32 = (float *)malloc(s->t.n * sizeof(float));
                    if (!s->f32) {
                        die("out of memory");
                    }
                }
                sig->out_bufs[i] = s->buf;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Discovery: buckets, cache set, cache length, hidden size.
// ---------------------------------------------------------------------------

static long g_buckets[64];
static int g_num_buckets = 0;
static long g_cache_length = 0;
static long g_hidden = 0;

static int parse_bucket(const char *key, long *out) {
    static const char prefix[] = "prefill_";
    size_t plen = sizeof prefix - 1;
    if (strncmp(key, prefix, plen) != 0) {
        return 0;
    }
    const char *p = key + plen;
    if (*p == '\0') {
        return 0;
    }
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (*end != '\0' || v <= 0) {
        return 0;
    }
    *out = v;
    return 1;
}

static void require_sig(int section_idx, const char *key) {
    if (!find_sig(section_idx, key)) {
        fprintf(stderr, APP ": model '%s' is missing the '%s' signature\n",
                g_sections[section_idx].role, key);
        exit(2);
    }
}

// The cache_update outputs must cover the whole cache set (the oracle
// replaces the entire cache dict with the runner outputs).
static void require_cache_outputs(Sig *cu) {
    for (int i = 0; i < g_num_caches; i++) {
        if (!sig_out_by_name(cu, g_caches[i].name)) {
            fprintf(stderr, APP ": '%s' does not output cache tensor '%s'\n",
                    cu->key, g_caches[i].name);
            exit(2);
        }
    }
}

static void discover(void) {
    Sig *decode = find_sig(SEC_PD, "decode");
    if (!decode) {
        die("prefill_decode model has no 'decode' signature");
    }
    // Cache set from the decode signature's inputs containing "cache".
    g_num_caches = 0;
    for (size_t i = 0; i < decode->num_in; i++) {
        Slot *s = &decode->in[i];
        if (strstr(s->name, "cache") != NULL) {
            if (g_num_caches >= MAX_CACHES) {
                die("too many cache tensors");
            }
            CacheT *c = &g_caches[g_num_caches++];
            memset(c, 0, sizeof *c);
            strcpy(c->name, s->name);
            ttype_copy(&c->t, &s->t);
        }
    }
    if (g_num_caches == 0) {
        die("decode signature has no kv_cache inputs");
    }
    // Per-channel quantized inputs/outputs are not supported: reject them
    // up front rather than guess how to convert.
    for (int si = 0; si < NUM_SEC_SLOTS; si++) {
        Section *sec = &g_sections[si];
        if (!sec->loaded) {
            continue;
        }
        for (size_t k = 0; k < sec->num_sigs; k++) {
            Sig *sig = &sec->sigs[k];
            for (size_t i = 0; i < sig->num_in; i++) {
                Slot *s = &sig->in[i];
                if (s->t.qkind == Q_PER_CHANNEL) {
                    fail(2, "signature '%s' input '%s' has per-channel "
                            "quantization; only per-tensor quantization is "
                            "supported",
                         sig->key, s->name);
                }
            }
            for (size_t i = 0; i < sig->num_out; i++) {
                Slot *s = &sig->out[i];
                if (s->t.qkind == Q_PER_CHANNEL) {
                    fail(2, "signature '%s' output '%s' has per-channel "
                            "quantization; only per-tensor quantization is "
                            "supported",
                         sig->key, s->name);
                }
            }
        }
    }
    // Verify every cache tensor appears with a consistent shape wherever it
    // does (the element type may differ between signatures).
    for (int si = 0; si < NUM_SEC_SLOTS; si++) {
        Section *sec = &g_sections[si];
        if (!sec->loaded) {
            continue;
        }
        for (size_t k = 0; k < sec->num_sigs; k++) {
            Sig *sig = &sec->sigs[k];
            for (size_t i = 0; i < sig->num_in; i++) {
                int ci = cache_index(sig->in[i].name);
                if (ci >= 0) {
                    ttype_check_shape(&g_caches[ci].t, &sig->in[i].t,
                                     g_caches[ci].name, sig->in[i].name);
                }
            }
            for (size_t i = 0; i < sig->num_out; i++) {
                int ci = cache_index(sig->out[i].name);
                if (ci >= 0) {
                    ttype_check_shape(&g_caches[ci].t, &sig->out[i].t,
                                     g_caches[ci].name, sig->out[i].name);
                }
            }
        }
    }
    // Cache length: the shared sequence dimension, the largest dimension of
    // every cache tensor (the KV sequence capacity).
    long cache_len = 0;
    for (int i = 0; i < g_num_caches; i++) {
        long max_dim = 0;
        for (unsigned d = 0; d < g_caches[i].t.layout.rank; d++) {
            long dim = (long)g_caches[i].t.layout.dimensions[d];
            if (dim > max_dim) {
                max_dim = dim;
            }
        }
        if (cache_len == 0) {
            cache_len = max_dim;
        } else if (cache_len != max_dim) {
            fail(2, "cache tensor '%s' has no dimension matching the cache "
                    "length %ld",
                 g_caches[i].name, cache_len);
        }
    }
    if (cache_len <= 0) {
        die("could not determine the cache length from the kv_cache shapes");
    }
    g_cache_length = cache_len;
    // Time dimension per cache tensor: the dimension sized to the cache
    // length (e.g. dim 2 of the [1,1,CACHE,128] K cache, dim 3 of the
    // [1,1,128,CACHE] V cache), derived from the shape. The slice time
    // dimension is resolved per runner output the same way (against the
    // chunk size S), in host_cache_update.
    for (int i = 0; i < g_num_caches; i++) {
        int td = -1, nmatch = 0;
        for (unsigned d = 0; d < g_caches[i].t.layout.rank; d++) {
            if ((long)g_caches[i].t.layout.dimensions[d] == cache_len) {
                if (td < 0) {
                    td = (int)d;
                }
                nmatch++;
            }
        }
        if (nmatch > 1) {
            // Ambiguous (e.g. a [1,1,512,512] V cache when the head size
            // equals the cache length): the time dimension is the one where
            // the decode signature's one-token kv_slice output has size 1.
            char sname[256];
            const char *cp = strstr(g_caches[i].name, "cache");
            if (cp) {
                snprintf(sname, sizeof sname, "%.*sslice%s",
                         (int)(cp - g_caches[i].name), g_caches[i].name,
                         cp + 5);
                const Slot *sl = sig_out_by_name(find_sig(SEC_PD, "decode"),
                                                 sname);
                if (sl && sl->t.layout.rank == g_caches[i].t.layout.rank) {
                    for (unsigned d = 0; d < sl->t.layout.rank; d++) {
                        if (sl->t.layout.dimensions[d] == 1 &&
                            (long)g_caches[i].t.layout.dimensions[d] ==
                                cache_len) {
                            td = (int)d;
                            break;
                        }
                    }
                }
            }
        }
        if (td < 0) {
            fail(2, "cache tensor '%s' has no dimension matching the cache "
                    "length %ld",
                 g_caches[i].name, cache_len);
        }
        g_caches[i].time_dim = td;
        // Slice name: the runner's fresh-KV output for this cache tensor
        // (kv_cache_* -> kv_slice_*).
        const char *cp = strstr(g_caches[i].name, "cache");
        if (cp) {
            size_t pre = (size_t)(cp - g_caches[i].name);
            memcpy(g_caches[i].slice_name, g_caches[i].name, pre);
            snprintf(g_caches[i].slice_name + pre,
                     sizeof g_caches[i].slice_name - pre, "slice%s", cp + 5);
        } else {
            g_caches[i].slice_name[0] = '\0';
        }
    }
    // Verify the runners' slice outputs against the cache layouts and decide
    // per tensor whether the slice can be merged with a raw element copy:
    // only bit-identical when the slice and the cache have the same dtype
    // and the same scale/zero_point; otherwise the elements are requantized
    // (dequant with the slice params, quant with the cache params).
    for (int i = 0; i < g_num_caches; i++) {
        g_caches[i].raw_copy = 1;
    }
    for (int si = SEC_PD; si < g_num_sections; si++) {
        if (si != SEC_PD && si != SEC_PD2) {
            continue;
        }
        Section *sec = &g_sections[si];
        for (size_t k = 0; k < sec->num_sigs; k++) {
            Sig *sig = &sec->sigs[k];
            long s_size = 0;  // chunk size S for this runner (0: not a runner)
            if (strcmp(sig->key, "decode") == 0) {
                s_size = 1;
            } else {
                (void)parse_bucket(sig->key, &s_size);
            }
            if (s_size <= 0) {
                continue;
            }
            for (int i = 0; i < g_num_caches; i++) {
                CacheT *c = &g_caches[i];
                const Slot *sl = sig_out_by_name(sig, c->slice_name);
                if (!sl) {
                    continue;  // this runner does not produce the slice
                }
                const TType *ct = &c->t;
                const TType *st = &sl->t;
                if (st->layout.rank != ct->layout.rank) {
                    fail(2, "'%s' slice '%s' has rank %u, cache '%s' has "
                            "rank %u",
                         sig->key, sl->name, st->layout.rank, c->name,
                         ct->layout.rank);
                }
                // Slice sequence dimension: the cache's time dimension when
                // it carries the chunk size there, else the first dimension
                // equal to the chunk size.
                int s_td = -1;
                for (unsigned d = 0; d < st->layout.rank; d++) {
                    if ((long)st->layout.dimensions[d] == s_size &&
                        (int)d == c->time_dim) {
                        s_td = (int)d;
                        break;
                    }
                }
                if (s_td < 0) {
                    for (unsigned d = 0; d < st->layout.rank; d++) {
                        if ((long)st->layout.dimensions[d] == s_size) {
                            s_td = (int)d;
                            break;
                        }
                    }
                }
                if (s_td < 0) {
                    fail(2, "'%s' slice '%s' has no dimension matching the "
                            "chunk size %ld",
                         sig->key, sl->name, s_size);
                }
                long outer_c = 1, inner_c = 1, outer_s = 1, inner_s = 1;
                for (unsigned d = 0; d < ct->layout.rank; d++) {
                    long dim = (long)ct->layout.dimensions[d];
                    if ((int)d < c->time_dim) {
                        outer_c *= dim;
                    } else if ((int)d > c->time_dim) {
                        inner_c *= dim;
                    }
                }
                for (unsigned d = 0; d < st->layout.rank; d++) {
                    long dim = (long)st->layout.dimensions[d];
                    if ((int)d < s_td) {
                        outer_s *= dim;
                    } else if ((int)d > s_td) {
                        inner_s *= dim;
                    }
                }
                if (outer_c != outer_s || inner_c != inner_s) {
                    fail(2, "'%s' slice '%s' does not match the '%s' cache "
                            "layout (outer/inner %ld/%ld vs %ld/%ld)",
                         sig->key, sl->name, c->name, outer_c, inner_c,
                         outer_s, inner_s);
                }
                if (!ttype_bitwise_same(ct, st)) {
                    c->raw_copy = 0;
                }
            }
        }
    }
    // Logits sampling: raw int16 argmax when the decode logits output is
    // int16 per-tensor quantized with a positive scale (monotonic in the
    // dequantized value, same index).
    const Slot *logits = sig_out_by_name(decode, "logits");
    if (!logits) {
        die("decode signature has no 'logits' output");
    }
    g_logits_raw_argmax = (logits->t.elem == kLiteRtElementTypeInt16 &&
                           logits->t.qkind == Q_PER_TENSOR &&
                           logits->t.scale > 0.0f);

    // Hidden size from the decode signature's `embeddings` input.
    Slot *emb = sig_in_by_name(decode, "embeddings");
    if (!emb) {
        die("decode signature has no 'embeddings' input");
    }
    if (emb->t.layout.rank == 0) {
        die("decode 'embeddings' input has no dimensions");
    }
    g_hidden = (long)emb->t.layout.dimensions[emb->t.layout.rank - 1];
    if (g_hidden <= 0) {
        die("decode 'embeddings' input has a non-positive hidden size");
    }

    // Prefill buckets from the prefill_decode signature names.
    g_num_buckets = 0;
    for (int si = SEC_PD; si < g_num_sections; si = (si == SEC_PD ? SEC_PD2 : g_num_sections)) {
        Section *pd = &g_sections[si];
        for (size_t k = 0; k < pd->num_sigs; k++) {
            long b;
            if (parse_bucket(pd->sigs[k].key, &b)) {
                if (g_num_buckets >= (int)(sizeof g_buckets / sizeof g_buckets[0])) {
                    die("too many prefill buckets");
                }
                g_buckets[g_num_buckets++] = b;
            }
        }
    }
    if (g_num_buckets == 0) {
        die("prefill_decode model has no prefill_<N> signatures");
    }
    for (int i = 1; i < g_num_buckets; i++) {
        for (int j = i; j > 0 && g_buckets[j] < g_buckets[j - 1]; j--) {
            long tmp = g_buckets[j];
            g_buckets[j] = g_buckets[j - 1];
            g_buckets[j - 1] = tmp;
        }
    }
    // Required signatures for decode and per bucket.
    require_sig(SEC_AU, "decode_mask");
    require_sig(SEC_AU, "decode_rope");
    require_sig(SEC_AU, "decode_cache_update");
    require_sig(SEC_ED, "decode_embedder");
    char key[128];
    for (int i = 0; i < g_num_buckets; i++) {
        snprintf(key, sizeof key, "prefill_mask_%ld", g_buckets[i]);
        require_sig(SEC_AU, key);
        snprintf(key, sizeof key, "prefill_rope_%ld", g_buckets[i]);
        require_sig(SEC_AU, key);
        snprintf(key, sizeof key, "prefill_cache_update_%ld", g_buckets[i]);
        require_sig(SEC_AU, key);
        require_cache_outputs(find_sig(SEC_AU, key));
    }
    require_cache_outputs(find_sig(SEC_AU, "decode_cache_update"));
}

// ---------------------------------------------------------------------------
// Inputs.
// ---------------------------------------------------------------------------

static int32_t *g_ids = NULL;  // T prompt token ids
static long g_T = 0;
static float *g_padded_embeds = NULL;  // padded_total * H floats
static int g_have_ple = 0;          // model dir has a per_layer_embedder section
static long g_ple_n = 0;            // per-layer-embedding elements per token (layers x dim)
static float *g_padded_ple = NULL;  // padded_total * g_ple_n floats (Gemma per-layer embeddings)
static Sig *ple_sig(void);
static void ple_token(Sig *ple, int32_t token);
static void fill_ple_rows(Sig *sig, long pos, long rows);
static long g_padded_total = 0;
static float *g_features = NULL;  // g_feat_rows x H audio feature rows
static long g_feat_rows = 0;
static int32_t g_audio_pad_id = 151676;
static int g_server = 0;        // --server: serve requests from stdin
static long g_max_new = 64;     // --max-new (default for a request's max_new)
static double g_setup_ms = 0.0;  // one-time setup (load+compile), for the
                                 // per-request timing line and READY

static void *read_file(const char *path, size_t expected, const char *what) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fail(2, "cannot open '%s' (%s): %s", path, what, strerror(errno));
    }
    struct stat st;
    if (fstat(fileno(f), &st) != 0) {
        fail(2, "fstat('%s'): %s", path, strerror(errno));
    }
    if ((size_t)st.st_size != expected) {
        fail(2, "'%s' (%s) is %lld bytes; expected exactly %zu", path, what,
             (long long)st.st_size, expected);
    }
    void *buf = malloc(expected ? expected : 1);
    if (!buf) {
        die("out of memory");
    }
    if (expected > 0 && fread(buf, 1, expected, f) != expected) {
        fail(2, "short read of '%s'", path);
    }
    fclose(f);
    return buf;
}

// Simulate the prefill chunk loop to get the total padded length (same rule
// as the oracle's prefill_embeds / ref_decode.py).
static long padded_total_for(long T) {
    long ts = 0, total = 0;
    while (ts < T) {
        long rem = T - ts;
        long size = 0;
        for (int b = 0; b < g_num_buckets; b++) {
            if (g_buckets[b] >= rem) {
                size = g_buckets[b];
                break;
            }
        }
        if (size == 0) {
            size = g_buckets[g_num_buckets - 1];
        }
        total += size;
        ts += (size < rem ? size : rem);
    }
    if (getenv("ASR_PREFILL_CHUNK")) {
        total = T + g_buckets[g_num_buckets - 1];  // capped chunks start anywhere
    }
    return total;
}

// ---------------------------------------------------------------------------
// Runs.
// ---------------------------------------------------------------------------

static int g_num_eos = 0;
static int32_t g_eos[16];

static int is_eos(int32_t tok) {
    for (int i = 0; i < g_num_eos; i++) {
        if (tok == g_eos[i]) {
            return 1;
        }
    }
    return 0;
}

static double g_prefill_ms = 0.0;
static int g_chunks = 0;
static int g_tail_steps = 0;    // prompt-tail steps (decode signature, no
                                // sampling), counted inside g_prefill_ms
static double g_tail_ms = 0.0;  // total wall time of the prompt-tail steps
static double g_decode_ms = 0.0;
static int g_decode_tokens = 0;
static FILE *g_logits_dump = NULL;

// Run the signature (the LiteRtRunCompiledModel call only). The input and
// output tensor buffers were created once in allocate_buffers and are
// reused here.
static void run_chain(Sig *sig);
static void run_sig_exec(Sig *sig) {
    if (sig->chain) {
        run_chain(sig);
        return;
    }
    LiteRtStatus st = LiteRtRunCompiledModel(
        g_sections[sig->section_idx].compiled, sig->idx, sig->num_in,
        sig->in_bufs, sig->num_out, sig->out_bufs);
    if (st != kLiteRtStatusOk) {
        fprintf(stderr, APP ": FAILED: run signature '%s' -> %d (%s)\n",
                sig->key, (int)st, status_str(st));
        exit(1);
    }
}

// Dequantize the outputs to float32 staging (the oracle's
// try_get_dequantized_output); float outputs pass through. The runners'
// kv_slice outputs and the decode logits keep no float32 staging: the host
// cache merge and the argmax read their raw bytes.
static void dequant_outputs(Sig *sig) {
    for (size_t i = 0; i < sig->num_out; i++) {
        Slot *o = &sig->out[i];
        if (!o->f32) {
            continue;
        }
        const uint8_t *base =
            (const uint8_t *)slot_lock(o, kLiteRtTensorBufferLockModeRead,
                                       o->name);
        for (size_t j = 0; j < o->t.n; j++) {
            o->f32[j] = slot_read(o, base, j);
        }
        slot_unlock(o);
    }
}

// Copy every output of src_sig into the same-named input of dst_sig when one
// exists (mirrors the oracle's **dict merge, where extra keys are ignored).
static void merge_outputs(Sig *src_sig, Sig *dst_sig) {
    for (size_t i = 0; i < src_sig->num_out; i++) {
        Slot *dst = sig_in_by_name(dst_sig, src_sig->out[i].name);
        if (dst) {
            copy_out_to_in(dst, &src_sig->out[i]);
        }
    }
}

static int name_in_outputs(const Sig *sig, const char *name) {
    for (size_t i = 0; i < sig->num_out; i++) {
        if (strcmp(sig->out[i].name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

// Verify every input of `sig` has a known source: the cache set, one of
// `extra_names`, or an output of one of `src_sigs`.
static void verify_inputs_bound(Sig *sig, const char *const *extra_names,
                                Sig **src_sigs, int num_src) {
    for (size_t i = 0; i < sig->num_in; i++) {
        const char *name = sig->in[i].name;
        if (cache_index(name) >= 0) {
            continue;
        }
        int known = 0;
        for (size_t k = 0; extra_names[k] != NULL; k++) {
            if (strcmp(name, extra_names[k]) == 0) {
                known = 1;
                break;
            }
        }
        if (!known) {
            for (int k = 0; k < num_src; k++) {
                if (name_in_outputs(src_sigs[k], name)) {
                    known = 1;
                    break;
                }
            }
        }
        if (!known) {
            fail(2, "signature '%s' has input '%s' with no known source",
                 sig->key, name);
        }
    }
}

// Read element i of the persistent cache buffer as a plain float32,
// dequantizing with the cache tensor's own params (the oracle's
// uniform_dequantize: integer q - zp first, then the float32 multiply).
static float cache_read(const CacheT *c, const uint8_t *base, size_t i) {
    const TType *t = &c->t;
    float v = read_elem(t, base + i * elem_size(t->elem));
    if (t->qkind == Q_NONE) {
        return v;
    }
    return (float)((int32_t)(int64_t)v - (int32_t)t->zp) * t->scale;
}

// Fill one cache input slot whose dtype differs from the persistent buffer's
// (e.g. a float32 kv_cache input on a quantized model): dequantize the
// persistent buffer with the cache params and write it with the slot's own
// params, like the NPU executor's float-cache-with-quantized-slice path.
static void fill_cache_one(Slot *s, int ci) {
    CacheT *c = &g_caches[ci];
    if (s->cache_direct) {
        return;  // the slot IS the persistent buffer (shared storage)
    }
    const uint8_t *cbase =
        (const uint8_t *)buf_lock(c->buf, kLiteRtTensorBufferLockModeRead,
                                  c->name);
    uint8_t *sbase =
        (uint8_t *)slot_lock(s, kLiteRtTensorBufferLockModeWrite, s->name);
    for (size_t i = 0; i < s->t.n; i++) {
        slot_write(s, sbase, i, cache_read(c, cbase, i));
    }
    slot_unlock(s);
    buf_unlock(c->buf);
}

// Fill every cache input of `sig` from the persistent cache state.
static void fill_cache_inputs(Sig *sig) {
    for (size_t i = 0; i < sig->num_in; i++) {
        int ci = cache_index(sig->in[i].name);
        if (ci >= 0) {
            fill_cache_one(&sig->in[i], ci);
        }
    }
}

// Merge the runner's kv_slice outputs into the persistent KV cache buffers
// in place on the host, replacing the aux *_cache_update signature run.
// The exported *_cache_update subgraph copies the ENTIRE slice to
// [input_pos[0], input_pos[0] + S): its valid_mask input does not gate the
// copy (verified against the model with full / prefix / tail / alternating
// masks), so all S positions of the chunk are written, which keeps the cache
// contents identical to the oracle's cache_update round trip. time_step + S
// never exceeds the cache length (checked before each run), so no
// wraparound is needed.
static void host_cache_update(const Sig *runner, long time_step, long S) {
    for (int ci = 0; ci < g_num_caches; ci++) {
        CacheT *c = &g_caches[ci];
        if (c->slice_name[0] == '\0') {
            continue;  // no derived slice name: the cache is not updated
        }
        const Slot *slice = sig_out_by_name(runner, c->slice_name);
        if (!slice) {
            continue;  // the runner does not produce this cache's slice
        }
        const TType *ct = &c->t;
        const TType *st = &slice->t;
        // Slice sequence dimension: the cache's time dimension when it
        // carries S there, else the first dimension equal to S.
        int s_td = -1;
        if ((long)st->layout.dimensions[c->time_dim] == S) {
            s_td = c->time_dim;
        } else {
            for (unsigned d = 0; d < st->layout.rank; d++) {
                if ((long)st->layout.dimensions[d] == S) {
                    s_td = (int)d;
                    break;
                }
            }
        }
        if (s_td < 0) {
            fail(2, "'%s' slice '%s' has no dimension matching the chunk "
                    "size %ld",
                 runner->key, slice->name, S);
        }
        // Element strides: the time stride is the product of the dimensions
        // after the sequence dimension (the inner block size), and the outer
        // stride is time stride x sequence length. The (time, inner) block
        // is contiguous in row-major memory, so the raw-copy merge is one
        // memcpy per outer index.
        long st_c = 1, st_s = 1;
        for (unsigned d = (unsigned)c->time_dim + 1; d < ct->layout.rank; d++) {
            st_c *= (long)ct->layout.dimensions[d];
        }
        for (unsigned d = (unsigned)s_td + 1; d < st->layout.rank; d++) {
            st_s *= (long)st->layout.dimensions[d];
        }
        if (st_c != st_s) {
            fail(2, "'%s' slice '%s' inner size %ld does not match the "
                    "'%s' cache inner size %ld",
                 runner->key, slice->name, st_s, c->name, st_c);
        }
        long so_c = st_c * (long)ct->layout.dimensions[c->time_dim];
        long so_s = st_s * (long)st->layout.dimensions[s_td];
        long outer = 1;
        for (unsigned d = 0; d < (unsigned)c->time_dim; d++) {
            outer *= (long)ct->layout.dimensions[d];
        }
        uint8_t *cbase =
            (uint8_t *)buf_lock(c->buf, kLiteRtTensorBufferLockModeWrite,
                                c->name);
        const uint8_t *sbase = (const uint8_t *)slot_lock(
            slice, kLiteRtTensorBufferLockModeRead, slice->name);
        if (c->raw_copy) {
            // Raw element copy (slice and cache share dtype and the same
            // scale/zero_point, verified at startup).
            size_t esize = elem_size(ct->elem);
            for (long o = 0; o < outer; o++) {
                memcpy(cbase + (size_t)o * (size_t)so_c * esize +
                           (size_t)time_step * (size_t)st_c * esize,
                       sbase + (size_t)o * (size_t)so_s * esize,
                       (size_t)S * (size_t)st_c * esize);
            }
        } else {
            // Requantize element by element: dequant with the slice params,
            // quant with the cache params (same rounding as slot_write).
            Slot tmp;
            memset(&tmp, 0, sizeof tmp);
            tmp.t = *ct;
            for (long o = 0; o < outer; o++) {
                for (long p = 0; p < S; p++) {
                    for (long i = 0; i < st_c; i++) {
                        float v = slot_read(slice, sbase,
                                            (size_t)(o * so_s + p * st_s + i));
                        slot_write(
                            &tmp, cbase,
                            (size_t)(o * so_c + (time_step + p) * st_c + i),
                            v);
                    }
                }
            }
        }
        slot_unlock(slice);
        buf_unlock(c->buf);
    }
}

// Prompt tail: one decode-signature step per prompt position in
// [from, g_T - 1). Each step runs exactly like a decode step (the aux
// mask/rope and the host cache update), except the `embeddings` input is the
// pre-built row `from + k` of g_padded_embeds (NOT the decode_embedder
// output: audio rows are features, not token embeddings, so the embedder is
// not run) and nothing is sampled, counted as a decode token, or dumped.
// The step wall time counts into g_prefill_ms (and g_tail_ms); with
// --profile each step's total goes to the tail stage. The caller returns
// g_T - 1 after this: the first real decode step re-embeds prompt token
// g_T - 1 at position g_T - 1, exactly as in the one-chunk prefill.
static void run_prompt_tail(long from) {
    Sig *mask = find_sig(SEC_AU, "decode_mask");
    Sig *rope = find_sig(SEC_AU, "decode_rope");
    Sig *decode = find_sig(SEC_PD, "decode");
    Sig *cu = find_sig(SEC_AU, "decode_cache_update");
    if (!mask || !rope || !decode || !cu) {
        die("missing decode signature");
    }
    Slot *emb = sig_in_by_name(decode, "embeddings");
    if (!emb) {
        die("decode signature has no 'embeddings' input");
    }
    static const char *const dec_extra[] = {"embeddings", "per_layer_embeddings", NULL};
    for (long p = from; p < g_T - 1; p++) {
        double t0 = now_ms();
        double tq = tnow();
        // Aux mask: time_step = p, input_tokens = the prompt token at p,
        // valid_mask (if present).
        int have_time_step = 0, have_input_tokens = 0;
        for (size_t i = 0; i < mask->num_in; i++) {
            Slot *s = &mask->in[i];
            if (strcmp(s->name, "time_step") == 0) {
                if (s->t.n != 1) {
                    die("mask time_step must be a scalar");
                }
                int32_t v = (int32_t)p;
                fill_i32(s, &v);
                have_time_step = 1;
            } else if (strcmp(s->name, "input_tokens") == 0) {
                if (s->t.n != 1) {
                    die("mask input_tokens must be a single token");
                }
                int32_t tok = g_ids[p];
                fill_i32(s, &tok);
                have_input_tokens = 1;
            } else if (strcmp(s->name, "valid_mask") == 0) {
                if (s->t.n != 1) {
                    die("mask valid_mask must be a single value");
                }
                uint8_t *base = (uint8_t *)slot_lock(
                    s, kLiteRtTensorBufferLockModeWrite, s->name);
                slot_write(s, base, 0, 1.0f);
                slot_unlock(s);
            } else {
                fail(2, "decode mask has input '%s' with no known source",
                     s->name);
            }
        }
        if (!have_time_step || !have_input_tokens) {
            die("decode mask is missing time_step or input_tokens");
        }
        // The embeddings input is the pre-built row p of the assembled
        // embeddings (the decode_embedder is not run).
        fill_f32_off(emb, g_padded_embeds, (size_t)p * (size_t)g_hidden);
        fill_ple_rows(decode, p, 1);
        // Aux rope for this single position.
        int have_input_pos = 0;
        for (size_t i = 0; i < rope->num_in; i++) {
            Slot *s = &rope->in[i];
            if (strcmp(s->name, "input_pos") == 0) {
                if (s->t.n != 1) {
                    die("rope input_pos must be a single position");
                }
                int32_t v = (int32_t)p;
                fill_i32(s, &v);
                have_input_pos = 1;
            } else {
                fail(2, "decode rope has input '%s' with no known source",
                     s->name);
            }
        }
        if (!have_input_pos) {
            die("decode rope is missing input_pos");
        }
        fill_cache_inputs(decode);
        tick(STG_INPUT_QUANT, tnow() - tq);
        tq = tnow();
        run_sig_exec(mask);
        tick(STG_AUX_MASK, tnow() - tq);
        tq = tnow();
        dequant_outputs(mask);
        tick(STG_OUTPUT_DEQUANT, tnow() - tq);
        tq = tnow();
        run_sig_exec(rope);
        tick(STG_AUX_ROPE, tnow() - tq);
        tq = tnow();
        dequant_outputs(rope);
        tick(STG_OUTPUT_DEQUANT, tnow() - tq);
        tq = tnow();
        merge_outputs(mask, decode);
        merge_outputs(rope, decode);
        tick(STG_INPUT_QUANT, tnow() - tq);
        {
            Sig *srcs[2] = {mask, rope};
            verify_inputs_bound(decode, dec_extra, srcs, 2);
        }
        tq = tnow();
        run_sig_exec(decode);
        tick(STG_DECODE, tnow() - tq);
        tq = tnow();
        dequant_outputs(decode);
        tick(STG_OUTPUT_DEQUANT, tnow() - tq);
        // No argmax/sampling/logits dump: the step only fills the cache.
        tq = tnow();
        host_cache_update(decode, p, 1);
        tick(STG_CACHE_UPDATE, tnow() - tq);
        double dt = now_ms() - t0;
        g_prefill_ms += dt;
        g_tail_ms += dt;
        g_tail_steps++;
        if (g_profile) {
            tick(STG_TAIL, dt);
        }
    }
}

// Prefill loop mirroring Executor.prefill_chunk. Returns the time_step after
// prefill (T-1; the oracle rolls back so decode re-embeds the last prompt
// token).
static long run_prefill(void) {
    long time_step = 0;
    int generate = 0;
    long max_bucket = g_buckets[g_num_buckets - 1];
    int32_t *positions =
        (int32_t *)malloc((size_t)max_bucket * sizeof(int32_t));
    if (!positions) {
        die("out of memory");
    }
    char key[128];
    static const char *const pd_extra[] = {"embeddings", "per_layer_embeddings", NULL};
    // A second prefill chunk (time_step > 0) gives garbage on the MediaTek
    // NPU (the compiled prefill mishandles a non-empty cache; the CPU
    // reference is correct), so by default the prompt tail beyond the first
    // prefill chunk runs with the DECODE signature (run_prompt_tail: one
    // step per prompt position, no sampling). ASR_PREFILL_CHUNK (diagnostic)
    // keeps the old multi-chunk prefill; ASR_PREFILL_SPLIT=N (diagnostic,
    // N >= 1) splits after [0, min(N, max bucket)) even when the prompt
    // fits one bucket. It is ignored when ASR_PREFILL_CHUNK is set.
    int use_tail = 0;
    long tail_s = 0;
    if (!getenv("ASR_PREFILL_CHUNK")) {
        const char *split_env = getenv("ASR_PREFILL_SPLIT");
        long split = split_env ? atol(split_env) : 0;
        if (split >= 1) {
            tail_s = split < max_bucket ? split : max_bucket;
            if (tail_s > g_T) {
                tail_s = g_T;
            }
            use_tail = 1;
        } else if (g_T > max_bucket) {
            tail_s = max_bucket;
            use_tail = 1;
        }
    }
    while (!generate) {
        long remaining = g_T - time_step;
        if (use_tail && time_step == 0) {
            // Tail path: the first chunk covers [0, tail_s) only.
            remaining = tail_s;
        }
        long input_size = 0;
        for (int b = 0; b < g_num_buckets; b++) {
            if (g_buckets[b] >= remaining) {
                input_size = g_buckets[b];
                break;
            }
        }
        if (input_size == 0) {
            input_size = max_bucket;
        }
        if (time_step + input_size > g_cache_length) {
            fail(2, "prefill chunk exceeds the cache length");
        }
        long real = remaining < input_size ? remaining : input_size;
        {
            // Diagnostic: ASR_PREFILL_CHUNK=N caps the real tokens per prefill chunk (forces multi-chunk prefill).
            const char *cap_env = getenv("ASR_PREFILL_CHUNK");
            long cap = cap_env ? atol(cap_env) : 0;
            if (cap > 0 && real > cap) {
                real = cap;
            }
        }

        snprintf(key, sizeof key, "prefill_mask_%ld", input_size);
        Sig *mask = find_sig(SEC_AU, key);
        snprintf(key, sizeof key, "prefill_rope_%ld", input_size);
        Sig *rope = find_sig(SEC_AU, key);
        snprintf(key, sizeof key, "prefill_%ld", input_size);
        Sig *prefill = find_sig(SEC_PD, key);
        snprintf(key, sizeof key, "prefill_cache_update_%ld", input_size);
        Sig *cu = find_sig(SEC_AU, key);
        if (!mask || !rope || !prefill || !cu) {
            die("missing prefill signature");
        }

        double tq = tnow();
        // Aux mask inputs: time_step, input_tokens, valid_mask (if present).
        int have_time_step = 0, have_input_tokens = 0;
        for (size_t i = 0; i < mask->num_in; i++) {
            Slot *s = &mask->in[i];
            if (strcmp(s->name, "time_step") == 0) {
                if (s->t.n != 1) {
                    die("mask time_step must be a scalar");
                }
                int32_t v = (int32_t)time_step;
                fill_i32(s, &v);
                have_time_step = 1;
            } else if (strcmp(s->name, "input_tokens") == 0) {
                if (s->t.n != (size_t)input_size) {
                    fail(2, "mask input_tokens has %zu elements, expected "
                            "%ld",
                         s->t.n, input_size);
                }
                fill_i32_zeros_tail(s, g_ids + time_step, (size_t)real);
                have_input_tokens = 1;
            } else if (strcmp(s->name, "valid_mask") == 0) {
                if (s->t.n != (size_t)input_size) {
                    fail(2, "mask valid_mask has %zu elements, expected %ld",
                         s->t.n, input_size);
                }
                uint8_t *base = (uint8_t *)slot_lock(
                    s, kLiteRtTensorBufferLockModeWrite, s->name);
                for (size_t j = 0; j < s->t.n; j++) {
                    slot_write(s, base, j, j < (size_t)real ? 1.0f : 0.0f);
                }
                slot_unlock(s);
            } else {
                fail(2, "mask signature '%s' has input '%s' with no known "
                        "source",
                     mask->key, s->name);
            }
        }
        if (!have_time_step || !have_input_tokens) {
            die("mask signature is missing time_step or input_tokens");
        }
        // Aux rope: input_pos = arange(time_step, time_step + input_size).
        int have_input_pos = 0;
        for (size_t i = 0; i < rope->num_in; i++) {
            Slot *s = &rope->in[i];
            if (strcmp(s->name, "input_pos") == 0) {
                if (s->t.n != (size_t)input_size) {
                    fail(2, "rope input_pos has %zu elements, expected %ld",
                         s->t.n, input_size);
                }
                for (size_t j = 0; j < (size_t)input_size; j++) {
                    positions[j] = (int32_t)(time_step + (long)j);
                }
                fill_i32(s, positions);
                have_input_pos = 1;
            } else {
                fail(2, "rope signature '%s' has input '%s' with no known "
                        "source",
                     rope->key, s->name);
            }
        }
        if (!have_input_pos) {
            die("rope signature is missing input_pos");
        }
        // Embeddings slice (precomputed, zero-padded to padded_total).
        Slot *emb = sig_in_by_name(prefill, "embeddings");
        if (!emb) {
            die("prefill signature has no 'embeddings' input");
        }
        if (emb->t.n != (size_t)input_size * (size_t)g_hidden) {
            fprintf(stderr, APP ": prefill embeddings has %zu elements, "
                    "expected %ld\n",
                    emb->t.n, input_size * g_hidden);
            exit(2);
        }
        fill_f32_off(emb, g_padded_embeds,
                     (size_t)time_step * (size_t)g_hidden);
        fill_ple_rows(prefill, time_step, input_size);
        tick(STG_INPUT_QUANT, tnow() - tq);

        double t0 = now_ms();
        tq = tnow();
        run_sig_exec(mask);
        tick(STG_AUX_MASK, tnow() - tq);
        tq = tnow();
        dequant_outputs(mask);
        tick(STG_OUTPUT_DEQUANT, tnow() - tq);
        tq = tnow();
        run_sig_exec(rope);
        tick(STG_AUX_ROPE, tnow() - tq);
        tq = tnow();
        dequant_outputs(rope);
        tick(STG_OUTPUT_DEQUANT, tnow() - tq);
        tq = tnow();
        merge_outputs(mask, prefill);
        merge_outputs(rope, prefill);
        fill_cache_inputs(prefill);
        tick(STG_INPUT_QUANT, tnow() - tq);
        {
            Sig *srcs[2] = {mask, rope};
            verify_inputs_bound(prefill, pd_extra, srcs, 2);
        }
        tq = tnow();
        run_sig_exec(prefill);
        tick(STG_PREFILL, tnow() - tq);
        tq = tnow();
        dequant_outputs(prefill);
        tick(STG_OUTPUT_DEQUANT, tnow() - tq);
        // Host-side merge of the fresh kv_slice outputs into the persistent
        // KV buffers (the aux cache_update signature is not run).
        tq = tnow();
        host_cache_update(prefill, time_step, input_size);
        tick(STG_CACHE_UPDATE, tnow() - tq);
        g_prefill_ms += now_ms() - t0;
        g_chunks++;

        time_step += real;
        if (use_tail) {
            // Tail path: the prompt tail [tail_s, g_T - 1) runs with the
            // decode signature, then decode starts at g_T - 1 (it re-embeds
            // prompt token g_T - 1, exactly as after the one-chunk prefill).
            run_prompt_tail(time_step);
            free(positions);
            return g_T - 1;
        }
        if (time_step == g_T) {
            // Oracle: generate = True; time_step -= 1.
            generate = 1;
            time_step -= 1;
        }
    }
    free(positions);
    return time_step;
}

// Decode loop mirroring Executor.decode_step. Samples at most max_steps
// tokens, stopping early at an EOS token.
static void run_decode(long max_steps, long time_step, int32_t token,
                       int32_t *tokens_out) {
    Sig *mask = find_sig(SEC_AU, "decode_mask");
    Sig *embedder = find_sig(SEC_ED, "decode_embedder");
    Sig *rope = find_sig(SEC_AU, "decode_rope");
    Sig *decode = find_sig(SEC_PD, "decode");
    Sig *cu = find_sig(SEC_AU, "decode_cache_update");
    if (!mask || !embedder || !rope || !decode || !cu) {
        die("missing decode signature");
    }
    const Slot *logits = sig_out_by_name(decode, "logits");
    if (!logits) {
        die("decode signature has no 'logits' output");
    }
    Slot *emb = sig_in_by_name(decode, "embeddings");
    if (!emb) {
        die("decode signature has no 'embeddings' input");
    }
    const Slot *emb_out = sig_out_by_name(embedder, "embeddings");
    if (!emb_out) {
        die("decode_embedder has no 'embeddings' output");
    }
    static const char *const dec_extra[] = {"embeddings", "per_layer_embeddings", NULL};
    for (long k = 0; k < max_steps; k++) {
        double t0 = now_ms();
        double snap[STG_COUNT];
        if (g_profile) {
            for (int i = 0; i < STG_COUNT; i++) {
                snap[i] = g_stage[i].total_ms;
            }
        }
        double tq = tnow();
        // Aux mask: time_step, input_tokens, valid_mask (if present).
        int have_time_step = 0, have_input_tokens = 0;
        for (size_t i = 0; i < mask->num_in; i++) {
            Slot *s = &mask->in[i];
            if (strcmp(s->name, "time_step") == 0) {
                if (s->t.n != 1) {
                    die("mask time_step must be a scalar");
                }
                int32_t v = (int32_t)time_step;
                fill_i32(s, &v);
                have_time_step = 1;
            } else if (strcmp(s->name, "input_tokens") == 0) {
                if (s->t.n != 1) {
                    die("mask input_tokens must be a single token");
                }
                fill_i32(s, &token);
                have_input_tokens = 1;
            } else if (strcmp(s->name, "valid_mask") == 0) {
                if (s->t.n != 1) {
                    die("mask valid_mask must be a single value");
                }
                uint8_t *base = (uint8_t *)slot_lock(
                    s, kLiteRtTensorBufferLockModeWrite, s->name);
                slot_write(s, base, 0, 1.0f);
                slot_unlock(s);
            } else {
                fail(2, "decode mask has input '%s' with no known source",
                     s->name);
            }
        }
        if (!have_time_step || !have_input_tokens) {
            die("decode mask is missing time_step or input_tokens");
        }
        // Embed the current token via the decode embedder.
        int have_token_ids = 0;
        for (size_t i = 0; i < embedder->num_in; i++) {
            Slot *s = &embedder->in[i];
            if (strcmp(s->name, "token_ids") == 0) {
                if (s->t.n != 1) {
                    die("embedder token_ids must be a single token");
                }
                fill_i32(s, &token);
                have_token_ids = 1;
            } else {
                fail(2, "decode_embedder has input '%s' with no known "
                        "source",
                     s->name);
            }
        }
        if (!have_token_ids) {
            die("decode_embedder is missing token_ids");
        }
        // Aux rope for this single position.
        int have_input_pos = 0;
        for (size_t i = 0; i < rope->num_in; i++) {
            Slot *s = &rope->in[i];
            if (strcmp(s->name, "input_pos") == 0) {
                if (s->t.n != 1) {
                    die("rope input_pos must be a single position");
                }
                int32_t v = (int32_t)time_step;
                fill_i32(s, &v);
                have_input_pos = 1;
            } else {
                fail(2, "decode rope has input '%s' with no known source",
                     s->name);
            }
        }
        if (!have_input_pos) {
            die("decode rope is missing input_pos");
        }
        fill_cache_inputs(decode);
        tick(STG_INPUT_QUANT, tnow() - tq);
        tq = tnow();
        run_sig_exec(mask);
        tick(STG_AUX_MASK, tnow() - tq);
        tq = tnow();
        dequant_outputs(mask);
        tick(STG_OUTPUT_DEQUANT, tnow() - tq);
        tq = tnow();
        run_sig_exec(embedder);
        tick(STG_EMBEDDER, tnow() - tq);
        tq = tnow();
        dequant_outputs(embedder);
        tick(STG_OUTPUT_DEQUANT, tnow() - tq);
        tq = tnow();
        run_sig_exec(rope);
        tick(STG_AUX_ROPE, tnow() - tq);
        tq = tnow();
        dequant_outputs(rope);
        tick(STG_OUTPUT_DEQUANT, tnow() - tq);
        tq = tnow();
        copy_out_to_in(emb, emb_out);
        if (g_have_ple) {
            Sig *ple = ple_sig();
            double tp = tnow();
            ple_token(ple, token);
            tick(STG_EMBEDDER, tnow() - tp);
            copy_out_to_in(sig_in_by_name(decode, "per_layer_embeddings"),
                           &ple->out[0]);
        }
        merge_outputs(mask, decode);
        merge_outputs(rope, decode);
        tick(STG_INPUT_QUANT, tnow() - tq);
        {
            Sig *srcs[2] = {mask, rope};
            verify_inputs_bound(decode, dec_extra, srcs, 2);
        }
        tq = tnow();
        run_sig_exec(decode);
        tick(STG_DECODE, tnow() - tq);
        tq = tnow();
        dequant_outputs(decode);
        tick(STG_OUTPUT_DEQUANT, tnow() - tq);
        // Greedy sample on the logits (oracle sample_logits; ties go to the
        // last index among equal maxima). For int16 per-tensor quantized
        // logits with a positive scale the argmax runs directly on the raw
        // int16 values (monotonic in the dequantized value, same index); the
        // row is dequantized to float32 only when --dump-logits is given or
        // the raw path is unavailable.
        tq = tnow();
        int32_t next_token;
        int have_f32_row = 0;
        const uint8_t *lbase =
            (const uint8_t *)slot_lock(logits, kLiteRtTensorBufferLockModeRead,
                                       "logits");
        if (g_logits_raw_argmax) {
            next_token = (int32_t)argmax_i16((const int16_t *)lbase,
                                             logits->t.n);
        } else if (logits->t.elem == kLiteRtElementTypeFloat32) {
            next_token = (int32_t)argmax_f32((const float *)lbase,
                                             logits->t.n);
        } else {
            for (size_t i = 0; i < logits->t.n; i++) {
                g_logits_scratch[i] = slot_read(logits, lbase, i);
            }
            have_f32_row = 1;
            next_token = (int32_t)argmax_f32(g_logits_scratch, logits->t.n);
        }
        if (g_logits_dump) {
            // One dequantized float32 row per step (ref_logits.f32 layout).
            if (logits->t.elem == kLiteRtElementTypeFloat32) {
                if (fwrite(lbase, sizeof(float), logits->t.n,
                           g_logits_dump) != logits->t.n) {
                    fail(1, "short write to the logits dump");
                }
            } else {
                if (!have_f32_row) {
                    for (size_t i = 0; i < logits->t.n; i++) {
                        g_logits_scratch[i] = slot_read(logits, lbase, i);
                    }
                }
                if (fwrite(g_logits_scratch, sizeof(float), logits->t.n,
                           g_logits_dump) != logits->t.n) {
                    fail(1, "short write to the logits dump");
                }
            }
        }
        slot_unlock(logits);
        tick(STG_ARGMAX, tnow() - tq);
        // Host-side merge of the fresh kv_slice outputs into the persistent
        // KV buffers (the aux cache_update signature is not run).
        tq = tnow();
        host_cache_update(decode, time_step, 1);
        tick(STG_CACHE_UPDATE, tnow() - tq);
        tokens_out[k] = next_token;
        double dt = now_ms() - t0;
        g_decode_ms += dt;
        g_decode_tokens++;
        if (g_profile) {
            double named = 0.0;
            for (int i = 0; i < STG_OTHER; i++) {
                named += g_stage[i].total_ms - snap[i];
            }
            double other = dt - named;
            if (other < 0.0) {
                other = 0.0;
            }
            g_stage[STG_OTHER].total_ms += other;
            g_stage[STG_OTHER].calls++;
            record_step(dt);
        }
        if (is_eos(next_token)) {
            break;
        }
        token = next_token;
        time_step++;
    }
}

// ---------------------------------------------------------------------------
// On-device input assembly (--audio-features/--prompt-ids): run the
// decode_embedder per prompt id (one token at a time, dequantizing its
// output if quantized) and splice the audio feature rows into the pad rows
// in order, mirroring make_feat_fixture.py exactly.
// ---------------------------------------------------------------------------

// Write n float32 values (the T x H prefill embeddings) to the --dump-embeds
// file.
static void dump_embeds_file(const char *path, const float *embeds, long n) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fail(2, "cannot open '%s' for writing: %s", path, strerror(errno));
    }
    if (fwrite(embeds, sizeof(float), (size_t)n, f) != (size_t)n) {
        fail(1, "short write to the embeds dump");
    }
    if (fclose(f) != 0) {
        fail(1, "fclose on the embeds dump: %s", strerror(errno));
    }
}

// Returns a freshly allocated g_T x g_hidden float32 array (the caller
// frees it).
static float *assemble_embeds(void) {
    Sig *embedder = find_sig(SEC_ED, "decode_embedder");
    if (!embedder) {
        die("missing decode_embedder signature");
    }
    Slot *tok = sig_in_by_name(embedder, "token_ids");
    if (!tok) {
        die("decode_embedder is missing token_ids");
    }
    if (tok->t.n != 1) {
        die("embedder token_ids must be a single token");
    }
    for (size_t i = 0; i < embedder->num_in; i++) {
        Slot *s = &embedder->in[i];
        if (strcmp(s->name, "token_ids") != 0) {
            fprintf(stderr, APP ": decode_embedder has input '%s' with no "
                    "known source\n",
                    s->name);
            exit(2);
        }
    }
    const Slot *emb_out = sig_out_by_name(embedder, "embeddings");
    if (!emb_out) {
        die("decode_embedder has no 'embeddings' output");
    }
    if (!emb_out->f32) {
        die("decode_embedder output has no float32 staging");
    }
    if (emb_out->t.n != (size_t)g_hidden) {
        fprintf(stderr, APP ": decode_embedder output has %zu elements, "
                "expected %ld\n",
                emb_out->t.n, g_hidden);
        exit(2);
    }
    float *out =
        (float *)malloc((size_t)g_T * (size_t)g_hidden * sizeof(float));
    if (!out) {
        die("out of memory");
    }
    double tq = tnow();
    for (long t = 0; t < g_T; t++) {
        fill_i32(tok, &g_ids[t]);
        run_sig_exec(embedder);
        dequant_outputs(embedder);
        memcpy(out + (size_t)t * (size_t)g_hidden, emb_out->f32,
               (size_t)g_hidden * sizeof(float));
    }
    // Splice the feature rows into the pad rows in order
    // (make_feat_fixture.py: emb[ids == PAD] = feats). The counts were
    // verified at startup, so k never runs past the features.
    size_t k = 0;
    for (long t = 0; t < g_T; t++) {
        if (g_ids[t] == g_audio_pad_id) {
            memcpy(out + (size_t)t * (size_t)g_hidden,
                   g_features + k * (size_t)g_hidden,
                   (size_t)g_hidden * sizeof(float));
            k++;
        }
    }
    tick(STG_ASSEMBLE, tnow() - tq);
    return out;
}

// ---------------------------------------------------------------------------
// Per-layer embeddings (Gemma 3n / Gemma 4, 2026-11-11): an optional CPU
// section "per_layer_embedder" maps token ids to a [layers x dim] row that the
// prefill/decode signatures take as input "per_layer_embeddings". Prompt rows
// are built once per request into g_padded_ple (zero-padded like
// g_padded_embeds); decode steps run decode_per_layer_embedder on the sampled
// token.
// ---------------------------------------------------------------------------
static Sig *ple_sig(void) {
    Sig *ple = find_sig(SEC_PLE, "decode_per_layer_embedder");
    if (!ple) {
        die("per_layer_embedder has no decode_per_layer_embedder signature");
    }
    if (ple->num_in != 1 || strcmp(ple->in[0].name, "token_ids") != 0 ||
        ple->in[0].t.n != 1) {
        die("decode_per_layer_embedder must take one input token_ids[1]");
    }
    if (ple->num_out != 1 || !ple->out[0].f32) {
        die("decode_per_layer_embedder must have one float-staged output");
    }
    if (g_ple_n == 0) {
        g_ple_n = (long)ple->out[0].t.n;
        Sig *decode = find_sig(SEC_PD, "decode");
        Slot *in = decode ? sig_in_by_name(decode, "per_layer_embeddings")
                          : NULL;
        if (!in || in->t.n != (size_t)g_ple_n) {
            fail(2, "decode per_layer_embeddings input missing or not %ld "
                    "elements",
                 g_ple_n);
        }
    }
    return ple;
}

// Run the per-layer embedder for one token; the row is in ple->out[0].f32.
static void ple_token(Sig *ple, int32_t token) {
    fill_i32(&ple->in[0], &token);
    run_sig_exec(ple);
    dequant_outputs(ple);
}

// Build g_padded_ple for the current prompt (g_ids, g_T, g_padded_total).
static void build_padded_ple(void) {
    free(g_padded_ple);
    g_padded_ple = NULL;
    if (!g_have_ple) {
        return;
    }
    Sig *ple = ple_sig();
    g_padded_ple = (float *)calloc((size_t)g_padded_total * (size_t)g_ple_n,
                                   sizeof(float));
    if (!g_padded_ple) {
        die("out of memory");
    }
    double tq = tnow();
    for (long t = 0; t < g_T; t++) {
        ple_token(ple, g_ids[t]);
        memcpy(g_padded_ple + (size_t)t * (size_t)g_ple_n, ple->out[0].f32,
               (size_t)g_ple_n * sizeof(float));
    }
    tick(STG_EMBEDDER, tnow() - tq);
}

// Fill a prefill/decode signature's per_layer_embeddings input from
// g_padded_ple rows starting at position `pos` (no-op without PLE).
static void fill_ple_rows(Sig *sig, long pos, long rows) {
    if (!g_have_ple) {
        return;
    }
    Slot *in = sig_in_by_name(sig, "per_layer_embeddings");
    if (!in || in->t.n != (size_t)rows * (size_t)g_ple_n) {
        fail(2, "signature '%s' per_layer_embeddings missing or not %ld x %ld",
             sig->key, rows, g_ple_n);
    }
    fill_f32_off(in, g_padded_ple, (size_t)pos * (size_t)g_ple_n);
}

// ---------------------------------------------------------------------------
// Server mode (--server): the setup (environment, model load, compile,
// buffer allocation) runs once, then requests are served from stdin one
// line at a time until EOF or a QUIT line:
//   <audio_features_path> <prompt_ids_path> [max_new] [dump_logits_path]
// Each request resets ALL per-request state to what a fresh process starts
// with (the KV cache back to its initial value, time step 0, sampled
// tokens, logits/argmax state, the prefill/decode bookkeeping and the
// timing/profile counters) and runs exactly the one-shot
// --audio-features/--prompt-ids pipeline: assemble the embeddings, prefill,
// decode, then print the one-shot output lines (the TOKENS line and the
// timing line, plus the --profile stages when enabled) followed by DONE.
// A malformed or failing request prints ERROR <message> then DONE and the
// server keeps running; only the fatal LiteRT paths exit.
// ---------------------------------------------------------------------------

// Snapshot of the persistent KV cache buffers taken right after
// allocate_buffers: the initial state a fresh process starts with (zeros in
// float -> the quantized 0.0 per tensor). Restored before every request.
static uint8_t *g_cache_snap[MAX_CACHES];
static size_t g_cache_snap_bytes[MAX_CACHES];

static void snapshot_caches(void) {
    for (int i = 0; i < g_num_caches; i++) {
        size_t bytes = g_caches[i].t.n * elem_size(g_caches[i].t.elem);
        g_cache_snap[i] = (uint8_t *)malloc(bytes ? bytes : 1);
        if (!g_cache_snap[i]) {
            die("out of memory");
        }
        g_cache_snap_bytes[i] = bytes;
        const uint8_t *base = (const uint8_t *)buf_lock(
            g_caches[i].buf, kLiteRtTensorBufferLockModeRead, g_caches[i].name);
        memcpy(g_cache_snap[i], base, bytes);
        buf_unlock(g_caches[i].buf);
    }
}

static void reset_caches(void) {
    for (int i = 0; i < g_num_caches; i++) {
        if (g_cache_snap_bytes[i] == 0) {
            continue;
        }
        uint8_t *base = (uint8_t *)buf_lock(g_caches[i].buf,
                                            kLiteRtTensorBufferLockModeWrite,
                                            g_caches[i].name);
        memcpy(base, g_cache_snap[i], g_cache_snap_bytes[i]);
        buf_unlock(g_caches[i].buf);
    }
}

// Read one line from f into a grown buffer (NULL at EOF before any char).
static char *read_request_line(FILE *f) {
    size_t cap = 8192, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) {
        die("out of memory");
    }
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '\n') {
            break;
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char *p = (char *)realloc(buf, cap);
            if (!p) {
                free(buf);
                die("out of memory");
            }
            buf = p;
        }
        buf[len++] = (char)c;
    }
    if (len == 0 && c == EOF) {
        free(buf);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

// Run one request: load the inputs exactly like the one-shot
// --audio-features/--prompt-ids path, reset the per-request state, run the
// pipeline and print the one-shot output lines. Returns 0 on success; on a
// request-level failure sets err to the message and returns 1 (the fatal
// LiteRT paths still exit).
static int do_request(const char *feat_path, const char *ids_path,
                      long max_new, const char *dump_path, char *err,
                      size_t err_cap) {
    int rc = 0;
    int32_t *tokens = NULL;
    float *assembled = NULL;
    g_in_request = 1;
    if (setjmp(g_req_jmp) == 0) {
        // Inputs (the one-shot --audio-features/--prompt-ids checks).
        struct stat st;
        if (stat(ids_path, &st) != 0) {
            fail(2, "stat('%s'): %s", ids_path, strerror(errno));
        }
        if ((size_t)st.st_size % 4 != 0) {
            fail(2, "ids file is %lld bytes; expected a multiple of 4",
                 (long long)st.st_size);
        }
        g_T = (long)(st.st_size / 4);
        if (g_T <= 0) {
            fail(2, "ids file is empty");
        }
        if (g_T >= g_cache_length) {
            fail(2, "T=%ld is not below the cache length %ld", g_T,
                 g_cache_length);
        }
        if (g_T + max_new > g_cache_length) {
            fail(2, "prompt has %ld tokens; with max_new %ld it exceeds the "
                    "cache length %ld",
                 g_T, max_new, g_cache_length);
        }
        g_ids = (int32_t *)read_file(ids_path, (size_t)g_T * 4, "ids");
        g_padded_total = padded_total_for(g_T);
        if (stat(feat_path, &st) != 0) {
            fail(2, "stat('%s'): %s", feat_path, strerror(errno));
        }
        size_t row_bytes = (size_t)g_hidden * sizeof(float);
        if ((size_t)st.st_size % row_bytes != 0) {
            fail(2, "'%s' (audio-features) is %lld bytes; expected a "
                    "multiple of %zu (%ld x float32)",
                 feat_path, (long long)st.st_size, row_bytes, g_hidden);
        }
        g_feat_rows = (long)(st.st_size / row_bytes);
        long n_pad = 0;
        for (long t = 0; t < g_T; t++) {
            if (g_ids[t] == g_audio_pad_id) {
                n_pad++;
            }
        }
        if (n_pad != g_feat_rows) {
            fail(2, "%ld prompt ids equal --audio-pad-id %d but the "
                    "audio-features file has %ld rows",
                 n_pad, (int)g_audio_pad_id, g_feat_rows);
        }
        g_features = (float *)read_file(feat_path,
                                        (size_t)g_feat_rows * row_bytes,
                                        "audio-features");
        fprintf(stderr,
                APP ": T=%ld H=%ld cache_length=%ld buckets=%d first=%ld "
                "last=%ld padded=%ld\n",
                g_T, g_hidden, g_cache_length, g_num_buckets, g_buckets[0],
                g_buckets[g_num_buckets - 1], g_padded_total);
        // Reset ALL per-request state to the fresh-process starting point:
        // the KV cache, time step 0 (run_prefill starts from 0), the
        // sampled tokens and the logits/argmax state (recomputed from the
        // next run on), the prefill/tail bookkeeping and the
        // timing/profile counters.
        g_prefill_ms = 0.0;
        g_chunks = 0;
        g_tail_steps = 0;
        g_tail_ms = 0.0;
        g_decode_ms = 0.0;
        g_decode_tokens = 0;
        if (g_profile) {
            memset(g_stage, 0, sizeof g_stage);
            g_num_steps = 0;
        }
        g_logits_dump = NULL;
        reset_caches();
        // On-device assembly, exactly like the one-shot feat path.
        assembled = assemble_embeds();
        free(g_padded_embeds);
        g_padded_embeds = (float *)calloc(
            (size_t)g_padded_total * (size_t)g_hidden, sizeof(float));
        if (!g_padded_embeds) {
            die("out of memory");
        }
        memcpy(g_padded_embeds, assembled,
               (size_t)g_T * (size_t)g_hidden * sizeof(float));
        free(assembled);
        assembled = NULL;
        build_padded_ple();
        if (dump_path) {
            g_logits_dump = fopen(dump_path, "wb");
            if (!g_logits_dump) {
                fail(2, "cannot open '%s' for writing: %s", dump_path,
                     strerror(errno));
            }
        }
        long time_step = run_prefill();
        long max_steps = max_new < (g_cache_length - time_step)
                             ? max_new
                             : (g_cache_length - time_step);
        if (max_steps < 0) {
            max_steps = 0;
        }
        tokens = (int32_t *)malloc(
            (size_t)(max_steps > 0 ? max_steps : 1) * sizeof(int32_t));
        if (!tokens) {
            die("out of memory");
        }
        int32_t next_token = g_ids[g_T - 1];  // oracle: next_decode_token
        run_decode(max_steps, time_step, next_token, tokens);
        if (g_logits_dump && fclose(g_logits_dump) != 0) {
            fail(1, "fclose on the logits dump: %s", strerror(errno));
        }
        g_logits_dump = NULL;
        // The one-shot output lines.
        // Leading newline: a library log line right before (e.g. DestroyAccelerator) may lack its own
        // trailing newline and would glue onto TOKENS, breaking the line-anchored parsers (see 2026-11-08 E1).
        printf("\nTOKENS");
        for (int i = 0; i < g_decode_tokens; i++) {
            printf(" %d", tokens[i]);
        }
        printf("\n");
        if (g_tail_steps > 0) {
            printf("prefill_ms=%.1f chunks=%d tail_steps=%d tail_ms=%.1f "
                   "decode_tokens=%d decode_ms_per_tok=%.2f setup_ms=%.1f\n",
                   g_prefill_ms, g_chunks, g_tail_steps, g_tail_ms,
                   g_decode_tokens,
                   g_decode_tokens > 0
                       ? g_decode_ms / (double)g_decode_tokens
                       : 0.0,
                   g_setup_ms);
        } else {
            printf("prefill_ms=%.1f chunks=%d decode_tokens=%d "
                   "decode_ms_per_tok=%.2f setup_ms=%.1f\n",
                   g_prefill_ms, g_chunks, g_decode_tokens,
                   g_decode_tokens > 0
                       ? g_decode_ms / (double)g_decode_tokens
                       : 0.0,
                   g_setup_ms);
        }
        if (g_profile) {
            print_profile();
        }
        free(tokens);
        tokens = NULL;
    } else {
        // A request-level failure: report it and keep serving.
        snprintf(err, err_cap, "%s", g_req_err);
        rc = 1;
    }
    // Per-request teardown: drop the request's buffers (the next request
    // sets everything it uses; the KV cache is restored from the snapshot at
    // the start of the next request).
    free(tokens);
    free(assembled);
    free(g_padded_embeds);
    g_padded_embeds = NULL;
    free(g_padded_ple);
    g_padded_ple = NULL;
    free(g_ids);
    g_ids = NULL;
    g_T = 0;
    free(g_features);
    g_features = NULL;
    g_feat_rows = 0;
    g_padded_total = 0;
    if (g_logits_dump) {
        fclose(g_logits_dump);
        g_logits_dump = NULL;
    }
    g_in_request = 0;
    return rc;
}

// The server loop: serve one request per stdin line until EOF or QUIT.
static void serve_requests(void) {
    for (;;) {
        char *line = read_request_line(stdin);
        if (!line) {
            break;  // EOF
        }
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == ' ' || line[len - 1] == '\t' ||
                           line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        char *body = line;
        while (*body == ' ' || *body == '\t') {
            body++;
        }
        if (strcmp(body, "QUIT") == 0) {
            free(line);
            break;
        }
        // <audio_features_path> <prompt_ids_path> [max_new]
        // [dump_logits_path]
        char *tok[5] = {NULL, NULL, NULL, NULL, NULL};
        int ntok = 0;
        char *save = NULL;
        for (char *t = strtok_r(body, " \t", &save); t != NULL && ntok < 5;
             t = strtok_r(NULL, " \t", &save)) {
            tok[ntok++] = t;
        }
        if (ntok < 2 || ntok > 4) {
            printf("ERROR malformed request: expected "
                   "<audio_features_path> <prompt_ids_path> [max_new] "
                   "[dump_logits_path]\n");
        } else {
            int bad = 0;
            long max_new = g_max_new;
            if (ntok >= 3) {
                char *end = NULL;
                long v = strtol(tok[2], &end, 10);
                if (!end || *end != '\0' || v < 1) {
                    printf("ERROR max_new must be a positive integer\n");
                    bad = 1;
                } else {
                    max_new = v;
                }
            }
            if (!bad) {
                const char *dump_path = (ntok == 4) ? tok[3] : NULL;
                char err[512];
                int rc = do_request(tok[0], tok[1], max_new, dump_path, err,
                                    sizeof err);
                if (rc) {
                    printf("ERROR %s\n", err);
                }
            }
        }
        printf("DONE\n");
        fflush(stdout);
        free(line);
    }
}

// ---------------------------------------------------------------------------
// The one-shot run: input loading, buffer allocation, embedding assembly,
// prefill, decode, and output. --server mode runs the same pipeline per
// request in do_request instead (from the fresh per-request state).
// ---------------------------------------------------------------------------

static void run_oneshot(double setup_ms, int have_embeds_pair,
                        const char *embeds_path, const char *ids_path,
                        const char *audio_features_path,
                        const char *prompt_ids_path,
                        const char *dump_embeds_path, const char *dump_path,
                        long max_new) {
    // Inputs.
    {
        const char *ids_eff = have_embeds_pair ? ids_path : prompt_ids_path;
        struct stat st;
        if (stat(ids_eff, &st) != 0) {
            fail(2, "stat('%s'): %s", ids_eff, strerror(errno));
        }
        if ((size_t)st.st_size % 4 != 0) {
            fail(2, "ids file is %lld bytes; expected a multiple of 4",
                 (long long)st.st_size);
        }
        g_T = (long)(st.st_size / 4);
    }
    if (g_T <= 0) {
        die("ids file is empty");
    }
    if (g_T >= g_cache_length) {
        fail(2, "T=%ld is not below the cache length %ld", g_T,
             g_cache_length);
    }
    if (g_T + max_new > g_cache_length) {
        fail(2, "prompt has %ld tokens; with max_new %ld it exceeds the "
                "cache length %ld",
             g_T, max_new, g_cache_length);
    }
    g_ids = (int32_t *)read_file(have_embeds_pair ? ids_path
                                                  : prompt_ids_path,
                                 (size_t)g_T * 4, "ids");
    g_padded_total = padded_total_for(g_T);
    if (have_embeds_pair) {
        size_t expected_embeds = (size_t)g_T * (size_t)g_hidden * 4;
        float *embeds = (float *)read_file(embeds_path, expected_embeds,
                                           "embeds");
        g_padded_embeds = (float *)calloc(
            (size_t)g_padded_total * (size_t)g_hidden, sizeof(float));
        if (!g_padded_embeds) {
            die("out of memory");
        }
        memcpy(g_padded_embeds, embeds, expected_embeds);
        free(embeds);
        if (g_have_ple) {
            die("--embeds/--ids cannot feed per-layer embeddings; use "
                "--audio-features/--prompt-ids");
        }
        if (dump_embeds_path) {
            dump_embeds_file(dump_embeds_path, g_padded_embeds,
                             g_T * g_hidden);
        }
    } else {
        // Audio features: float32 rows of the hidden size; the number of
        // feature rows must equal the number of --audio-pad-id prompt ids.
        struct stat st;
        if (stat(audio_features_path, &st) != 0) {
            fail(2, "stat('%s'): %s", audio_features_path, strerror(errno));
        }
        size_t row_bytes = (size_t)g_hidden * sizeof(float);
        if ((size_t)st.st_size % row_bytes != 0) {
            fail(2, "'%s' (audio-features) is %lld bytes; expected a "
                    "multiple of %zu (%ld x float32)",
                 audio_features_path, (long long)st.st_size, row_bytes,
                 g_hidden);
        }
        g_feat_rows = (long)(st.st_size / row_bytes);
        long n_pad = 0;
        for (long t = 0; t < g_T; t++) {
            if (g_ids[t] == g_audio_pad_id) {
                n_pad++;
            }
        }
        if (n_pad != g_feat_rows) {
            fail(2, "%ld prompt ids equal --audio-pad-id %d but the "
                    "audio-features file has %ld rows",
                 n_pad, (int)g_audio_pad_id, g_feat_rows);
        }
        g_features = (float *)read_file(audio_features_path,
                                        (size_t)g_feat_rows * row_bytes,
                                        "audio-features");
    }
    fprintf(stderr,
            APP ": T=%ld H=%ld cache_length=%ld buckets=%d first=%ld "
            "last=%ld padded=%ld\n",
            g_T, g_hidden, g_cache_length, g_num_buckets, g_buckets[0],
            g_buckets[g_num_buckets - 1], g_padded_total);

    allocate_buffers();
    bind_chains();

    // On-device assembly (the --embeds path loaded the embeddings instead):
    // build them from the prompt ids and the audio features.
    if (!have_embeds_pair) {
        float *assembled = assemble_embeds();
        g_padded_embeds = (float *)calloc(
            (size_t)g_padded_total * (size_t)g_hidden, sizeof(float));
        if (!g_padded_embeds) {
            die("out of memory");
        }
        memcpy(g_padded_embeds, assembled,
               (size_t)g_T * (size_t)g_hidden * sizeof(float));
        free(assembled);
        build_padded_ple();
        if (dump_embeds_path) {
            dump_embeds_file(dump_embeds_path, g_padded_embeds,
                             g_T * g_hidden);
        }
    }

    if (dump_path) {
        g_logits_dump = fopen(dump_path, "wb");
        if (!g_logits_dump) {
            fail(2, "cannot open '%s' for writing: %s", dump_path,
                 strerror(errno));
        }
    }

    long time_step = run_prefill();
    long max_steps = max_new < (g_cache_length - time_step)
                         ? max_new
                         : (g_cache_length - time_step);
    if (max_steps < 0) {
        max_steps = 0;
    }
    int32_t *tokens = (int32_t *)malloc(
        (size_t)(max_steps > 0 ? max_steps : 1) * sizeof(int32_t));
    if (!tokens) {
        die("out of memory");
    }
    int32_t next_token = g_ids[g_T - 1];  // oracle: next_decode_token
    run_decode(max_steps, time_step, next_token, tokens);

    if (g_logits_dump && fclose(g_logits_dump) != 0) {
        fail(1, "fclose on the logits dump: %s", strerror(errno));
    }

    printf("\nTOKENS");
    for (int i = 0; i < g_decode_tokens; i++) {
        printf(" %d", tokens[i]);
    }
    printf("\n");
    if (g_tail_steps > 0) {
        printf("prefill_ms=%.1f chunks=%d tail_steps=%d tail_ms=%.1f "
               "decode_tokens=%d decode_ms_per_tok=%.2f setup_ms=%.1f\n",
               g_prefill_ms, g_chunks, g_tail_steps, g_tail_ms,
               g_decode_tokens,
               g_decode_tokens > 0 ? g_decode_ms / (double)g_decode_tokens
                                    : 0.0,
               setup_ms);
    } else {
        printf("prefill_ms=%.1f chunks=%d decode_tokens=%d decode_ms_per_tok="
               "%.2f setup_ms=%.1f\n",
               g_prefill_ms, g_chunks, g_decode_tokens,
               g_decode_tokens > 0 ? g_decode_ms / (double)g_decode_tokens
                                    : 0.0,
               setup_ms);
    }
    if (g_profile) {
        print_profile();
    }
    free(tokens);
}

// ---------------------------------------------------------------------------
// main.
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    const char *model_dir = NULL, *embeds_path = NULL, *ids_path = NULL;
    const char *audio_features_path = NULL, *prompt_ids_path = NULL;
    const char *dump_path = NULL;
    const char *dump_embeds_path = NULL;
    const char *accel_arg = "cpu";
    const char *eos_arg = "151643,151645";
    long max_new = 64;
    long cache_len_arg = 0;
    int have_cache_len_arg = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--model-dir") == 0) {
            if (i + 1 >= argc) {
                die_usage("--model-dir needs a directory argument");
            }
            model_dir = argv[++i];
        } else if (strcmp(a, "--embeds") == 0) {
            if (i + 1 >= argc) {
                die_usage("--embeds needs a file argument");
            }
            embeds_path = argv[++i];
        } else if (strcmp(a, "--ids") == 0) {
            if (i + 1 >= argc) {
                die_usage("--ids needs a file argument");
            }
            ids_path = argv[++i];
        } else if (strcmp(a, "--audio-features") == 0) {
            if (i + 1 >= argc) {
                die_usage("--audio-features needs a file argument");
            }
            audio_features_path = argv[++i];
        } else if (strcmp(a, "--prompt-ids") == 0) {
            if (i + 1 >= argc) {
                die_usage("--prompt-ids needs a file argument");
            }
            prompt_ids_path = argv[++i];
        } else if (strcmp(a, "--audio-pad-id") == 0) {
            if (i + 1 >= argc) {
                die_usage("--audio-pad-id needs a numeric argument");
            }
            char *end = NULL;
            long v = strtol(argv[++i], &end, 10);
            if (!end || *end != '\0' || v < INT32_MIN || v > INT32_MAX) {
                die_usage("--audio-pad-id must be an integer");
            }
            g_audio_pad_id = (int32_t)v;
        } else if (strcmp(a, "--dump-embeds") == 0) {
            if (i + 1 >= argc) {
                die_usage("--dump-embeds needs a file argument");
            }
            dump_embeds_path = argv[++i];
        } else if (strcmp(a, "--max-new") == 0) {
            if (i + 1 >= argc) {
                die_usage("--max-new needs a numeric argument");
            }
            char *end = NULL;
            max_new = strtol(argv[++i], &end, 10);
            if (!end || *end != '\0' || max_new < 1) {
                die_usage("--max-new must be a positive integer");
            }
        } else if (strcmp(a, "--cache-length") == 0) {
            if (i + 1 >= argc) {
                die_usage("--cache-length needs a numeric argument");
            }
            char *end = NULL;
            cache_len_arg = strtol(argv[++i], &end, 10);
            if (!end || *end != '\0' || cache_len_arg < 1) {
                die_usage("--cache-length must be a positive integer");
            }
            have_cache_len_arg = 1;
        } else if (strcmp(a, "--eos") == 0) {
            if (i + 1 >= argc) {
                die_usage("--eos needs a comma-separated argument");
            }
            eos_arg = argv[++i];
        } else if (strcmp(a, "--accel") == 0) {
            if (i + 1 >= argc) {
                die_usage("--accel needs an argument");
            }
            accel_arg = argv[++i];
            if (strcmp(accel_arg, "cpu") != 0 &&
                strcmp(accel_arg, "npu") != 0) {
                die_usage("--accel must be cpu or npu");
            }
        } else if (strcmp(a, "--dump-logits") == 0) {
            if (i + 1 >= argc) {
                die_usage("--dump-logits needs a file argument");
            }
            dump_path = argv[++i];
        } else if (strcmp(a, "--profile") == 0) {
            g_profile = 1;
        } else if (strcmp(a, "--server") == 0) {
            g_server = 1;
        } else if (a[0] == '-' && a[1] != '\0') {
            die_usage("unknown option");
        } else {
            die_usage("unexpected positional argument");
        }
    }
    g_max_new = max_new;
    if (!model_dir) {
        die_usage("missing --model-dir");
    }
    int have_embeds_pair = (embeds_path != NULL) && (ids_path != NULL);
    int have_feat_pair = (audio_features_path != NULL) &&
                         (prompt_ids_path != NULL);
    if (g_server) {
        if (embeds_path != NULL || ids_path != NULL) {
            die_usage("--server reads each request from stdin; do not give "
                      "--embeds/--ids");
        }
        if (audio_features_path != NULL || prompt_ids_path != NULL) {
            die_usage("--server reads each request from stdin; do not give "
                      "--audio-features/--prompt-ids");
        }
        if (dump_path != NULL) {
            die_usage("--server takes the logits dump path per request; do "
                      "not give --dump-logits");
        }
        if (dump_embeds_path != NULL) {
            die_usage("--server does not take --dump-embeds");
        }
    }
    if (embeds_path != NULL && ids_path == NULL) {
        die_usage("--embeds requires --ids");
    }
    if (ids_path != NULL && embeds_path == NULL) {
        die_usage("--ids requires --embeds");
    }
    if (audio_features_path != NULL && prompt_ids_path == NULL) {
        die_usage("--audio-features requires --prompt-ids");
    }
    if (prompt_ids_path != NULL && audio_features_path == NULL) {
        die_usage("--prompt-ids requires --audio-features");
    }
    if (!g_server && have_embeds_pair == have_feat_pair) {
        die_usage("give exactly one of --embeds F --ids F or "
                  "--audio-features F --prompt-ids F");
    }

    // EOS list.
    {
        char *eos_copy = strdup(eos_arg);
        if (!eos_copy) {
            die("out of memory");
        }
        char *save = NULL;
        for (char *tok = strtok_r(eos_copy, ",", &save); tok != NULL;
             tok = strtok_r(NULL, ",", &save)) {
            char *end = NULL;
            long v = strtol(tok, &end, 10);
            if (!end || *end != '\0' || g_num_eos >= 16) {
                die_usage("--eos must be a comma-separated integer list");
            }
            g_eos[g_num_eos++] = (int32_t)v;
        }
        free(eos_copy);
    }

    LiteRtHwAcceleratorSet accel =
        strcmp(accel_arg, "cpu") == 0 ? kLiteRtHwAcceleratorCpu
                                      : kLiteRtHwAcceleratorNpu;

    const char *dispatch_dir = getenv("DISPATCH_DIR");
    char exe_buf[4096];
    if (!dispatch_dir || dispatch_dir[0] == '\0') {
        exe_dir(exe_buf, sizeof exe_buf);
        dispatch_dir = exe_buf;
    }
    fprintf(stderr, APP ": dispatch_dir=%s accel=%s\n", dispatch_dir,
            accel_arg);

    char pd_paths[2][1024], ed_path[1024], au_path[1024], ple_path[1024];
    static char chunk_paths[MAX_CHAINS * MAX_CHUNKS][1024];
    int num_chunk_files = 0;
    {
        DIR *d = opendir(model_dir);
        if (!d) {
            fail(2, "cannot open model dir '%s': %s", model_dir, strerror(errno));
        }
        struct dirent *e;
        while ((e = readdir(d)) != NULL) {
            if (strstr(e->d_name, "prefill_decode") && chunk_index_of(e->d_name) >= 0) {
                if (num_chunk_files >= MAX_CHAINS * MAX_CHUNKS) {
                    die("too many chunk files");
                }
                snprintf(chunk_paths[num_chunk_files++], 1024, "%s/%s", model_dir, e->d_name);
            }
        }
        closedir(d);
    }
    int num_pd = num_chunk_files ? 0 : find_model_files(model_dir, "prefill_decode", pd_paths, 2);
    find_model_file_x(model_dir, "embedder", "per_layer", 0, ed_path,
                      sizeof ed_path);
    g_have_ple = find_model_file_x(model_dir, "per_layer_embedder", NULL, 1,
                                   ple_path, sizeof ple_path);
    find_model_file(model_dir, "aux", au_path, sizeof au_path);

    double setup_start = now_ms();
    LiteRtAny dir_value;
    memset(&dir_value, 0, sizeof dir_value);
    dir_value.type = kLiteRtAnyTypeString;
    dir_value.str_value = dispatch_dir;
    LiteRtEnvOption env_options[2];
    env_options[0].tag = kLiteRtEnvOptionTagCompilerPluginLibraryDir;
    env_options[0].value = dir_value;
    env_options[1].tag = kLiteRtEnvOptionTagDispatchLibraryDir;
    env_options[1].value = dir_value;
    CHECK_LITERT(LiteRtCreateEnvironment(2, env_options, &g_env));

    LiteRtOptions options = NULL;
    CHECK_LITERT(LiteRtCreateOptions(&options));
    CHECK_LITERT(LiteRtSetOptionsHardwareAccelerators(options, accel));

    if (num_chunk_files) {
        g_chain_seq = getenv("ASR_CHAIN_SEQUENTIAL") && getenv("ASR_CHAIN_SEQUENTIAL")[0] == '1';
        for (int i = 0; i < num_chunk_files; i++) {
            const char *base = strrchr(chunk_paths[i], '/');
            load_chunk(chunk_paths[i], chunk_index_of(base ? base + 1 : chunk_paths[i]), options);
        }
        build_chain_section();
    } else {
        load_section(SEC_PD, pd_paths[0], "prefill_decode", options);
    }
    if (num_pd == 2) {
        load_section(SEC_PD2, pd_paths[1], "prefill_decode(2)", options);
        g_num_sections = 4;
    }
    load_section(SEC_ED, ed_path, "embedder", options);
    load_section(SEC_AU, au_path, "aux", options);
    if (g_have_ple) {
        load_section(SEC_PLE, ple_path, "per_layer_embedder", options);
    }
    double setup_ms = now_ms() - setup_start;
    g_setup_ms = setup_ms;

    discover();
    if (have_cache_len_arg && cache_len_arg != g_cache_length) {
        fail(2, "--cache-length %ld does not match the model cache length "
                "%ld",
             cache_len_arg, g_cache_length);
    }

    if (g_server) {
        // Setup is done once: allocate the buffers and snapshot the KV
        // cache (the fresh-process starting state every request resets to).
        allocate_buffers();
        bind_chains();
        snapshot_caches();
        printf("READY setup_ms=%.1f\n", setup_ms);
        fflush(stdout);
        serve_requests();
    } else {
        run_oneshot(setup_ms, have_embeds_pair, embeds_path, ids_path,
                    audio_features_path, prompt_ids_path, dump_embeds_path,
                    dump_path, max_new);
    }

    // Cleanup.
    for (int si = 0; si < NUM_SEC_SLOTS; si++) {
        Section *sec = &g_sections[si];
        if (!sec->loaded) {
            continue;
        }
        for (size_t k = 0; k < sec->num_sigs; k++) {
            Sig *sig = &sec->sigs[k];
            for (size_t i = 0; i < sig->num_in; i++) {
                if (sig->in[i].owns_buf) {
                    LiteRtDestroyTensorBuffer(sig->in[i].buf);
                }
                ttype_free(&sig->in[i].t);
            }
            for (size_t i = 0; i < sig->num_out; i++) {
                if (sig->out[i].owns_buf) {
                    LiteRtDestroyTensorBuffer(sig->out[i].buf);
                }
                free(sig->out[i].f32);
                ttype_free(&sig->out[i].t);
            }
            free(sig->in);
            free(sig->out);
            free(sig->in_bufs);
            free(sig->out_bufs);
        }
        free(sec->sigs);
        if (sec->compiled) {
            LiteRtDestroyCompiledModel(sec->compiled);
        }
        if (sec->model) {
            LiteRtDestroyModel(sec->model);
        }
    }
    for (int i = 0; i < g_num_caches; i++) {
        if (g_caches[i].buf) {
            LiteRtDestroyTensorBuffer(g_caches[i].buf);
        }
        ttype_free(&g_caches[i].t);
    }
    free(g_logits_scratch);
    free(g_padded_embeds);
    free(g_ids);
    free(g_features);
    free(g_step_ms);
    for (int i = 0; i < g_num_caches; i++) {
        free(g_cache_snap[i]);
    }
    LiteRtDestroyOptions(options);
    LiteRtDestroyEnvironment(g_env);
    return 0;
}
