#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cupti.h>
#include <iostream>
#include <map>
#include <memory>
#include <spdlog/async.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <string>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

// device pre alloc memory
#define DEVICE_BUFFER_POOL_PRE_ALLOC_SIZE 3
#define DEVICE_BUFFER_PRE_ALLOC_SIZE 3200000

// 1s == 1000ms flush
#define CUPTI_FLUSH_TIME 1000

// host pre alloc memory 1mb * 10
#define CLIENT_BUFFER_SIZE static_cast<std::size_t>(1 * 1024 * 1024)
#define CLIENT_BUFFER_NUM 10
#define CLIENT_BUFFER_LIMIT 32

#define VTIMELINE_SUCCESS 0
#define VTIMELINE_ERROR 1

namespace {

using buffer_ptr = std::unique_ptr<uint8_t[]>;

std::mutex g_buffer_pool_mutex;
std::vector<buffer_ptr> g_buffer_pool(CLIENT_BUFFER_NUM);
std::size_t g_buffer_pool_cnt = CLIENT_BUFFER_NUM;

std::thread g_consume_buffer_from_cupti_task;
std::condition_variable g_client_buffer_notify;
std::mutex g_process_buffer_mutex;
std::vector<std::pair<buffer_ptr, size_t>> g_process_buffer;

bool g_need_to_stop = false;

bool g_is_init = false;
bool g_is_enable = false;

// ##################################//
//  prepare the name of dir and path //
// ##################################//
const char *_logger_dir = std::getenv("VTIMELINE_LOGGER_DIR");
const char *_rank = std::getenv("RANK");

const std::string g_logger_dir =
    _logger_dir == nullptr ? "/var/log" : std::string(_logger_dir);
const std::string g_rank = _rank == nullptr ? "-1" : std::string(_rank);

void init_spdlog_env() {
    const size_t max_file_size = static_cast<size_t>(200 * 1024 * 1024); // 200M
    const size_t max_files = 5;

    const size_t q_size = 102400;
    const size_t thread_count = 3;

    const uint64_t flush_time = 3;

    static std::mutex env_mutex;
    std::lock_guard<std::mutex> locker(env_mutex);

    std::string log_dir_name = g_logger_dir + "/CUPTI/";
    std::string log_file_name = log_dir_name + "/rank_" + g_rank + ".log";

    auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_file_name, max_file_size, max_files);

    std::vector<spdlog::sink_ptr> sinks{rotating_sink};

    spdlog::init_thread_pool(q_size, thread_count);

    auto logger = std::make_shared<spdlog::async_logger>(
        "cupti", sinks.begin(), sinks.end(), spdlog::thread_pool(),
        spdlog::async_overflow_policy::discard_new);

    spdlog::set_default_logger(logger);
    spdlog::set_pattern("%v");
    spdlog::flush_every(std::chrono::seconds(flush_time));
}

void CUPTIAPI requestBuffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *maxNumRecords = 0;

    {
        std::unique_lock<std::mutex> locker(g_buffer_pool_mutex);
        if (!g_buffer_pool.empty()) {
            *buffer = g_buffer_pool.back().release();
            *size = CLIENT_BUFFER_SIZE;
            g_buffer_pool.pop_back();
            return;
        }
    }

    if (g_buffer_pool_cnt >= CLIENT_BUFFER_LIMIT) {
        std::cerr << "Buffer pool limit reached, cannot allocate new buffer!\n";
        *buffer = nullptr;
        *size = 0;
        return;
    }

    // extend buffer
    std::cerr << "Allocating new buffer, pool count: " << g_buffer_pool_cnt << "\n";
    auto buffer_addr = std::make_unique<uint8_t[]>(CLIENT_BUFFER_SIZE);

    *buffer = buffer_addr.release();
    *size = CLIENT_BUFFER_SIZE;
    g_buffer_pool_cnt += 1;
}

// NOTE: the context and stream_id is NULL, be deprecated as of CUDA 6.0
void CUPTIAPI processBuffer(CUcontext context [[maybe_unused]],
                            uint32_t stream_id [[maybe_unused]], uint8_t *buffer,
                            size_t size, size_t valid_size) {
    if (size == 0 || buffer == nullptr) {
        return;
    }

    {
        std::unique_lock<std::mutex> locker(g_process_buffer_mutex);
        auto item = std::make_pair(buffer_ptr(buffer), valid_size);
        g_process_buffer.emplace_back(std::move(item));
    }

    g_client_buffer_notify.notify_one();
}

void consume_record_from_buffer(CUpti_Activity *record) {
    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_KERNEL: {
        CUpti_ActivityKernel9 *activity =
            reinterpret_cast<CUpti_ActivityKernel9 *>(record);
        uint64_t start_ts = activity->start;
        uint64_t end_ts = activity->end;

        uint32_t stream_id = activity->streamId;
        const std::string name(activity->name);

        // one rank for one device
        spdlog::info("{},{},{},KERNEL,{},B", start_ts, g_rank, stream_id, name);
        spdlog::info("{},{},{},KERNEL,{},E", end_ts, g_rank, stream_id, name);
        break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
        CUpti_ActivityMemcpy5 *activity =
            reinterpret_cast<CUpti_ActivityMemcpy5 *>(record);

        uint64_t start_ts = activity->start;
        uint64_t end_ts = activity->end;
        uint32_t stream_id = activity->streamId;

        uint8_t src_kind = activity->srcKind;
        uint8_t dst_kind = activity->dstKind;

        static std::map<uint8_t, const std::string> kind2name = {
            {CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN, "unknown"},
            {CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE, "pageable"},
            {CUPTI_ACTIVITY_MEMORY_KIND_PINNED, "pinned"},
            {CUPTI_ACTIVITY_MEMORY_KIND_DEVICE, "device"},
            {CUPTI_ACTIVITY_MEMORY_KIND_ARRAY, "array"},
            {CUPTI_ACTIVITY_MEMORY_KIND_MANAGED, "managed"},
            {CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC, "device_static"},
            {CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC, "managed_static"},
        };
        auto src_it = kind2name.find(src_kind);
        auto dst_it = kind2name.find(dst_kind);

        std::string name;

        if (src_it == kind2name.end() || dst_it == kind2name.end()) [[unlikely]] {
            name = "unknown";
        } else {
            name = src_it->second + "-" + dst_it->second;
        }

        spdlog::info("{},{},{},MEMCPY,{},B", start_ts, g_rank, stream_id, name);
        spdlog::info("{},{},{},MEMCPY,{},E", end_ts, g_rank, stream_id, name);
        break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
        CUpti_ActivityMemcpyPtoP4 *activity =
            reinterpret_cast<CUpti_ActivityMemcpyPtoP4 *>(record);

        uint64_t start_ts = activity->start;
        uint64_t end_ts = activity->end;
        uint32_t stream_id = activity->streamId;

        uint32_t src_device = activity->srcDeviceId;
        uint32_t dst_device = activity->dstDeviceId;

        spdlog::info("{},{},{},P2P,gpu{}-gpu{},B", start_ts, g_rank, stream_id, src_device,
                     dst_device);
        spdlog::info("{},{},{},P2P,gpu{}-gpu{},E", end_ts, g_rank, stream_id, src_device,
                     dst_device);
        break;
    }
    default:
        break;
    }
}

void consume_buffer_from_cupti() {
    while (true) {
        buffer_ptr buffer_addr;
        size_t valid_size = 0;

        {
            std::unique_lock<std::mutex> locker(g_process_buffer_mutex);
            if (g_process_buffer.empty()) {
                return;
            }
            auto buffer_item = std::move(g_process_buffer.back());
            g_process_buffer.pop_back();

            buffer_addr = std::move(buffer_item.first);
            valid_size = buffer_item.second;
        }

        if (valid_size > 0) {
            CUpti_Activity *record = nullptr;
            auto *buffer = buffer_addr.get();
            do {
                CUptiResult status =
                    cuptiActivityGetNextRecord(buffer, valid_size, &record);

                if (status == CUPTI_SUCCESS) {
                    consume_record_from_buffer(record);
                } else {
                    break;
                }
            } while (true);
        }

        {
            std::unique_lock<std::mutex> locker(g_buffer_pool_mutex);
            g_buffer_pool.emplace_back(std::move(buffer_addr));
        }
    }
}

void init_buffer_process_task() {
    // init the buffer pool
    std::generate_n(g_buffer_pool.begin(), CLIENT_BUFFER_NUM, []() {
        return std::make_unique<uint8_t[]>(CLIENT_BUFFER_SIZE);
    });

    g_need_to_stop = false;

    g_consume_buffer_from_cupti_task = std::move(std::thread([]() {
        while (true) {
            {
                std::unique_lock<std::mutex> locker(g_process_buffer_mutex);
                g_client_buffer_notify.wait(locker, []() {
                    return !g_process_buffer.empty() || g_need_to_stop;
                });
            }

            // process record
            consume_buffer_from_cupti();

            if (g_need_to_stop) {
                return;
            }
        }
    }));
}

bool init_cupti_env() {
    CUpti_SubscriberHandle handler = nullptr;
    CUptiResult result = cuptiSubscribe(&handler, nullptr, nullptr);
    if (result != CUPTI_SUCCESS) {
        std::cerr << "Failed to subscribe CUPTI, error code " << static_cast<int>(result)
                  << "\n";
        return false;
    }

    size_t attr_value_size = 0;

    // set device buffer pool limit == pre allocate value
    // set device buffer size, so peak cuda memory size is buffer size * buffer limit
    // set flush time
    // set callbacks to enable the function
    size_t buffer_pool_limit_size = DEVICE_BUFFER_POOL_PRE_ALLOC_SIZE;
    attr_value_size = sizeof(buffer_pool_limit_size);
    result = cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                                       &attr_value_size, (void *)&buffer_pool_limit_size);
    if (result != CUPTI_SUCCESS) {
        std::cerr << "Failed to set device buffer pool limit, error code: "
                  << static_cast<int>(result) << "\n";
        return false;
    }

    result =
        cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE,
                                  &attr_value_size, (void *)&buffer_pool_limit_size);
    if (result != CUPTI_SUCCESS) {
        std::cerr << "Failed to set device buffer pre-allocate value, error code: "
                  << static_cast<int>(result) << "\n";
        return false;
    }

    size_t buffer_size = DEVICE_BUFFER_PRE_ALLOC_SIZE;
    attr_value_size = sizeof(buffer_size);
    result = cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                                       &attr_value_size, (void *)&buffer_size);
    if (result != CUPTI_SUCCESS) {
        std::cerr << "Failed to set device buffer size, error code: "
                  << static_cast<int>(result) << "\n";
        return false;
    }

    result = cuptiActivityRegisterCallbacks(requestBuffer, processBuffer);
    if (result != CUPTI_SUCCESS) {
        std::cerr << "Failed to register activity callbacks, error code: "
                  << static_cast<int>(result) << "\n";
        return false;
    }

    return true;
}

void enable_cupti_activity() {
    std::array activity_kinds = {
        CUPTI_ACTIVITY_KIND_KERNEL,
        // CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_MEMCPY2,
    };
    for (const CUpti_ActivityKind kind : activity_kinds) {
        CUptiResult result = cuptiActivityEnable(kind);
        if (result != CUPTI_SUCCESS) {
            std::cerr << "Failed to enable activity: " << static_cast<int>(kind)
                      << ", error code: " << static_cast<int>(result) << "\n";
        }
    }
}

} // namespace

extern "C" int init_vtimeline(void) {
    if (g_is_init) {
        return VTIMELINE_SUCCESS;
    }

    ::init_spdlog_env();
    ::init_buffer_process_task();

    g_is_init = true;

    return CUPTI_SUCCESS;
}

extern "C" int enable_vtimeline(void) {
    if (!g_is_init) {
        return VTIMELINE_ERROR;
    }

    if (g_is_enable) {
        return VTIMELINE_SUCCESS;
    }

    if (!::init_cupti_env()) {
        return VTIMELINE_ERROR;
    }

    ::enable_cupti_activity();

    g_is_enable = true;

    return VTIMELINE_SUCCESS;
}

extern "C" int disable_vtimeline(void) {
    if (!g_is_init || !g_is_enable) {
        return VTIMELINE_ERROR;
    }

    CUptiResult result = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    if (result != CUPTI_SUCCESS) {
        std::cerr << "Failed to flush all activity, error code: "
                  << static_cast<int>(result) << "\n";
    }

    result = cuptiFinalize();
    if (result != CUPTI_SUCCESS) {
        std::cerr << "Failed to finalize cupti, error code: " << static_cast<int>(result)
                  << "\n";
    }

    g_is_enable = false;

    return CUPTI_SUCCESS;
}

extern "C" int deinit_vtimeline(void) {
    ::disable_vtimeline();

    g_need_to_stop = true;
    g_client_buffer_notify.notify_one();

    if (g_consume_buffer_from_cupti_task.joinable()) {
        g_consume_buffer_from_cupti_task.join();
    }

    return CUPTI_SUCCESS;
}