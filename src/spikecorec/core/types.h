#pragma once

#include <any>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <ratio>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace spikecorec {

    using String = std::string;
    using String_View = std::string_view;
    using Any = std::any;

    using f32 = float;
    using f64 = double;
    using s32 = int32_t;
    using s64 = int64_t;
    using u8 = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;
    using usize = size_t;

    template <typename Key_Type, typename Value_Type>
    using Unordered_Map = std::unordered_map<Key_Type, Value_Type>;

    template <typename Key_Type, typename Value_Type>
    using Ordered_Map = std::map<Key_Type, Value_Type>;

    template <typename Some_Type>
    using Vector = std::vector<Some_Type>;

    template <typename Some_Type>
    using Set = std::unordered_set<Some_Type>;

    template <typename Some_Type>
    using Ordered_Set = std::set<Some_Type>;

    template <typename Some_Type>
    using Deque = std::deque<Some_Type>;

    template <typename Some_Type>
    using Optional = std::optional<Some_Type>;

    template <typename Some_Type>
    using Stack = std::stack<Some_Type>;

    template <typename First_Type, typename Second_Type>
    using Pair = std::pair<First_Type, Second_Type>;

    template <typename... Some_Types>
    using Tuple = std::tuple<Some_Types...>;

    template <typename Some_Type>
    using Unique_Pointer = std::unique_ptr<Some_Type>;

    template <typename Some_Type>
    using Shared_Pointer = std::shared_ptr<Some_Type>;

    template <typename Signature>
    using Function = std::function<Signature>;

    template <typename Some_Type>
    using Atomic = std::atomic<Some_Type>;

    template <typename Some_Type>
    using Lock_Guard = std::lock_guard<Some_Type>;

    template <typename Some_Type>
    using Unique_Lock = std::unique_lock<Some_Type>;

    template <typename Some_Type>
    using Numeric_Limits = std::numeric_limits<Some_Type>;

    template <bool Condition, typename True_Type, typename False_Type>
    using Conditional = std::conditional<Condition, True_Type, False_Type>;

    template <typename Character_Type>
    using Input_Stream_Buffer_Iterator = std::istreambuf_iterator<Character_Type>;

    template <typename Representation, typename Period = std::ratio<1>>
    using Duration = std::chrono::duration<Representation, Period>;

    template <typename Real_Type = double>
    using Normal_Distribution = std::normal_distribution<Real_Type>;

    template <typename Integer_Type = int>
    using Uniform_Integer_Distribution = std::uniform_int_distribution<Integer_Type>;

    template <typename Real_Type = double>
    using Uniform_Real_Distribution = std::uniform_real_distribution<Real_Type>;

    template <typename Real_Type = double>
    using Exponential_Distribution = std::exponential_distribution<Real_Type>;

    template <typename Integer_Type = int>
    using Poisson_Distribution = std::poisson_distribution<Integer_Type>;

    using Bernoulli_Distribution = std::bernoulli_distribution;
    using Mersenne_Twister = std::mt19937;
    using Mersenne_Twister_64 = std::mt19937_64;
    using Random_Device = std::random_device;

    using Path = std::filesystem::path;
    using Directory_Iterator = std::filesystem::directory_iterator;

    using Exception = std::exception;
    using Exception_Pointer = std::exception_ptr;
    using Runtime_Error = std::runtime_error;
    using Invalid_Argument = std::invalid_argument;
    using Error_Code = std::error_code;

    using Stream_Base = std::ios;
    using Output_Stream = std::ostream;
    using Output_String_Stream = std::ostringstream;
    using Input_String_Stream = std::istringstream;
    using String_Stream = std::stringstream;
    using Input_File_Stream = std::ifstream;
    using Output_File_Stream = std::ofstream;

    using Mutex = std::mutex;
    using Condition_Variable = std::condition_variable;
    using Thread = std::thread;
    using Once_Flag = std::once_flag;
    using Launch = std::launch;

    using Steady_Clock = std::chrono::steady_clock;
    using System_Clock = std::chrono::system_clock;
    using Milliseconds = std::chrono::milliseconds;
    using Seconds = std::chrono::seconds;

    struct alignas(16) float4 {
        f32 x, y, z, w;
    };


} // namespace spikecorec
