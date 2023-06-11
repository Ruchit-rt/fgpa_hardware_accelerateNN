#include <sycl/sycl.hpp>
#include <utility>

namespace {
template <typename Func, template <std::size_t> typename Name,
          std::size_t Index>
class SubmitOneComputeUnit {
public:
  SubmitOneComputeUnit(Func &&f, sycl::queue &q) {
    q.single_task<Name<Index>>([=] {
      static_assert(
          std::is_invocable_v<Func, std::integral_constant<std::size_t, Index>>,
          "The callable Func passed to SubmitComputeUnits must take a single "
          "argument of type auto");
      f(std::integral_constant<std::size_t, Index>());
    });
  }
};

template <typename Func, template <std::size_t, std::size_t> typename Name, std::size_t P,
          std::size_t Index>
class SubmitOneComputeUnit_N {
public:
  SubmitOneComputeUnit_N(Func &&f, sycl::queue &q) {
    q.single_task<Name<P,Index>>([=] {
      static_assert(
          std::is_invocable_v<Func, std::integral_constant<std::size_t, Index>>,
          "The callable Func passed to SubmitComputeUnits must take a single "
          "argument of type auto");
      f(std::integral_constant<std::size_t, Index>());
    });
  }
};

template <template <std::size_t> typename Name, typename Func,
          std::size_t... Indices>
inline constexpr void ComputeUnitUnroller(sycl::queue &q, Func &&f,
                                          std::index_sequence<Indices...>) {
  (SubmitOneComputeUnit<Func, Name, Indices>(f, q), ...); // fold expression
}
template <template <std::size_t, std::size_t> typename Name, std::size_t P,typename Func,
          std::size_t... Indices>
inline constexpr void ComputeUnitUnroller_N(sycl::queue &q, Func &&f,
                                          std::index_sequence<Indices...>) {
  (SubmitOneComputeUnit_N<Func, Name, P, Indices>(f, q), ...); // fold expression
}
} // namespace

template <std::size_t N,                           // Number of compute units
          template <std::size_t ID> typename Name, // Name for the compute units
          typename Func>                           // Callable defining compute
                                                   // units' functionality

// Func must take a single argument. This argument is the compute unit's ID.
// The compute unit ID is a constexpr, and it can be used to specialize
// the kernel's functionality.
// Note: the type of Func's single argument must be 'auto', because Func
// will be called with various indices (i.e., the ID for each compute unit)
constexpr void SubmitComputeUnits(sycl::queue &q, Func &&f) {
  std::make_index_sequence<N> indices;
  ComputeUnitUnroller<Name>(q, f, indices);
}

template <std::size_t N,                           // Number of compute units
          template <std::size_t ID,std::size_t PID> typename Name,std::size_t P,// Name for the compute units
          typename Func>                           // Callable defining compute
                                                   // units'   functionality

// Func must take a single argument. This argument is the compute unit's ID.
// The compute unit ID is a constexpr, and it can be used to specialize
// the kernel's functionality.
// Note: the type of Func's single argument must be 'auto', because Func
// will be called with various indices (i.e., the ID for each compute unit)
constexpr void SubmitComputeUnits_N(sycl::queue &q, Func &&f) {
  std::make_index_sequence<N> indices;
  ComputeUnitUnroller_N<Name, P>(q, f, indices);
}