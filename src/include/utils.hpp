#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <sycl/sycl.hpp>

using namespace sycl;

void exception_handler(sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
    }
  }
}

double print_exec_time(event e, string name){
    auto start_time = e.template
            get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time = e.template
            get_profiling_info<sycl::info::event_profiling::command_end>();
    double dur = (end_time - start_time) / 1.0e9;

    cout <<  name << " event time: " << dur << std::endl;
    
    return dur;
}
/* Show the matrix. If snapshot = true, only shwo the first 5 * 5 corner. */
void peek(int row, int col, float *matrix, bool snapshot)
{
    int peek_level = 16;
    for (int i = 0; i < (snapshot && row > peek_level ? peek_level : row); i++)
    {
        for (int j = 0; j < (snapshot && col > peek_level ? peek_level : col); j++)
        {
            cout << std::fixed << matrix[i * col + j] << "\t";
        }
        if (col > peek_level && snapshot)
        {
            cout << "...";
        }
        cout << std::endl;
    }
    if (row > peek_level && snapshot)
    {
        cout << "...";
    }
    cout << std::endl;
}

/* Show the matrix. If snapshot = true, only shwo the first 5 * 5 corner. */
void peek_int(int row, int col, int8_t *matrix, bool snapshot)
{
    for (int i = 0; i < (snapshot && row > 5 ? 5 : row); i++)
    {
        for (int j = 0; j < (snapshot && col > 5 ? 5 : col); j++)
        {
            cout << std::fixed << (int32_t)(matrix[i * col + j]) << "\t";
        }
        if (col > 5 && snapshot)
        {
            cout << "...";
        }
        cout << std::endl;
    }
    if (row > 5 && snapshot)
    {
        cout << "...";
    }
    cout << std::endl;
}


std::vector<int> findTopIndices(const  array) {
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> minHeap;

    for (int i = 0; i < array.size(); i++) {
        if (minHeap.size() < 9) {
            minHeap.push(std::make_pair(array[i], i));
        } else if (array[i] > minHeap.top().first) {
            minHeap.pop();
            minHeap.push(std::make_pair(array[i], i));
        }
    }

    std::vector<int> topIndices;
    while (!minHeap.empty()) {
        topIndices.push_back(minHeap.top().second);
        minHeap.pop();
    }

    return topIndices;
}


namespace {
template <typename Func, std::size_t Index>
    class SubmitOnePipeWrite {
    public:
    SubmitOnePipeWrite(Func &&f) {
        f(std::integral_constant<std::size_t, Index>());
    }
};

template <typename Func, std::size_t... Indices>
inline constexpr void PipeWriteUnroller(Func &&f,
                                        std::index_sequence<Indices...>) {
(SubmitOnePipeWrite<Func, Indices>(f), ...); // fold expression
}
} //namespace

template <std::size_t N,
          typename Func>

constexpr void SubmitPipeWrites(Func &&f) {
  std::make_index_sequence<N> indices;
  PipeWriteUnroller<>(f, indices);
}

#endif