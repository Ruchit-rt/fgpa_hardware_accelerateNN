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

#endif