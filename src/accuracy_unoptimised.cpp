#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <fstream>
#include <limits>
#include "dpc_common.hpp"
#include <oneapi/dpl/random>
#include <oneapi/mkl.hpp>
#include <cinttypes>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace std::chrono;
using namespace std;
using namespace sycl;


// Number of iterations.
#define N 5

// Define a struct to store parameter information
struct ParameterInfo {
    std::string name;
    std::vector<int> shape;
};

// Function to load model parameters from a text file
std::vector<ParameterInfo> loadModelParameters(const std::string& filename, std::vector<float*>& paramArrays) {
    std::ifstream file(filename);
    std::vector<ParameterInfo> paramInfo;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return paramInfo;
    }

    std::string line;
    while (std::getline(file, line)) {
        ParameterInfo info;
        info.name = line;
        
        // Read the shape as a comma-separated list of integers
        std::getline(file, line);
        std::stringstream shapeStream(line);
        int shapeValue;
        char comma;
        while (shapeStream >> shapeValue) {
            info.shape.push_back(shapeValue);
            shapeStream >> comma; // Read the comma
        }
        
        // Read the flattened parameter values into a dynamic array
        std::vector<float> paramValues;
        while (std::getline(file, line) && !line.empty()) {
            std::stringstream valueStream(line);
            float paramValue;
            while (valueStream >> paramValue) {
                paramValues.push_back(paramValue);
            }
        }
        
        // Allocate memory for the parameter array and copy values
        if (!paramValues.empty()) {
            float* paramArray = new float[paramValues.size()];

            for (int i = 0; i < paramValues.size(); i++) {
              if (paramValues[i] > 10) {
                cout << "ERR value large: " << paramValues[i] << " for " << info.name << std::endl;
              }
            }

            std::memcpy(paramArray, paramValues.data(), paramValues.size() * sizeof(float));
            paramArrays.push_back(paramArray);
        }

        paramInfo.push_back(info);
    }

    file.close();
    return paramInfo;
}

template <std::size_t ID> class Conv1ID;
template <std::size_t ID> class Conv2ID;

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

void print_exec_time(event e, string name){
    auto start_time = e.template
            get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time = e.template
            get_profiling_info<sycl::info::event_profiling::command_end>();
    double dur = (end_time - start_time) / 1.0e9;

    cout <<  name << " event time: " << dur << std::endl;
}

/* Make a matrix of the given size and initialise all values to x. */
void mk_mat(queue &q, int row, int col, float x, float *result) {
  {
    buffer r_buf(result, range(row, col));
    q.submit([&](auto &h) {
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(row, col), [=](auto index) {
        r[index] = x;
      });
    });
  }
}

/* Make a matrix of the given size and initliase all values randomly between 0 and 1. */
void mk_mat_rand(queue &q, int row, int col, float *result) {
  {
    buffer r_buf(result, range(row, col));
    q.submit([&](auto &h) {
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(row, col), [=](auto index) {
        oneapi::dpl::minstd_rand engine(114, index.get_linear_id());
        oneapi::dpl::uniform_real_distribution<float> distr;
        r[index] = distr(engine);
      });
    });
  }
}

/* Show the matrix. If snapshot = true, only shwo the first 5 * 5 corner. */
void peek(int row, int col, float *matrix, bool snapshot) {
  for (int i = 0; i < (snapshot && row > 5 ? 5 : row); i++) {
    for (int j = 0; j < (snapshot && col > 5 ? 5 : col); j++) {
      cout << std::fixed << matrix[i * col + j] << "\t";
    }
    if (col > 5 && snapshot) {
      cout << "...";
    }
    cout << std::endl;
  }
  if (row > 5 && snapshot) {
    cout << "...";
  }
  cout << std::endl;
}

/* Carry out MaxPool on the given matrices (C * H * W) with the given stride. */
void max_pool(queue &q, int chn, int row, int col, float *matrix, int stride,
    int *new_row, int *new_col, float *result) {
  const int nr = 1 + (row - 1) / stride;
  const int nc = 1 + (col - 1) / stride;
  {
    buffer m_buf(matrix, range(chn, row, col));
    buffer r_buf(result, range(chn, nr, nc));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn, nr, nc), [=](auto index) {
        int _chn = index[0];
        int max_r = (index[1] + 1) * stride > row ? row : (index[1] + 1) * stride;
        int max_c = (index[2] + 1) * stride > col ? col : (index[2] + 1) * stride;
        float f = numeric_limits<float>::lowest();
        float cur;
        for (int i = index[1] * stride; i < max_r; i++) {
          for (int j = index[2] * stride; j < max_c; j++) {
            cur = m[_chn][i][j];
            f = f > cur ? f : cur;
          }
        }
        r[index] = f;
      });
    });
  }

  if (new_row) *new_row = nr;
  if (new_col) *new_col = nc;
}

void relu(queue &q, int length, float *matrix, float *result) {
  {
    buffer m_buf(matrix, range(length));
    buffer r_buf(result, range(length));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(length), [=](auto index) {
        r[index] = r[index] >= 0.0 ? r[index] : 0.0;
      });
    });
  }
}

/* Carry out convolution (no padding) between the tensor (B * C * W * H) and the filters (D * C * K * K).
   size must be an odd number and less than min(row, col). */
void conv(queue &q, int batch, int chn, int row, int col, float *tensor,
    int d, int size, float *filter, float *biases, int *new_row, int *new_col, float *result) {
  const int nr = row - size + 1;
  const int nc = col - size + 1;
  {
    buffer m_buf(tensor, range(batch * chn, row, col));
    buffer f_buf(filter, range(d * chn, size, size));
    buffer b_buf(biases, range(d));
    buffer r_buf(result, range(batch * d, nr, nc));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor f(f_buf, h, read_only);
      accessor b(b_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(batch * d, nr, nc), [=](auto index) {
        float sum = 0.0f;
        int _d = index[0] / batch;
        int _batch = index[0] / d;
        for (int c = 0; c < chn; c++) {
          for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
              sum += f[_d * chn + c][i][j] * m[_batch * chn + c][index[1] + i][index[2] + j];
            }
          }
        }
        r[index] = sum + b[_d];
      });
    });
  }

  if (new_row) *new_row = nr;
  if (new_col) *new_col = nc;
}

// /* Carry out convolution (with padding) between the tensor (B * C * W * H) and the filters (D * C * K * K).
//    size must be an odd number and less than min(row, col). */

// void conv_pad(queue &q, int batch, int chn, int row, int col, float *tensor,
//     int d, int size, float *filter, float *biases, float *result) {
//   {
//     // if (chn == 3) {
//     //   std::ofstream file("/home/u196631/urop/final-year-project/weights1_biases1_read.txt");
//     //   for (size_t j = 0; j < 1728; ++j) {
//     //     file << filter[j] << "\n";
//     //   }
//     //   file << "\n\n\n";
//     //   file << "Biases \n";

//     //   for (size_t j = 0; j < 64; ++j) {
//     //     file << biases[j] << "\n";
//     //   }
//     //   // Separate parameter sections with an empty line
//     //   file << "\n";
//     // }

//     // float *temp = new float[batch * d * row * col];

    // buffer m_buf(tensor, range(batch * chn * row * col));
    // buffer f_buf(filter, range(d * chn * size * size));
    // buffer b_buf(biases, range(d));
    // buffer r_buf(result, range(batch * d * row * col));
//     // buffer t_buf(temp, range(batch * d, row, col));
    
//     auto corner_event = q.submit([&](handler &h)
//     {
//       accessor tensor_d(m_buf, h, read_only);
//       accessor filter_d(f_buf, h, read_only);
//       accessor bias_d(b_buf, h, read_only);
//       accessor result_d(r_buf, h, write_only, no_init);
//       // accessor result_d(t_buf, h, write_only, no_init);

//       h.parallel_for(range(d), [=](auto index) {                
//             float sum = 0;
//             for (int c = 0; c < chn; c++) {
//                 int _fi = index * chn + c;
//                 sum += filter_d[_fi * 9 + 4] * tensor_d[c * size * size];
//                 sum += filter_d[_fi * 9 + 5] * tensor_d[c * size * size + 1];
//                 sum += filter_d[_fi * 9 + 7] * tensor_d[c * size * size + size];
//                 sum += filter_d[_fi * 9 + 8] * tensor_d[c * size * size + size + 1];
//             }

//             sum += bias_d[index];
//             result_d[index * size * size] = sum + bias_d[index];

//             sum = 0;
//             for (int c = 0; c < chn; c++) {
//                 int _fi = index * chn + c;
//                 sum += filter_d[_fi * 9 + 3] * tensor_d[c * size * size + size - 2];
//                 sum += filter_d[_fi * 9 + 4] * tensor_d[c * size * size + size - 1];
//                 sum += filter_d[_fi * 9 + 6] * tensor_d[c * size * size + size + size - 2];
//                 sum += filter_d[_fi * 9 + 7] * tensor_d[c * size * size + size + size - 1];
//             }

//             result_d[index * size *size + size - 1] = sum + bias_d[index];

//             sum = 0;
//             for (int c = 0; c < chn; c++) {
//                 int _fi = index * chn + c;
//                 sum += filter_d[_fi * 9 + 1] * tensor_d[c * size * size + (size - 2) * size];        
//                 sum += filter_d[_fi * 9 + 2] * tensor_d[c * size * size + (size - 2) * size + 1];        
//                 sum += filter_d[_fi * 9 + 4] * tensor_d[c * size * size + (size - 1) * size];        
//                 sum += filter_d[_fi * 9 + 5] * tensor_d[c * size * size + (size - 1) * size + 1];        
//             }

//             result_d[index * size * size + (size - 1) * size] = sum + bias_d[index];
            
//             sum = 0;
//             for (int c = 0; c < chn; c++) {
//                 int _fi = index * chn + c;
//                 sum += filter_d[_fi * 9] * tensor_d[c * size * size + (size - 2) * size + size - 2];
//                 sum += filter_d[_fi * 9 + 1] * tensor_d[c * size * size + (size - 2) * size + size - 1];
//                 sum += filter_d[_fi * 9 + 3] * tensor_d[c * size * size + (size - 1) * size + size - 2];
//                 sum += filter_d[_fi * 9 + 4] * tensor_d[c * size * size + (size - 1) * size + size - 1];
//             }

//             result_d[index * size * size + (size - 1) * size + size - 1] = sum + bias_d[index];
//         });
//     });

//     // Task for the boundary elements.
//     auto boundry_event = q.submit([&](handler &h) {
//       accessor tensor_d(m_buf, h, read_only);
//       accessor filter_d(f_buf, h, read_only);
//       accessor bias_d(b_buf, h, read_only);
//       accessor result_d(r_buf, h, write_only, no_init);
//       // accessor result_d(t_buf, h, write_only, no_init);

//       h.parallel_for(range(d * (size - 2)), [=](auto i) {
//                 int index[2];
//                 index[0] = (int) i / (size - 2);
//                 index[1] = (int) i % (size - 2);
//                 float sum1 = 0;

//                 // #pragma unroll
//                 for (int c = 0; c < chn; c++) {
//                     int _fi = index[0] * chn + c;
//                     #pragma unroll
//                     for (int i = 0; i <= 1; i++) {
//                         #pragma unroll
//                         for (int j = 0; j <= 2; j++) {
//                             sum1 += filter_d[_fi * 9 + (i+1) * 3 + j] * tensor_d[c * size * size + (i * size) + index[1] + j];
//                         }
//                     }
//                 }

//                 sum1 += bias_d[index[0]];
//                 result_d[index[0] * size * size + index[1] + 1] = sum1;

//                 float sum2 = 0;
//                 for (int c = 0; c < chn; c++) {
//                     int _fi = index[0] * chn + c;
//                     #pragma unroll
//                     for (int i = -2; i <= -1; i++) {
//                         #pragma unroll
//                         for (int j = 0; j <= 2; j++) {
//                             sum2 += filter_d[_fi * 9 + (i + 2) * 3 + j] * tensor_d[c * size * size + (size + i) * size + index[1] + j];      
//                         }
//                     }
//                 }

//                 sum2 += bias_d[index[0]];
//                 result_d[index[0] * size * size + (size - 1) * size + index[1] + 1] = sum2;

//                 float sum3 = 0;
//                 for (int c = 0; c < chn; c++) {
//                     int _fi = index[0] * chn + c;
//                     #pragma unroll
//                     for (int i = 0; i <= 2; i++) {
//                         #pragma unroll
//                         for (int j = 0; j <= 1; j++) {
//                             sum3 += filter_d[_fi * 9 + i * 3 + j + 1] * tensor_d[c * size * size + (index[1] + i) * size + j];        
//                         }
//                     }
//                 }

//                 sum3 += bias_d[index[0]];
//                 result_d[index[0] * size * size + (index[1] + 1) * size] = sum3;

//                 float sum4 = 0;
//                 for (int c = 0; c < chn; c++) {
//                     int _fi = index[0] * chn + c;
//                     #pragma unroll
//                     for (int i = 0; i <= 2; i++) {
//                         #pragma unroll
//                         for (int j = -2; j <= -1; j++) {
//                         sum4 += filter_d[_fi * 9 + i * 3 + j + 2] * tensor_d[c * size * size + (index[1] + i) * size + size + j];        
//                         }
//                     }
//                 }

//                 sum4 += bias_d[index[0]];
//                 result_d[index[0] * size * size + (index[1] + 1) * size + size - 1] = sum4;
//             });
//     });

//     // Task for interior elements (that uses all 3 * 3 filters).
//     auto interior_event = q.submit([&](handler &h) {
//       accessor tensor_d(m_buf, h, read_only);
//       accessor filter_d(f_buf, h, read_only);
//       accessor bias_d(b_buf, h, read_only);
//       accessor result_d(r_buf, h, write_only, no_init);
//       // accessor result_d(t_buf, h, write_only, no_init);

//       h.parallel_for(range((size - 2) * (size - 2)), [=](auto i) {
//                 // int index[3] = { i / ((size - 2) * (size - 2)), (i / (size -2)) % (size - 2), i % (size - 2) };
//                 int index[3];
//                 index[0] = (int) i / ((size - 2) * (size - 2));
//                 index[1] = (int) (i / (size -2)) % (size - 2);
//                 index[2] = (int) i % (size - 2);
//                 float sum = 0;
//                 #pragma unroll 2 // Partial unrolling for the outermost loop.
//                 for (int c = 0; c < chn; c++) {
//                     int _fi = index[0] * chn + c;
//                     for (int i = 0; i <= 2; i++) {
//                         for (int j = 0; j <= 2; j++) {
//                             sum += filter_d[index[0] * chn + c * 9 + i * 3 + j] * tensor_d[c * size * size + (index[1] + i) * size + index[2] + j];        
//                         }
//                     }
//                 }

//                 sum += bias_d[index[0]];
//                 result_d[index[0] * size * size + (index[1] + 1) * size + index[2] + 1] = sum;
//         });
//     });

//     corner_event.wait();
//     boundry_event.wait();
//     interior_event.wait();
//     q.wait();

//     // cout << "Result of Conv" << " are: "
//     // << result[0] << " " <<result[100] << " "<< result[2500] << std::endl;

//     // if (chn == 3) {
//     //   std::ofstream file("/home/u196631/urop/final-year-project/temp_buffer_read.txt");
//     //   for (size_t j = 0; j < batch * d * row * col; ++j) {
//     //     file << temp[j] << "\n";
//     //   }
//     //   // Separate parameter sections with an empty line
//     //   file << "\n";
//     // }

//   }
// }

// back version of original
// void conv_pad(queue &q, int batch, int chn, int row, int col, float *tensor,
//     int d, int size, float *filter, float *biases, float *result) {
//   {
//     // // if (chn == 3) {
//     // //   std::ofstream file("/home/u196631/urop/final-year-project/weights1_biases1_read.txt");
//     // //   for (size_t j = 0; j < 1728; ++j) {
//     // //     file << filter[j] << "\n";
//     // //   }
//     // //   file << "\n\n\n";
//     // //   file << "Biases \n";

//     // //   for (size_t j = 0; j < 64; ++j) {
//     // //     file << biases[j] << "\n";
//     // //   }
//     // //   // Separate parameter sections with an empty line
//     // //   file << "\n";
//     // // }

//     // float *temp = new float[batch * d * row * col];

//     buffer m_buf(tensor, range(batch * chn, row, col));
//     buffer f_buf(filter, range(d * chn, size, size));
//     buffer b_buf(biases, range(d));
//     buffer r_buf(result, range(batch * d, row, col));
//     // buffer t_buf(temp, range(batch * d, row, col));

//     q.submit([&](auto &h) {
//       accessor m(m_buf, h, read_only);
//       accessor f(f_buf, h, read_only);
//       accessor b(b_buf, h, read_only);
//       accessor r(r_buf, h, write_only, no_init);
//       // accessor t(t_buf, h, write_only, no_init);

//       h.parallel_for(range(batch * d, row, col), [=](auto index) {
//         float sum = 0.0f;
//         int _d = index[0] / batch;
//         int _batch = index[0] / d;
//         for (int c = 0; c < chn; c++) {
//           for (int i = - size / 2; i <= size / 2; i++) {
//             for (int j = - size / 2; j <= size / 2; j++) {
//               if (index[1] + i < row && index[2] + j < col) {
//                 sum += f[_d * chn + c][i + size / 2][j + size / 2] * m[_batch * chn + c][index[1] + i][index[2] + j];
//               }
//             }
//           }
//         }
//         r[index] = sum + b[_d];
//         t[index] = sum + b[_d];
//       });
//     });

//     q.wait();

//     cout << "Result of Conv" << " are: "
//     << result[0] << " " <<result[100] << " "<< result[2500] << std::endl;
    
//     cout << "Temp buffer in Conv" << " are: "
//     << temp[0] << " " << temp[100] << " "<< temp[2500] << std::endl;

//     if (chn == 3) {
//       std::ofstream file("/home/u196631/urop/final-year-project/temp_buffer_read.txt");
//       for (size_t j = 0; j < batch * d * row * col; ++j) {
//         file << temp[j] << "\n";
//       }
//       // Separate parameter sections with an empty line
//       file << "\n";
//     }

//   }
// }

void conv_pad(queue &q, int batch, int chn, int row, int col, float *tensor,
    int d, int size, float *filter, float *biases, float *result) {
  {
    // float *temp = new float[2];
    // temp[0] = -1;
    // temp[1] = -1;

    buffer m_buf(tensor, range<3>(chn, row, col));
    buffer f_buf(filter, range<3>(d, chn, size * size));
    buffer b_buf(biases, range<1>(d));
    buffer r_buf(result, range<3>(d, row , col));
    // buffer t_buf(temp, range(2));

    q.submit([&](handler& h) {
      accessor m(m_buf, h, read_only);
      accessor f(f_buf, h, read_only);
      accessor b(b_buf, h, read_only);
      accessor r(r_buf, h, write_only, no_init);
      // accessor t(t_buf, h, read_write, no_init);

    h.parallel_for(range<3>(d, row, col), [=](auto item) {
      // int _b = item[0];
      int _d = item[0];
      // int idx = item[1];
      int _row = item[1];
      int _col = item[2];

      float sum = 0.0f;

      for (int c = 0; c < chn; c++) {
        // kernel size is 3
        for (int i = -1; i <= 1; i++) {
          for (int j = -1; j <= 1; j++) {
            int row_idx = _row + i;
            int col_idx = _col + j;
            // Check bounds and compute the convolution
            if (row_idx >= 0 && row_idx < row && col_idx >= 0 && col_idx < col) {
                // Use linear indices for m_buf and f_buf
                // int m_idx = ((_b * chn + c) * row * col) + (row_idx * col + col_idx);
                // int f_idx = ((_d * chn + c) * size * size) + ((i + 1) * size + j + 1);
                
                // if ((m_idx < (batch * chn * row * col)) && (f_idx < (d * chn * size * size))) {
                float pos = m[c][row_idx][col_idx] * f[_d][c][((i + 1) * size + j + 1)];
                if (pos < 10) {
                  sum += pos; 
                }
            }
          }
        }
    }
    
    // int r_idx = (_b * d * row * col) + (_d * row * col) + idx;
    r[_d][_row][_col] = sum + b[_d];
    });
  });

  q.wait();

  // cout << "Result of Conv" << " are: "
  // << result[0] << " " <<result[100] << " "<< result[2500] << std::endl;
  // cout << "FLAG: " << temp[0] << "   " << temp[1] << std::endl;
  }
}


/* Carry out the calculation for a fully-connected layer. */
void fully_connected(queue &q, int c_in, int c_out, float *vector,
    float *weights, float *bias, float *result) {
  {
    buffer v_buf(vector, range(c_in));
    buffer w_buf(weights, range(c_out, c_in));
    buffer b_buf(bias, range(c_out));
    buffer r_buf(result, range(c_out));
    q.submit([&](auto &h) {
      accessor v(v_buf, h, read_only);
      accessor w(w_buf, h, read_only);
      accessor b(b_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(c_out), [=](auto index) {
        float sum = 0.0f;
        for (int i = 0; i < c_in; i++) {
          sum += v[i] * w[index][i];
        }
        r[index] = sum + b[index];
      });
    });
  }
}

void l2_distance(queue &q, int chn, int length, float *matrices, int d, float *prototypes, float *result) {
  {
    buffer m_buf(matrices, range(chn, length));
    buffer p_buf(prototypes, range(d, chn));
    buffer r_buf(result, range(d, length));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor p(p_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(d, length), [=](auto index) {
        float sum = 0.0f;
        for (int c = 0; c < chn; c++) {
          sum += (m[c][index[1]] - p[index[0]][c]) * (m[c][index[1]] - p[index[0]][c]);
        }
        r[index] = sqrt(sum);
      });
    });
  }
}

void distance_2_similarity(queue &q, int length, float *vector, float *result) {
  {
    buffer v_buf(vector, range(length));
    buffer r_buf(result, range(length));
    q.submit([&](auto &h) {
      accessor v(v_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(length), [=](auto index) {
        r[index] = log((v[index] + 1) / (v[index] + 0.0001));
      });
    });
  }
}

void top9_average_pooling(queue &q, int chn, int length, float *tensor_ptr, float *result_ptr)
{
    buffer m_buf(tensor_ptr, range(chn * length));
    buffer r_buf(result_ptr, range(chn));
    auto top9_event = q.submit([&](auto &h) {

        accessor m(m_buf, h, read_only);
        accessor r(r_buf, h);
        h.parallel_for(range(chn), [=](auto index) {
            float r_[9];

            // set up sorted list using called r_ for each channel
            for (int i = 0; i < 9; i++) {
                r_[i] = m[index * length + i];
            }

            for (int i = 9; i < length; i++) {
                // if tensor_d[index * length + i] is more than min in the heap
                // insertion needs to happen
                int min_index = 0;
                float min_value = 10000.0;
                bool match = false;
                for (int j = 0; j < 9; j++) {
                        if (r_[j] < min_value) {
                            min_value = r_[j];
                            min_index = j;
                        }
                }
                if (m[index * length + i] > min_value) r_[min_index] = m[index * length + i];
            }
            // for (int u = 0; u < 9; u++) {
            //     temp_buffer[u] = r_[u];
            // }
            r[index] = (r_[0] + r_[1] + r_[2] + r_[3] + r_[4] + r_[5] + r_[6] + r_[7] + r_[8]) / 9;
        });
    });
    top9_event.wait();
    // print_exec_time(top9_event, "Top 9");
    // cout << "Top 9 result: ";
    // for (int i = 0; i < 15; i++){
    //     cout << result_ptr[i] << " ";
    // }
    // cout << std::endl;

    // cout << "Top 9 buffer: ";
    // for (int i = 0; i < 9; i++){
    //     cout << temp_arr[i] << " ";
    // }
    // cout << std::endl;
    // free(temp_arr, q);

    // return result_ptr;
}

void top9(queue &q, int chn, int length, float *tensor, float *result) {
  {
    buffer m_buf(tensor, range(chn, length));
    buffer r_buf(result, range(chn, 9));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h);

      h.parallel_for(range(chn), [=](auto index) {
        for (int i = 0; i < 9; i++) {
          r[index][i] = m[index][i];
          int k = i;
          while (true) {
            if (r[index][k] >= r[index][k / 2]) break;
            float temp = r[index][k];
            r[index][k] = r[index][k / 2];
            r[index][k / 2] = temp;
            k /= 2;
          }
        }

        for (int i = 9; i < length; i++) {
          if (m[index][i] > r[index][0]) {
            r[index][0] = m[index][i];
            int k = 0;
            while (k < 9) {
              if (k >= 4) break;

              if (r[index][k] > r[index][2 * k + 1] || r[index][k] > r[index][2 * k + 2]) {
                float temp = r[index][k];
                if (r[index][2 * k + 1] < r[index][2 * k + 2]) {
                  r[index][k] = r[index][2 * k + 1];
                  r[index][2 * k + 1] = temp;
                  k = 2 * k + 1;
                } else {
                  r[index][k] = r[index][2 * k + 2];
                  r[index][2 * k + 2] = temp;
                  k = 2 * k + 2;
                }
              } else {
                break;
              }
            }
          }
        }
      });
    });
  }
}

void bottom9(queue &q, int chn, int length, float *tensor, float *result) {
  {
    buffer m_buf(tensor, range(chn, length));
    buffer r_buf(result, range(chn, 9));
    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h);

      h.parallel_for(range(chn), [=](auto index) {
        for (int i = 0; i < 9; i++) {
          r[index][i] = m[index][i];
          int k = i;
          while (true) {
            if (r[index][k] <= r[index][k / 2]) break;
            float temp = r[index][k];
            r[index][k] = r[index][k / 2];
            r[index][k / 2] = temp;
            k /= 2;
          }
        }

        for (int i = 9; i < length; i++) {
          if (m[index][i] < r[index][0]) {
            r[index][0] = m[index][i];
            int k = 0;
            while (k < 9) {
              if (k >= 4) break;

              if (r[index][k] < r[index][2 * k + 1] || r[index][k] < r[index][2 * k + 2]) {
                float temp = r[index][k];
                if (r[index][2 * k + 1] > r[index][2 * k + 2]) {
                  r[index][k] = r[index][2 * k + 1];
                  r[index][2 * k + 1] = temp;
                  k = 2 * k + 1;
                } else {
                  r[index][k] = r[index][2 * k + 2];
                  r[index][2 * k + 2] = temp;
                  k = 2 * k + 2;
                }
              } else {
                break;
              }
            }
          }
        }
      });
    });
  }
}

void average(queue &q, int chn, int length, float *tensor, float *result) {
  {
    buffer m_buf(tensor, range(chn, length));
    buffer r_buf(result, range(chn));

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn), [=](auto index) {
        float sum = 0;
        for (int i = 0; i < length; i++) {
          sum += m[index][i];
        }
        r[index] = sum / length;
      });
    });
  }
}

void upsample4(queue &q, int chn, int row, int col, float *matrix, float *result) {
  {
    buffer m_buf(matrix, range(chn, row, col));
    buffer r_buf(result, range(chn, row * 4, col * 4));

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn), [=](auto index) {
        auto _r = 2 * (index[1] % 8) + 1;
        auto _q = (index[1] + 2) / 4;
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++){
            r[index][i][j] = m[index][0][0];
            r[index][i][4 * col - 1 - j] = m[index][0][col - 1];
            r[index][4 * row - 1 - i][j] = m[index][row - 1][0];
            r[index][4 * row - 1 - i][4 * col - 1 - j] = m[index][row - 1][col - 1];
          }
        }
      });
    });

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn, col * 4 - 4), [=](auto index) {
        auto _r = 2 * (index[1] % 8) + 1;
        auto _q = (index[1] + 2) / 4;
        for (int i = 0; i < 2; i++) {
          r[index[0]][i][index[1] + 2] = (m[index[0]][0][_q] * (8 - _r) + m[index[0]][0][_q + 1] * _r) / 8;
          r[index[0]][4 * row - 1 - i][index[1] + 2] = (m[index[0]][row - 1][_q] * (8 - _r) + m[index[0]][row - 1][_q + 1] * _r) / 8;
          }
      });
    });

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn, row * 4 - 4), [=](auto index) {
        auto _r = 2 * (index[1] % 8) + 1;
        auto _q = (index[1] + 2) / 4;
        for (int i = 0; i < 2; i++) {
          r[index[0]][index[1] + 2][i] = (m[index[0]][_q][0] * (8 - _r) + m[index[0]][_q + 1][0] * _r) / 8;
          r[index[0]][index[1] + 2][4 * col - 1 - i] = (m[index[0]][_q][col - 1] * (8 - _r) + m[index[0]][_q + 1][col - 1] * _r) / 8;
        }
      });
    });

    q.submit([&](auto &h) {
      accessor m(m_buf, h, read_only);
      accessor r(r_buf, h, write_only);

      h.parallel_for(range(chn, row * 4 - 4, col * 4 - 4), [=](auto index) {
        auto _r1 = 2 * (index[1] % 4) + 1;
        auto _q1 = index[1] / 4;
        auto _r2 = 2 * (index[2] % 4) + 1;
        auto _q2 = index[2] / 4;
        r[index[0]][index[1] + 2][index[2] + 2]
            = (m[index[0]][_q1][_q2] * (8 - _r1) + m[index[0]][_q1 + 1][_q2] * _r1) * (8 - _r2) / 64
            + (m[index[0]][_q1][_q2 + 1] * (8 - _r1) + m[index[0]][_q1 + 1][_q2 + 1] * _r1) * _r2 / 64;
      });
    });
  }
}

float *read_param(ifstream &rf) {
  int len;
  rf.read((char *)(&len), 4);
  float *result = new float[len];
  rf.read((char *)result, len * 4);
  cout << "Read " << len << " params" << std::endl;
  return result;
}

int main()
{
    cout.precision(4);

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#elif CPU
    auto selector = cpu_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
    property_list queue_properties{sycl::property::queue::enable_profiling()};
    queue q = sycl::queue(selector, exception_handler, queue_properties);

    auto device = q.get_device();

    if (!device.has(sycl::aspect::usm_device_allocations)) {
      std::cerr << "This design must either target a board that supports USM "
                   "Host/Shared allocations, or IP Component Authoring. "
                << std::endl;
      std::terminate();
    }
    if (!device.get_info<info::device::usm_host_allocations>()) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations\n";
      std::terminate();
    }

    // The file that encodes all parameters of the model.
    // ifstream rf("/home/u196631/urop/final-year-project/data/model_params.mmzk", ios::binary);
    // if(!rf) {
    //     cout << "Cannot open file!" << std::endl;
    //     return 1;
    // }

    // Get parameters
    // float *weights1 = read_param(rf);
    // float *biases1 = read_param(rf);
    // float *weights2 = read_param(rf);
    // float *biases2 = read_param(rf);
    // float *prototypes = read_param(rf);
    // float *fc_weights = read_param(rf);
        // rf.close();
    float *weights1 = (float *)malloc(1728 * sizeof(float));
    float *biases1 = (float *)malloc(64 * sizeof(float));
    float *weights2 = (float *)malloc(294912 * sizeof(float));
    float *biases2 = (float *)malloc(512 * sizeof(float));
    float *prototypes = (float *)malloc(7680 * sizeof(float));
    float *fc_weights = (float *)malloc(45 * sizeof(float));


    float fc_biases[3] = {0.0, 0.0, 0.0};

    /* Loading params from model_parameters.txt file */
    std::string filename = "/home/u196631/urop/final-year-project/model_parameters.txt";

    // Create a vector to store the parameter arrays.
    std::vector<float*> paramArrays;

    // Load model parameters from the file and store them in paramArrays.
    std::vector<ParameterInfo> paramInfo = loadModelParameters(filename, paramArrays);

    // Access and use the loaded parameters and their shapes as needed.
    cout << "Num params read: " << paramInfo.size() << std::endl;
    float **params = new float*[6];
    params[0] = prototypes;
    params[1] = weights1;
    params[2] = biases1;
    params[3] = weights2;
    params[4] = biases2;
    params[5] = fc_weights;

    int param_ptr = 0;
    for (size_t i = 0; i < paramInfo.size(); ++i) {
        // std::cout << "Parameter Name: " << paramInfo[i].name << std::endl;
        // std::cout << "Parameter Shape: ";
        int count = 1;
        for (int dim : paramInfo[i].shape) {
            // std::cout << dim << " ";
            count = count * dim;
        }
        // std::cout << std::endl;

        // Access the parameter array using paramArrays[i]
        memcpy(params[param_ptr], paramArrays[i], count * sizeof(float));
        param_ptr++;
    }

    // std::ofstream file("/home/u196631/urop/final-year-project/model_parameters_read.txt");

    // if (!file.is_open()) {
    //     std::cerr << "Error: Could not open file " << filename << std::endl;
    //     return -1;
    // }

    // for (size_t i = 0; i < paramInfo.size(); ++i) {
    //     // Write parameter name
    //     file << paramInfo[i].name << "\n";

    //     // Write parameter shape as comma-separated values
    //     int dim = 1;
    //     for (size_t j = 0; j < paramInfo[i].shape.size(); ++j) {
    //         file << paramInfo[i].shape[j];
    //         dim = dim * paramInfo[i].shape[j];
    //         if (j < paramInfo[i].shape.size() - 1) {
    //             file << ",";
    //         }
    //     }
    //     file << "\n";

    //     for (size_t j = 0; j < dim; ++j) {
    //         file << params[i][j] << "\n";
    //     }

    //     // Separate parameter sections with an empty line
    //     file << "\n";
    // }

    // file.close();

    // memset(weights1, 0, 1728 * sizeof(float));
    // memset(biases1, 0, 64 * sizeof(float));
    // memset(weights2, 0, 294912 * sizeof(float));
    // memset(biases2, 0, 512 * sizeof(float));
    // memset(prototypes, 0, NUM * sizeof(float));
    // memset(fc_weights, 0, NUM * sizeof(float));


    for (float* paramArray : paramArrays) {
        delete[] paramArray;
    }


    float *conved1 = new float[64 * 224 * 224];
    float *pooled1 = new float[64 * 112 * 112];
    float *conved2 = new float[512 * 112 * 112];
    float *pooled2 = new float[512 * 56 * 56];
    float *distances = new float[15 * 56 * 56];
    float *similarities = new float[15 * 56 * 56];
    float *top_similarities = new float[15 * 9];
    float *avg = new float[15];
    float *logits = new float[3];

    cout << "Model parameters file opened successfully" << std::endl;

    
    /* going through csv files from test_images_csv */
    string _csv_path = "/home/u196631/urop/final-year-project/test_images_csv/tomato/";
    int acc_count = 0;

    for (int i = 981; i <= 1400; i++){
      // memset(conved1, 0, 64 * 224 * 224 * sizeof(float));

      float *logits_f = new float[3];
      float *upsampled_f = new float[15 * 224 * 224];

      string name;
      if (i < 1000) name = _csv_path + "0" + std::to_string(i) +"_arr.csv";
      else name = _csv_path + std::to_string(i) +"_arr.csv";

      std::ifstream inputFile(name);
      cout << name << std::endl;
      std::vector<float> values;
      std::string line, value;

      if (!inputFile.is_open()) {
          std::cout << "Failed to open the " + name + " file." << std::endl;
          return -1;
      }

      while (std::getline(inputFile, line)) {
          std::istringstream iss(line);
          while (std::getline(iss, value, ',')) {
          float floatValue = std::stof(value);
          values.push_back(floatValue);
          }
      }
      inputFile.close();

      // Convert vector to float array
      float* input_ff = new float[values.size()];
      std::copy(values.begin(), values.end(), input_ff);

      // cout << "Input " << i << " are: "
      //     << input_ff[0] << " " << input_ff[1] << " "<< input_ff[2] << std::endl;

      conv_pad(q, 1, 3, 224, 224, input_ff, 64, 3, params[1], params[2], conved1);

      cout << "Conved 1 after" << " are: "
    << conved1[0] << " " << conved1[100] << " "<< conved1[2500] << std::endl;

      // std::ofstream outputFile("conved1_fixed.txt");
      // // Check if the file opened successfully
      // if (!outputFile.is_open()) {
      //     std::cerr << "Error: Could not open output file." << std::endl;
      //     return 1;
      // }

      // // Iterate through the data and write to the file
      // for (int index = 0; index < (64*224*224); ++index) {
      //     outputFile << conved1[index] << std::endl;
      // }

      // // Close the output file stream
      // outputFile.close();

      // std::cout << "Data has been written to crm_features.txt" << std::endl;


      relu(q, 64 * 224 * 224, conved1, conved1);

    //   cout << "Relu 1" << " are: "
    // << conved1[0] << " " << conved1[100] << " "<< conved1[1000] << std::endl;



      max_pool(q, 64, 224, 224, conved1, 2, NULL, NULL, pooled1);

      cout << "Pooled 1" << " are: "
    << pooled1[0] << " " << pooled1[100] << " "<< pooled1[1000] << std::endl;

      conv_pad(q, 1, 64, 112, 112, pooled1, 512, 3, weights2, biases2, conved2);

              cout << "Conved 2 after" << " are: "
    << conved2[0] << " " << conved2[100] << " "<< conved2[2500] << std::endl;

      relu(q, 512 * 112 * 112, conved2, conved2);
      max_pool(q, 512, 112, 112, conved2, 2, NULL, NULL, pooled2);

      // std::ofstream outputFile("crm_features.txt");

      // // Check if the file opened successfully
      // if (!outputFile.is_open()) {
      //     std::cerr << "Error: Could not open output file." << std::endl;
      //     return 1;
      // }

      // // Iterate through the data and write to the file
      // for (int index = 0; index < (56*56*512); ++index) {
      //     outputFile << pooled2[index] << std::endl;
      // }

      // // Close the output file stream
      // outputFile.close();

      // std::cout << "Data has been written to crm_features.txt" << std::endl;

      cout << "Pooled 2" << " are: "
          << pooled2[0] << " " << pooled2[100] << " "<< pooled2[1000] << std::endl;

      l2_distance(q, 512, 56 * 56, pooled2, 15, prototypes, distances);

      cout << "distances" << " are: "
          << distances[0] << " " << distances[100] << " "<< distances[1000] << std::endl;

      distance_2_similarity(q, 15 * 56 * 56, distances, similarities);

      cout << "Similarites are" << " are: "
          << similarities[0] << " " << similarities[100] << " "<< similarities[1000] << std::endl;
      top9_average_pooling(q, 15, 56 * 56, similarities, avg);
      // top9(q, 15, 56 * 56, similarities, top_similarities);
      // average(q, 15, 9, top_similarities, avg);
      // cout << "top9 averaging done" << std::endl;
      fully_connected(q, 15, 3, avg, fc_weights, fc_biases, logits);
      // cout << "fc done" << std::endl;
      // bottom9(q, 15, 56 * 56, distances, top_similarities);
      // average(q, 15, 9, top_similarities, avg);
      // upsample4(q, 15, 56, 56, similarities, conved1);
      // cout << "unsampling done" << std::endl;
      
      // carrot 65%
      // cabbage 22%
      // tomato 82%
      // index of the chosen classification. 0 For cabbage; 1 for carrot; 2 for tomato.
      int logit = 0;
      cout << "Logit " << i << " are: "
          << logits[0] << " " << logits[1] << " "<< logits[2] << std::endl;
      for (int i = 0; i < 3; i++) {
          if (logits[i] > logits[logit]) {
              logit = i;
          }
      }
      cout << "Classification: " << logit << std::endl;
      if (logit == 2) {
          acc_count++;
      }
      cout << acc_count << std::endl << std::endl;
      delete[] input_ff;
    }
    // cout << "Cabbage ac: " << cabbage_acc << std::endl;
    float acc = ( (float) acc_count / 420) * 100.00;
    cout << "Accuracy was: " << acc << " % " <<std::endl;
    
    free(weights1);
    free(biases1);
    free(weights2);
    free(biases2);
    free(fc_weights);
    free(prototypes);

    delete[] conved1;
    delete[] pooled1;
    delete[] conved2;
    delete[] pooled2;
    delete[] distances;
    delete[] similarities;
    delete[] top_similarities;
    delete[] logits;

    return (int) acc;
}