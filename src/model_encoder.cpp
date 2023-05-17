#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  
  queue q(gpu_selector_v);

  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  return 0;
}
