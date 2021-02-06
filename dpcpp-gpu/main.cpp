#include <iostream>
#include <CL/sycl.hpp>
#include <chrono>
#include <inttypes.h>
#include <cmath>
#include <memory>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <numeric>

constexpr int32_t N{100000000};
constexpr int32_t x1{-4};
constexpr int32_t x2{6};
constexpr int32_t len{x2 - x1};
constexpr float dx{static_cast<double>(len) / N};
constexpr uint32_t kernelsNum{1000};
constexpr uint32_t oneKernelIts{N / kernelsNum};

constexpr float oneKernelRange{static_cast<float>(len) / kernelsNum};

static auto exceptionHandler = [](sycl::exception_list e_list)
{
    for (std::exception_ptr const &e : e_list)
    {
        try
        {
            std::rethrow_exception(e);
        }
        catch (cl::sycl::runtime_error& syclRuntimeError)
        {
            std::cout << syclRuntimeError.what() << std::endl;
            std::terminate();
        }
        catch (std::exception const &e)
        {
            std::cout << "Failure" << std::endl;
            std::terminate();
        }
    }
};

std::shared_ptr<sycl::device_selector> buildSelector(std::string selectorName)
{
    if (selectorName == "gpu")
    {
        return std::make_shared<sycl::gpu_selector>();
    }
    else if (selectorName == "cpu")
    {
        return std::make_shared<sycl::cpu_selector>();
    }
    else if (selectorName == "fpga")
    {
        return std::make_shared<sycl::INTEL::fpga_selector>();    
    }
    else
    {
        return std::make_shared<sycl::default_selector>();
    }
}

float fun(float x)
{
    return x * x * x - 3 * x * x - 18 * x + 70;
}

int main()
{
    std::shared_ptr<sycl::device_selector> deviceSelector{buildSelector("gpu")};
    sycl::queue q(*deviceSelector, exceptionHandler);

    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    sycl::range<1> rng{kernelsNum};

    auto begin = std::chrono::high_resolution_clock::now();

    float* results = sycl::malloc_shared<float>(kernelsNum, q);

    auto e = q.parallel_for(rng,
                            [=](auto i)
                            {
                                results[i] = 0.f;
                                float a, b, part_field;
                                const float kBegin = x1 + i * oneKernelRange;

                                for (uint32_t idx = 0; idx < oneKernelIts - 1; ++idx)
                                {
                                    a = fun(kBegin + idx * dx);
                                    b = fun(kBegin + (idx + 1) * dx);
                                    part_field = ((a + b) / 2) * dx;
                                    
                                    results[i] += part_field;
                                }
                            });
    e.wait();

    double sum = 0;

    for (uint32_t i = 0; i < kernelsNum; ++i)
    {
        sum += results[i];
        std::cout << results[i] << std::endl;
    }


    std::cout << "Integral value: " << sum << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::cout << "Time in seconds: " << duration / 1000.0 << std::endl;
    free(results, q);
    return 0;
}
