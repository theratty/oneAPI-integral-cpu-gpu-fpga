#include <iostream>
#include <chrono>
#include <inttypes.h>
#include <cmath>

constexpr int32_t N{100000000};
constexpr int32_t x1{-4};
constexpr int32_t x2{6};
constexpr int32_t len{x2 - x1};
constexpr double dx{static_cast<double>(len) / N};

double fun(double x)
{
    return pow(x, 3) - 3 * pow(x, 2) - 18 * x + 70;
}

int main()
{
    double a, b, part_field, sum{0};

    auto begin = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < N - 1; ++i)
    {
        a = fun(x1 + i * dx);
        b = fun(x1 + (i + 1) * dx);
        part_field = ((a + b) / 2) * dx;
        
        sum += part_field;
    }

    std::cout << "Integral value: " << sum << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::cout << "Time in seconds: " << duration / 1000.0 << std::endl;

    return 0;
}
