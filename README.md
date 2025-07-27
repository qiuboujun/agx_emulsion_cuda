## Python Dependencies for Test Reference Scripts

The C++/CUDA tests rely on small Python helper scripts that use NumPy and SciPy to generate ground-truth reference data.  Install them once (preferably in a virtual environment):

```bash
python3 -m pip install -r requirements.txt  # installs numpy, scipy, colour-science
```

Afterwards you can run the whole validation suite via `make tests`. 

## Build & Test the C++/CUDA Port

Prerequisites:
1. NVIDIA GPU + CUDA Toolkit (tested with CUDA 11).
2. C++17-capable compiler (GCC ≥ 9).
3. Python 3 with packages listed in `requirements.txt`.

### One-time setup
```bash
# Install Python packages
python3 -m pip install -r requirements.txt
```

### Compile & run validation suite
```bash
# From repository root
make            # builds main demo app only
make tests      # builds all kernels + five unit tests and runs them
```
`make tests` performs the following:
1. Executes three Python helper scripts that generate reference data using NumPy / SciPy / Colour-Science.
2. Compiles CUDA sources (`gp_numpy.cu`, `gp_scipy.cu`, `gaussian_filter_kernel.cu`) plus utility C++ files.
3. Links five test executables:
   * `norm_test` – standard-normal PDF / CDF versus NumPy
   * `gaussian_test` – 2-D Gaussian blur versus SciPy   `ndimage.gaussian_filter`
   * `interp_test` – `interp1d` (linear & nearest) versus SciPy
   * `spline_test` – natural cubic-spline evaluation versus SciPy
   * `color_test` – linear colour-space conversions versus Colour-Science
4. Runs each executable; all must report **PASSED** with maximum absolute error ≤ the tolerances documented in the tests.

If you wish to clean everything:
```bash
make clean
```
This removes object files and test executables.

## Example: running the Gaussian blur kernel
```cpp
#include "kernels.h"
#include <Eigen/Dense>
#include <iostream>

int main(){
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> img(10,10);
    img.setRandom();
    auto blurred = gaussian_filter(img, 1.2);
    std::cout << blurred << std::endl;
}
```
Compile with:
```bash
g++ -std=c++17 -I. -I/usr/include/eigen3 demo.cpp gaussian_filter_kernel.cu \
    -lcudart -lcufft -o demo
```

## Extending
* Additional colour-spaces can be added in `color_models.cpp` by filling new matrices.
* Further SciPy replacements should follow the pattern: CUDA kernel in `.cu`, thin wrapper in header, Python reference + C++ test.

--- 