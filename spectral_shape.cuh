#pragma once
#include <Eigen/Dense>
#include <string>
#include <optional>
#include <cstddef>

class SpectralShape {
public:
    using Real   = double;
    using Vector = Eigen::VectorXd;

    SpectralShape(Real start, Real end, Real interval);
    ~SpectralShape();

    // Copy / move
    SpectralShape(const SpectralShape& other);
    SpectralShape& operator=(const SpectralShape& other);
    SpectralShape(SpectralShape&& other) noexcept;
    SpectralShape& operator=(SpectralShape&& other) noexcept;

    // Getters / setters
    Real start()    const noexcept { return _start;    }
    Real end()      const noexcept { return _end;      }
    Real interval() const noexcept { return _interval; }

    void start(Real v);
    void end(Real v);
    void interval(Real v);

    std::pair<Real, Real> boundaries() const noexcept { return {_start, _end}; }
    void boundaries(const std::pair<Real, Real>& b);

    // Range / wavelengths (host copy of device buffer)
    const Vector& range() const;
    const Vector& wavelengths() const { return range(); }

    // Iterator-like helpers (host side)
    const double* cbegin() const { return range().data(); }
    const double* cend()   const { return range().data() + range().size(); }

    // Contains
    bool contains(Real wavelength) const;
    bool contains(const Vector& wavelengths) const;

    // Size
    std::size_t size() const;

    // Comparison
    bool operator==(const SpectralShape& other) const;
    bool operator!=(const SpectralShape& other) const { return !(*this == other); }

    // String representations
    std::string str()  const;
    std::string repr() const;

    // Hash
    std::size_t hash() const;

private:
    Real _start     = 0.0;
    Real _end       = 0.0;
    Real _interval  = 1.0;

    int    _count   = 0;
    double* _d_wavelengths = nullptr;  // device buffer

    // Host cache
    mutable std::optional<Vector> _h_cache;
    mutable Real _cached_start    = 0.0;
    mutable Real _cached_end      = 0.0;
    mutable Real _cached_interval = 0.0;

    void allocate_and_fill_on_device();
    void free_device();
    void regenerate_if_needed() const;

    // CUDA helpers
    static bool cuda_contains_scalar(const double* d_data, int n,
                                     double value, double eps_digits10);
    static bool cuda_contains_vector(const double* d_data, int n,
                                     const double* d_query, int m,
                                     double eps_digits10);
};

// std::hash specialization
namespace std {
    template<>
    struct hash<SpectralShape> {
        std::size_t operator()(const SpectralShape& s) const noexcept {
            return s.hash();
        }
    };
}

