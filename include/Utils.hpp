#ifndef UTILSSSS
#define UTILSSSS

#include <AIToolbox/Types.hpp>

#define PARTICLEQUERY suggestQueryThompson
// #define PARTICLEQUERY suggestQueryThompsonMean
// #define PARTICLEQUERY suggestQueryMean
// #define PARTICLEQUERY suggestQueryMeanTWeight

void initParticles(AIToolbox::Matrix2D & particles);
std::string vectorToCSV(const AIToolbox::Vector & v);
double computeRegret(const std::vector<AIToolbox::Vector> &mu, const AIToolbox::Vector &weights, const AIToolbox::Vector &counters);

namespace AIToolbox {
    //                     weight, means,    covs
    using GMM = std::tuple<Vector, Matrix2D, Matrix3D>;

    using ContinuousState = Vector;
    using CState = ContinuousState;

    struct identity {
        template< class T>
        constexpr T&& operator()( T&& t ) const noexcept {
            return std::forward<T>(t);
        }
        using is_transparent = void;
    };

    template <typename M2D, typename M2DorV>
    auto makeDist(const Eigen::MatrixBase<M2D> & x1, const Eigen::MatrixBase<M2DorV> & x2) {
        if constexpr(M2DorV::ColsAtCompileTime == 1) {
            // If second argument is in vector form, we can simplify the
            // computation (we also assume that x2 is not in x1).
            return (Vector) (x1.rowwise() - x2.transpose()).rowwise().squaredNorm();
        } else {
            // Else, we compute the pairwise distances between the points of
            // both matrices, with no assumptions.
            Matrix2D retval = ((-2.0 * (x1 * x2.transpose())).colwise() + x1.rowwise().squaredNorm()).rowwise() + x2.rowwise().squaredNorm().transpose();

            // Make sure all elements are positive
            retval = retval.cwiseMax(0.0);

            // Output is (x1.rows(), x2.rows())
            return (Matrix2D) retval;
        }
    }

    template <typename V, typename Gen, typename P = identity>
    size_t sampleUnnormalizedProbability(const V & v, const double totalWeight, Gen & rnd, P proj = P{}) {
        std::uniform_real_distribution<double> sliceDist(0.0, totalWeight);
        double prob = sliceDist(rnd);

        size_t id = 0;
        for (; id < static_cast<size_t>(v.size()) - 1; ++id) {
            prob -= std::invoke(proj, v[id]);
            if (prob <= 0.0)
                break;
        }
        return id;
    }
}

#endif
