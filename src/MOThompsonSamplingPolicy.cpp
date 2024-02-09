#include <MOThompsonSamplingPolicy.hpp>

#include <random>

namespace AIToolbox::Bandit {
    MOThompsonSamplingPolicy::MOThompsonSamplingPolicy(const MOExperience & exp, const Vector & weights) :
            Base(exp.getA()), exp_(exp), weights_(weights) {}

    size_t MOThompsonSamplingPolicy::sampleAction() const {
        // For each arm, we sample its mean. Note that here we use a
        // standardized Student-t distribution, which we then scale using our
        // QFunction and counts parameters to obtain the correct mean
        // estimates.
        size_t bestAction = 0;
        double bestValue = std::numeric_limits<double>::min();

        const auto & counts = exp_.getVisitsTable();
        const auto & q = exp_.getRewardMatrix();
        const auto & m2 = exp_.getM2Matrix();

        for (size_t a = 0; a < A; ++a) {
            // We need at least 2 samples per arm with student-t to estimate
            // the variance.
            if (counts[a] < 2)
                return a;

            std::student_t_distribution<double> dist(counts[a] - 1);
            double val = 0.0;

            for (size_t o = 0; o < exp_.getW(); ++o) {
                //     mu = est_mu - t * s / sqrt(n)
                // where
                //     s^2 = sum_i (x_i - est_mu)^2 / (n-1)
                // and
                //     t = student_t sample with n-1 degrees of freedom
                val += weights_[o] * (q(a, o) + dist(rand_) * std::sqrt(m2(a, o) / (counts[a] * (counts[a] - 1))));
            }

            if (val > bestValue) {
                bestAction = a;
                bestValue = val;
            }
        }

        return bestAction;
    }

    double MOThompsonSamplingPolicy::getActionProbability(const size_t & a) const {
        // The true formula here is hard, so we don't compute this exactly.
        //
        // Instead we sample, which is easier and possibly faster if we just
        // want a rough approximation.
        constexpr unsigned trials = 1000;
        unsigned selected = 0;

        for (size_t i = 0; i < trials; ++i)
            if (sampleAction() == a)
                ++selected;

        return static_cast<double>(selected) / trials;
    }

    Vector MOThompsonSamplingPolicy::getPolicy() const {
        // The true formula here is hard, so we don't compute this exactly.
        //
        // Instead we sample, which is easier and possibly faster if we just
        // want a rough approximation.
        constexpr unsigned trials = 100000;

        Vector retval{A};
        retval.setZero();

        for (size_t i = 0; i < trials; ++i)
            retval[sampleAction()] += 1.0;

        retval /= retval.sum();
        return retval;
    }

    const MOExperience & MOThompsonSamplingPolicy::getExperience() const {
        return exp_;
    }

    const Vector & MOThompsonSamplingPolicy::getWeights() const { return weights_; }
}
