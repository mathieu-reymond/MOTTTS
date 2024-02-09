#include <MOExperience.hpp>

#include <algorithm>

namespace AIToolbox::Bandit {
    MOExperience::MOExperience(const size_t A, const size_t W) : q_(A, W), M2s_(A, W), counts_(A), timesteps_(0) {
        q_.setZero();
        M2s_.setZero();
    }

    void MOExperience::record(size_t a, const Vector & rews) {
        ++timesteps_;

        // Count update
        ++counts_[a];

        for (size_t o = 0; o < static_cast<size_t>(rews.size()); ++o) {
            const auto delta = rews[o] - q_(a, o);
            // Rolling average for this bandit arm
            q_(a, o) += delta / counts_[a];
            // Rolling sum of square diffs.
            M2s_(a, o) += delta * (rews[o] - q_(a, o));
        }
    }

    void MOExperience::reset() {
        q_.setZero();
        M2s_.setZero();
        std::fill(std::begin(counts_), std::end(counts_), 0);
        timesteps_ = 0;
    }

    unsigned long MOExperience::getTimesteps() const { return timesteps_; }
    size_t MOExperience::getA() const { return counts_.size(); }
    size_t MOExperience::getW() const { return q_.cols(); }
    const Matrix2D & MOExperience::getRewardMatrix() const { return q_; }
    const MOExperience::VisitsTable & MOExperience::getVisitsTable() const { return counts_; }
    const Matrix2D & MOExperience::getM2Matrix() const { return M2s_; }
}
