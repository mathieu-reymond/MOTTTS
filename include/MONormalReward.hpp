#ifndef MONORMALREWARD
#define MONORMALREWARD

#include <AIToolbox/Types.hpp>

class MONormalReward {
    public:
        MONormalReward(const AIToolbox::Vector & means, const AIToolbox::Vector & stds);

        template <typename Gen>
        const AIToolbox::Vector & operator()(Gen & gen) {
            for (size_t o = 0; o < dists_.size(); ++o)
                helper_[o] = dists_[o](gen);
            return helper_;
        }

    private:
        AIToolbox::Vector helper_;
        std::vector<std::normal_distribution<double>> dists_;
};

#endif
