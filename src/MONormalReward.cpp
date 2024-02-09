#include <MONormalReward.hpp>

using namespace AIToolbox;

MONormalReward::MONormalReward(const Vector & means, const Vector & stds) : helper_(means.size()) {
    dists_.reserve(means.size());
    for (size_t i = 0; i < static_cast<size_t>(means.size()); ++i)
        dists_.emplace_back(means[i], stds[i]);
}
