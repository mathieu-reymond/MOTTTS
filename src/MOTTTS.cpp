#include <MOTTTS.hpp>

#include <random>

namespace AIToolbox::Bandit {
    MOTopTwoThompsonSamplingPolicy::MOTopTwoThompsonSamplingPolicy(const MOExperience & exp, const Vector & weights, const double beta) :
            Base(exp.getA()), policy_(exp, weights), beta_(beta) {}

    size_t MOTopTwoThompsonSamplingPolicy::sampleAction() const {
        size_t bestAction = policy_.sampleAction();

        const auto & counts = policy_.getExperience().getVisitsTable();

        if (counts[bestAction] < 2) return bestAction;

        std::bernoulli_distribution pickBest(beta_);
        if (pickBest(rand_))
            return bestAction;

        size_t secondBestAction;
        do {
            secondBestAction = policy_.sampleAction();
        } while (bestAction == secondBestAction);

        return secondBestAction;
    }

    size_t MOTopTwoThompsonSamplingPolicy::recommendAction() const {
        const auto & exp = policy_.getExperience();

        size_t retval;
        (exp.getRewardMatrix() * policy_.getWeights()).maxCoeff(&retval);

        return retval;
    }

    double MOTopTwoThompsonSamplingPolicy::getActionProbability(const size_t & a) const {
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

    Vector MOTopTwoThompsonSamplingPolicy::getPolicy() const {
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

    const MOExperience & MOTopTwoThompsonSamplingPolicy::getExperience() const {
        return policy_.getExperience();
    }

    const Vector & MOTopTwoThompsonSamplingPolicy::getWeights() const { return policy_.getWeights(); }
}
