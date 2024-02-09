#ifndef UTILITY_FUNCTION_HEADER_FILE
#define UTILITY_FUNCTION_HEADER_FILE

#include <AIToolbox/Types.hpp>
using namespace AIToolbox;

// To decide whether to init something as a sample or as maximum likelihood.
enum class InitType {SAMPLE, ML};

class UtilityFunctionPosterior;

class UtilityFunction {
    public:
        UtilityFunction(Vector weights, double noise);
        UtilityFunction(const UtilityFunctionPosterior & posterior, double noise, InitType);

        void reset(const UtilityFunctionPosterior & posterior, InitType);

        double scalarizeReward(const Vector & rewards) const;
        bool evaluateDiff(const Vector & diff) const;

        const Vector & getWeights() const;

    private:
        Vector weights_;
        double noise_;
        mutable std::mt19937 rand_;
};

#endif
