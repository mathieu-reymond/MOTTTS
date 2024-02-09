#ifndef MOBANDIT_NORMAL_POSTERIOR_HEADER_FILE
#define MOBANDIT_NORMAL_POSTERIOR_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils/StorageEigen.hpp>
using namespace AIToolbox;

class MOBanditNormalPosterior {
    public:
        MOBanditNormalPosterior(size_t A, size_t W, bool saveSamples = false);

        void record(size_t arm, const Vector & rew);

        std::pair<double, double> sampleMeanStd(size_t arm, size_t obj, std::mt19937 & rnd) const;
        double sampleMean(size_t arm, size_t obj, std::mt19937 & rnd) const;

        std::pair<double, double> maxLikelihoodMeanStd(size_t arm, size_t obj) const;

        size_t getA() const;
        size_t getW() const;

        const Matrix2D & getMeans() const;
        const Matrix2D & getSampleSquares() const;
        const std::vector<unsigned> & getCounts() const;

        const std::vector<StorageMatrix2D> & getSamples() const;

    private:
        Matrix2D sampleMu_;
        Matrix2D sampleSumOfSquares_;
        std::vector<unsigned> counts_;

        bool saveSamples_;
        std::vector<StorageMatrix2D> samples_;
};

#endif
