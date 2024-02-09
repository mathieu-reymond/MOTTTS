#ifndef STATISTICS_HEADER_FILE
#define STATISTICS_HEADER_FILE

#include <random>
#include <AIToolbox/Types.hpp>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
using namespace AIToolbox;

using Float = double;

std::pair<double, double> sampleNormalPosterior(double sampleMu, double sampleSumOfSquares, double n, std::mt19937 & rnd, double priorMu = 0.0, double priorN = 1e-8, double alpha = 0.5, double beta = 0.1);

double sampleMeanPosterior(double sampleMu, double sampleSumOfSquares, unsigned n, std::mt19937 & rnd);

void sampleMultivariateNormalInline(const Vector & mu, const Eigen::LLT<Matrix2D> & llt, Vector & out, std::mt19937 & rnd);
void sampleMultivariateNormalInline(const Vector & mu, const Matrix2D & cov, Vector & out, std::mt19937 & rnd);

void sampleNormalizedMultivariateNormalInline(const Vector & mu, const Eigen::LLT<Matrix2D> & cov, Vector & out, std::mt19937 & rnd);
void sampleNormalizedMultivariateNormalInline(const Vector & mu, const Matrix2D & cov, Vector & out, std::mt19937 & rnd);


/**
 * @brief This function samples an index from a probability vector.
 *
 * This function randomly samples an index between 0 and d, given a
 * vector containing the probabilities of sampling each of the indexes.
 *
 * For performance reasons this function does not verify that the input
 * container is effectively a probability.
 *
 * The generator has to be supplied to the function, so that different
 * objects are able to maintain different generators, to reduce correlations
 * between different samples. The generator has to be compatible with
 * std::uniform_real_distribution<Float>, since that is what is used
 * to obtain the random sample.
 *
 * @tparam T The type of the container vector to sample.
 * @tparam G The type of the generator used.
 * @param in The external probability container.
 * @param d The size of the supplied container.
 * @param generator The generator used to sample.
 *
 * @return An index in range [0,d-1].
 */
size_t sampleProbability(const Vector& in, std::mt19937& generator);

/**
 * @brief This function generates a random probability vector.
 *
 * This function will sample uniformly from the simplex space with the
 * specified number of dimensions.
 *
 * S must be at least one or we don't guarantee any behaviour.
 *
 * @param S The number of entries of the output vector.
 * @param generator A random number generator.
 *
 * @return A new random probability vector.
 */
Vector makeRandomProbability(const size_t S, std::mt19937 & generator);

#endif

