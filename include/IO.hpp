#ifndef IO_HEADER_FILE
#define IO_HEADER_FILE

#include <iosfwd>

#include <AIToolbox/Types.hpp>
using namespace AIToolbox;

// Weights, mus, sigmas
std::tuple<Vector, std::vector<Vector>, std::vector<Vector>> parseModelParameters(std::istream & input);

void writeModelParameters(std::ostream & os, const std::vector<Vector> &mu, const std::vector<Vector> &sigma, const Vector weights);

void writeExperiment(const std::string & methodname, size_t e, size_t experiments, unsigned count = 50);

std::ostream & operator<<(std::ostream & os, const std::vector<size_t> & v);
std::ostream & operator<<(std::ostream & os, const std::vector<Vector> & v);

#endif
