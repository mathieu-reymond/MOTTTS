#include <IO.hpp>

#include <iostream>
#include <unistd.h>

// Return 1 ==> OK
// Return 0 ==> EOF
bool readLineVector(std::istream & input, std::vector<double> & out) {
    out.clear();
    double v;

    // Sanity preservers
    size_t tokensRead = 0;
    constexpr size_t MAXTOKENS = 10000;

    while (tokensRead < MAXTOKENS) {
        // Remove space and tabs whitespaces.
        char peek = input.peek();
        while (peek == ' ' || peek == '\t') {
            input.get();
            peek = input.peek();
        }
        // Check end of line.
        if (peek == '\n') {
            input.get();
            return 1;
        }
        // Try parse.
        if (!(input >> v)) {
            // EOF
            if (input.eof()) return 0;
            // ERROR
            throw std::runtime_error("IO READLINEVECTOR READ FAIL");
        }
        // INPUT
        out.push_back(v);
    }
    if (tokensRead == MAXTOKENS)
        throw std::runtime_error("IO READLINEVECTOR INFINITE LOOP");
    return 1;
}

Vector copyVector(const std::vector<double> & in) {
    Vector out(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = in[i];
    return out;
}

std::tuple<Vector, std::vector<Vector>, std::vector<Vector>> parseModelParameters(std::istream & input) {
    std::cout << "\n## STARTING FILE PARSING ##\n\n";
    std::vector<double> inV;

    bool eof = !readLineVector(input, inV);

    if (inV.size() == 0)
        throw std::runtime_error("parseModel: could not read UF weights (0 found).");

    Vector weights = copyVector(inV);

    std::cout << "Utility function read:\n"
              << "- N Objectives: " << weights.size() << '\n'
              << "- Weights: " << weights.transpose() << '\n';

    if (weights.sum() != 1.0) {
        weights = weights.cwiseAbs() / weights.cwiseAbs().sum();
        std::cout << "- WARNING: weights do not sum to 1.0; normalizing.\n"
                  << "  New weights: " << weights.transpose() << '\n';
    }

    std::cout << '\n';

    // We log the error here to give as much feedback as possible.
    if (eof)
        throw std::runtime_error("parseModel: read weights, but file has already ended.");

    // Arms
    std::vector<Vector> mus, stds;

    while (true) {
        // Remove whitespace so we can skip some lines if we want.
        input >> std::ws;
        // Check whether we reached EOF
        if (input.eof()) break;

        // Weights and stds must be on adjacent lines tho
        if (!readLineVector(input, inV))
            throw std::runtime_error("parseModel: read means, but file has already ended (no stds).");
        mus.push_back(copyVector(inV));

        readLineVector(input, inV);
        stds.push_back(copyVector(inV));

        // Sanity check
        if (mus.back().size() != weights.size())
            throw std::runtime_error("parseModel: could not read correct number of Bandit means.");

        if (stds.back().size() != weights.size())
            throw std::runtime_error("parseModel: could not read correct number of Bandit stds.");
    }

    std::cout << "MO Bandit read:\n"
              << "- N Arms: " << mus.size() << '\n';

    for (size_t a = 0; a < mus.size(); ++a) {
        std::cout << "# Arm " << a << ":\n"
                  << mus[a].transpose() << '\n'
                  << stds[a].transpose() << '\n';
    }

    if (mus.size() < 2)
        throw std::runtime_error("Could only read a single arm, you probably don't want this!");


    return {weights, mus, stds};
}

void writeModelParameters(std::ostream & os, const std::vector<Vector> &mu, const std::vector<Vector> &sigma, const Vector weights) {
    os << weights.transpose() << "\n\n";

    for (size_t a = 0; a < mu.size(); ++a) {
        os << mu[a].transpose() << '\n';
        os << sigma[a].transpose() << "\n\n";
    }
}

std::ostream & operator<<(std::ostream & os, const std::vector<size_t> & v) {
    std::cout << '[';
    for (size_t i = 0; i < v.size() - 1; ++i)
        std::cout << v[i] << ", ";
    std::cout << v.back() << ']';
    return os;
}

std::ostream & operator<<(std::ostream & os, const std::vector<Vector> & v) {
    std::cout << '[';
    for (size_t i = 0; i < v.size() - 1; ++i)
        std::cout << v[i].transpose() << ", ";
    std::cout << v.back() << ']';
    return os;
}

void writeExperiment(const std::string & methodname, size_t e, size_t experiments, unsigned steps) {
    if (!isatty(fileno(stdout))){
        if (!(e%5000)) std::cout << "\nRunning " << methodname << ". Iteration: " << e << '/' << experiments << "    " << std::flush;
    } else {
        if (!(e%steps)) std::cout << "\rRunning " << methodname << ". Iteration: " << e << '/' << experiments << "    " << std::flush;
    }
}
