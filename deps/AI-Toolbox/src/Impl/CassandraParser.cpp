#include <AIToolbox/Impl/CassandraParser.hpp>

#include <numeric>
#include <istream>
#include <algorithm>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

namespace AIToolbox::Impl {
    CassandraParser::CassandraParser() {
        // Assign an action to parse each value for the preambles. Lines parsed
        // in the preamble are parsed before the others.
        initMap_["values"] = [](const std::string &){};
        initMap_["states"] = [this](const std::string & line){
            S_ = extractIDs(line, stateMap_);
        };
        initMap_["actions"] = [this](const std::string & line){
            A_ = extractIDs(line, actionMap_);
        };
        initMap_["observations"] = [this](const std::string & line){
            O_ = extractIDs(line, observationMap_);
        };
        initMap_["discount"] = [this](const std::string & line){
            discount_ = std::stod(tokenize(line, ":").at(1));
        };
    }

    CassandraParser::MDPVals CassandraParser::parseMDP(std::istream & input) {
        MDPVals retval;
        auto & [S, A, T, R, discount] = retval;

        // Parse preamble.
        parseModelInfo(input);
        S = S_;
        A = A_;
        discount = discount_;

        if (!S || !A)
            throw std::runtime_error("MDP definition is incomplete");

        // Init matrices to store data.
        T.resize(boost::extents[S][A][S]);
        R.resize(boost::extents[S][A][S]);

        for (i_ = 0; i_ < lines_.size(); ++i_) {
            const auto & line = lines_[i_];

            if (boost::starts_with(line, "T")) {
                processMatrix(T, stateMap_, stateMap_);
                continue;
            }

            if (boost::starts_with(line, "R")) {
                processReward(R);
                continue;
            }
        }

        return retval;
    }

    CassandraParser::POMDPVals CassandraParser::parsePOMDP(std::istream & input) {
        POMDPVals retval;
        auto & [S, A, O, T, R, W, discount] = retval;

        // Parse preamble.
        parseModelInfo(input);
        S = S_;
        A = A_;
        O = O_;
        discount = discount_;

        if (!S || !A || !O)
            throw std::runtime_error("POMDP definition is incomplete");

        // Init matrices to store data.
        T.resize(boost::extents[S][A][S]);
        R.resize(boost::extents[S][A][S]);
        W.resize(boost::extents[S][A][O]);

        for (i_ = 0; i_< lines_.size(); ++i_) {
            const auto & line = lines_[i_];

            if (boost::starts_with(line, "T")) {
                processMatrix(T, stateMap_, stateMap_);
                continue;
            }

            if (boost::starts_with(line, "O")) {
                processMatrix(W, stateMap_, observationMap_);
                continue;
            }

            if (boost::starts_with(line, "R")) {
                processReward(R);
                continue;
            }
        }

        return retval;
    }

    // ############################
    // ####  PRIVATE FUNCTIONS  ###
    // ############################

    void CassandraParser::parseModelInfo(std::istream & input) {
        // This is the same for both MDP and POMDP, and we don't really care
        // here what we find or not.

        // Clear all variables that may have been setup. We don't have to clear
        // the matrices since we do that later during initialization.
        lines_.clear();
        S_ = 0, A_ = 0, O_ = 0;
        discount_ = 1.0;

        for(std::string line; std::getline(input, line); ) {
            boost::trim(line);
            if (line == "") continue;

            bool parsed = false;
            for (const auto & it : initMap_) {
                if (boost::starts_with(line, it.first)) {
                    it.second(line);
                    parsed = true;
                    break;
                }
            }

            if (!parsed)
                lines_.push_back(std::move(line));
        }
    }

    size_t CassandraParser::extractIDs(const std::string & line, IDMap & map) {
        map.clear();

        const auto split = tokenize(line, ":");
        const auto ids = tokenize(split.at(1), " ");

        // Try the number way
        if (ids.size() == 1) {
            try {
                return std::stoul(ids[0]);
            } catch (...) {}
        }

        for (size_t i = 0; i < ids.size(); ++i)
            map[boost::trim_copy(ids[i])] = i;

        return ids.size();
    }

    CassandraParser::Tokens CassandraParser::tokenize(const std::string & str, const char * list) {
        using Tokenizer = boost::tokenizer<boost::char_separator<char>>;
        boost::char_separator<char> sep(list);

        Tokenizer parser(str, sep);

        Tokens tokens;
        std::copy(std::begin(parser), std::end(parser), std::back_inserter(tokens));
        for (auto & str : tokens)
            boost::trim(str);

        return tokens;
    }

    std::vector<size_t> CassandraParser::parseIndeces(const std::string & str, const IDMap & map, const size_t max) {
        std::vector<size_t> retval;

        if (str == "*") {
            retval.resize(max);
            std::iota(std::begin(retval), std::end(retval), 0);
        } else {
            if (auto it = map.find(str); it != std::end(map)) {
                retval.push_back(it->second);
            } else {
                const size_t val = std::stoul(str);
                if (val >= max) throw std::runtime_error("Input value too high");
                retval.push_back(val);
            }
        }
        return retval;
    }

    std::vector<double> CassandraParser::parseVector(Tokens::const_iterator begin, Tokens::const_iterator end, const size_t N) {
        if (std::distance(begin, end) != (int)N)
            throw std::runtime_error("Wrong number of elements when parsing vector.");

        std::vector<double> retval;
        for (; begin < end; ++begin)
            retval.push_back(std::stod(*begin));

        return retval;
    }

    std::vector<double> CassandraParser::parseVector(const std::string & str, size_t N) {
        const auto tokens = tokenize(str, " ");
        return parseVector(std::begin(tokens), std::end(tokens), N);
    }

    void CassandraParser::processMatrix(DumbMatrix3D & M, const IDMap & d1map, const IDMap & d3map) {
        const std::string & str = lines_[i_];

        const size_t D1 = M.shape()[0];
        const size_t D2 = M.shape()[1];
        const size_t D3 = M.shape()[2];

        switch (std::count(std::begin(str), std::end(str), ':')) {
            case 3: {
                // M: <action> : <start-state> : <end-state> <prob>
                const auto tokens = tokenize(str, ": ");

                // Action is first both in transition and observation
                const auto av  = parseIndeces(tokens.at(1), actionMap_, D2);
                const auto d1v = parseIndeces(tokens.at(2), d1map, D1);
                const auto d3v = parseIndeces(tokens.at(3), d3map, D3);
                const auto val = std::stod(tokens.at(4));

                for (const auto d1 : d1v)
                    for (const auto a : av)
                        for (const auto d3 : d3v)
                            M[d1][a][d3] = val;
                break;
            }
            case 2: {
                // M: <action> : <start-state>
                // Here we need to read a vector
                const auto tokens = tokenize(str, ": ");

                const auto av  = parseIndeces(tokens.at(1), actionMap_, D2);
                const auto d1v = parseIndeces(tokens.at(2), d1map, D1);

                std::vector<double> v;
                if (tokens.size() == 3 + D3) {
                    // Parse at the end
                    v = parseVector(std::begin(tokens) + 3, std::end(tokens), D3);
                } else if (tokens.size() == 3) {
                    // Parse next line
                    v = parseVector(lines_.at(++i_), D3);
                } else {
                    std::runtime_error("Parsing error: wrong number of arguments in '" + str + "'");
                }
                for (const auto d1 : d1v)
                    for (const auto a : av)
                        for (size_t i = 0; i < v.size(); ++i)
                            M[d1][a][i] = v[i];
                break;
            }
            case 1: {
                // M: <action>
                // Here we need to read a whole 2D table
                const auto tokens = tokenize(str, ": ");
                const auto av  = parseIndeces(tokens.at(1), actionMap_, D2);

                for (size_t d1 = 0; d1 < D1; ++d1) {
                    const auto v = parseVector(lines_.at(++i_), D3);

                    for (const auto a : av)
                        for (size_t i = 0; i < v.size(); ++i)
                            M[d1][a][i] = v[i];
                }
                break;
            }
            default: throw std::runtime_error("Parsing error: wrong number of ':' in '" + str + "'");
        }
    }

    void CassandraParser::processReward(DumbMatrix3D & R) {
        const std::string & str = lines_[i_];

        const size_t S = R.shape()[0];
        const size_t A = R.shape()[1];

        switch (std::count(std::begin(str), std::end(str), ':')) {
            case 4: {
                // R: <action> : <start-state> : <end-state> : <obs> <prob>
                const auto tokens = tokenize(str, ": ");

                // Action is first both in transition and observation
                const auto av   = parseIndeces(tokens.at(1), actionMap_, A);
                const auto sv   = parseIndeces(tokens.at(2), stateMap_,  S);
                const auto s1v  = parseIndeces(tokens.at(3), stateMap_,  S);
                const auto val = std::stod(tokens.at(5));

                for (const auto s : sv)
                    for (const auto a : av)
                        for (const auto s1 : s1v)
                            R[s][a][s1] = val;
                break;
            }
            default: throw std::runtime_error("Parsing error: wrong number of ':' in '" + str + "'");
        }
    }
}
