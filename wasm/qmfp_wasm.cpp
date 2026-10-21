// qmfp featurization through RDKit MinimalLib, compiled to standalone WebAssembly for olinda.
//
// Molecules are parsed by MinimalLib::mol_from_input, the very function RDKit.js's get_mol
// calls, so validity matches RDKit.js for the same RDKit release and options. The Morgan
// generator is built exactly as MinimalLib::morgan_fp_as_bitvect builds it, but once per batch
// rather than once per molecule: re-reading the JSON options for every molecule cost more
// than the fingerprint itself.
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "common.h"

using namespace RDKit;

// The same as MinimalLib's LPT_OPT_GET, which common.h undefines after use.
#define QMFP_OPT_GET(opt) opt = pt.get(#opt, opt);

namespace {

// The options and the generator of MinimalLib::morgan_fp_as_bitvect, read from the same JSON keys
// with the same defaults (Code/MinimalLib/common.h).
std::unique_ptr<FingerprintGenerator<std::uint32_t>> morgan_generator(
    const char *details_json) {
  unsigned int radius = 2;
  unsigned int nBits = 2048;
  bool useCountSimulation = false;
  bool useChirality = false;
  bool useBondTypes = true;
  bool includeRedundantEnvironments = false;
  bool onlyNonzeroInvariants = false;
  if (details_json && std::strlen(details_json)) {
    std::istringstream ss;
    ss.str(details_json);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(ss, pt);
    QMFP_OPT_GET(radius);
    QMFP_OPT_GET(nBits);
    QMFP_OPT_GET(useCountSimulation);
    QMFP_OPT_GET(useChirality);
    QMFP_OPT_GET(useBondTypes);
    QMFP_OPT_GET(includeRedundantEnvironments);
    QMFP_OPT_GET(onlyNonzeroInvariants);
  }
  return std::unique_ptr<FingerprintGenerator<std::uint32_t>>{
      MorganFingerprint::getMorganGenerator<std::uint32_t>(
          radius, useCountSimulation, useChirality, useBondTypes,
          onlyNonzeroInvariants, includeRedundantEnvironments, nullptr, nullptr,
          nBits)};
}

}  // namespace

extern "C" {

// Featurize n NUL-separated SMILES. Writes n*nbytes fingerprint bytes to `out` and one validity
// flag per molecule to `valid`. Returns 0; -1 if a fingerprint's size is not nbytes (the options
// disagree with the caller's buffer); -2 if the fingerprint options cannot be read.
__attribute__((export_name("qmfp_featurize_batch"))) int qmfp_featurize_batch(
    const char *smiles_buf, int n, const char *mol_details, const char *fp_details,
    std::uint8_t *out, int nbytes, std::uint8_t *valid) {
  std::unique_ptr<FingerprintGenerator<std::uint32_t>> gen;
  try {
    gen = morgan_generator(fp_details);
  } catch (...) {
    return -2;
  }
  const std::string mol_json = mol_details ? mol_details : "";
  const char *s = smiles_buf;
  for (int i = 0; i < n; ++i) {
    std::uint8_t *row = out + static_cast<size_t>(i) * nbytes;
    std::memset(row, 0, nbytes);
    valid[i] = 0;
    std::string smi(s);
    s += smi.size() + 1;
    try {
      std::unique_ptr<RWMol> mol{MinimalLib::mol_from_input(smi, mol_json)};
      if (!mol || mol->getNumAtoms() == 0) {
        continue;
      }
      std::unique_ptr<ExplicitBitVect> fp{gen->getFingerprint(*mol)};
      std::string bytes = BitVectToBinaryText(*fp);
      if (static_cast<int>(bytes.size()) != nbytes) {
        return -1;
      }
      std::memcpy(row, bytes.data(), nbytes);
      valid[i] = 1;
    } catch (...) {
      // an unparseable or unsanitizable molecule: leave the zero row, valid = 0
    }
  }
  return 0;
}

__attribute__((export_name("qmfp_rdkit_version"))) const char *qmfp_rdkit_version() {
  return RDKit::rdkitVersion;
}

}  // extern "C"
