// qmfp featurization through RDKit MinimalLib, compiled to standalone WebAssembly for olinda.
//
// Molecules are parsed by MinimalLib::mol_from_input, the very function RDKit.js's get_mol
// calls, so validity matches RDKit.js for the same RDKit release and options. Fingerprints use
// the Morgan options MinimalLib::morgan_fp_as_bitvect reads, from the same JSON keys with the
// same defaults.
//
// With "countEncoding": "count4" (olinda's qmfp), the fingerprint is RDKit's folded Morgan count
// fingerprint over nBits slots, each count clipped at 15 and written as a 4-bit number: two slots
// per byte, the even slot in the low nibble. Exact counts up to 15 keep everything a model learns
// from counts (above 15 is 0.03% of nonzero slots), where RDKit's count simulation would bucket
// 2-3 and 4-7 together and lose accuracy. Without it, MinimalLib's own bit-vector path is used.
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

constexpr std::uint32_t kCountClip = 15;  // the largest count a nibble holds

struct Morgan {
  std::unique_ptr<FingerprintGenerator<std::uint32_t>> gen;
  bool count4 = false;  // 4-bit counts, rather than MinimalLib's bit vector
  unsigned int nBits = 0;
};

// MinimalLib::morgan_fp_as_bitvect's options, read from the same JSON keys with the same defaults
// (Code/MinimalLib/common.h), plus olinda's countEncoding.
Morgan morgan(const char *details_json) {
  unsigned int radius = 2;
  unsigned int nBits = 2048;
  bool useCountSimulation = false;
  bool useChirality = false;
  bool useBondTypes = true;
  bool includeRedundantEnvironments = false;
  bool onlyNonzeroInvariants = false;
  std::string countEncoding;
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
    QMFP_OPT_GET(countEncoding);
  }
  Morgan m;
  m.count4 = countEncoding == "count4";
  m.nBits = nBits;
  m.gen.reset(MorganFingerprint::getMorganGenerator<std::uint32_t>(
      radius, m.count4 ? false : useCountSimulation, useChirality, useBondTypes,
      onlyNonzeroInvariants, includeRedundantEnvironments, nullptr, nullptr, nBits));
  return m;
}

}  // namespace

extern "C" {

// Featurize n NUL-separated SMILES. Writes n*nbytes fingerprint bytes to `out` and one validity
// flag per molecule to `valid`. Returns 0; -1 if a fingerprint's size is not nbytes (the options
// disagree with the caller's buffer); -2 if the fingerprint options cannot be read.
__attribute__((export_name("qmfp_featurize_batch"))) int qmfp_featurize_batch(
    const char *smiles_buf, int n, const char *mol_details, const char *fp_details,
    std::uint8_t *out, int nbytes, std::uint8_t *valid) {
  Morgan fp;
  try {
    fp = morgan(fp_details);
  } catch (...) {
    return -2;
  }
  if (fp.count4 && static_cast<unsigned int>(nbytes) * 2 != fp.nBits) {
    return -1;
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
      if (fp.count4) {
        // Slot k's count sits in byte k/2: the even slot in the low nibble, the odd in the high.
        std::unique_ptr<SparseIntVect<std::uint32_t>> counts{
            fp.gen->getCountFingerprint(*mol)};
        for (const auto &[slot, count] : counts->getNonzeroElements()) {
          const std::uint32_t c = count < kCountClip ? count : kCountClip;
          row[slot / 2] |= static_cast<std::uint8_t>(c << (4 * (slot % 2)));
        }
      } else {
        std::unique_ptr<ExplicitBitVect> bits{fp.gen->getFingerprint(*mol)};
        std::string bytes = BitVectToBinaryText(*bits);
        if (static_cast<int>(bytes.size()) != nbytes) {
          return -1;
        }
        std::memcpy(row, bytes.data(), nbytes);
      }
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
