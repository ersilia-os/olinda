// qmfp featurization through RDKit MinimalLib, compiled to standalone WebAssembly for olinda.
//
// Molecules are parsed by MinimalLib::mol_from_input, the very function RDKit.js's get_mol
// calls, so validity matches RDKit.js for the same RDKit release and options. Fingerprints use
// the Morgan options MinimalLib::morgan_fp_as_bitvect reads, from the same JSON keys with the
// same defaults.
//
// With count simulation at RDKit's default bounds (1, 2, 4, 8), which is the qmfp layout, the
// fingerprint is computed as true folded counts, quantized and packed here, instead of as RDKit's
// count-simulation bit vector. The bytes are identical (checked on the whole reference library),
// and the count fingerprint is cheaper: it skips building a 4x-wide bit vector. Any other options
// take MinimalLib's own bit-vector path.
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

// Bits per slot under count simulation with RDKit's default count bounds.
constexpr unsigned int kBitsPerSlot = 4;
// A true count's thermometer nibble: counts 1, 2-3, 4-7 and >=8 set 1, 2, 3 and 4 low bits.
constexpr std::uint8_t kNibble[9] = {0x0, 0x1, 0x3, 0x3, 0x7, 0x7, 0x7, 0x7, 0xF};

struct Morgan {
  std::unique_ptr<FingerprintGenerator<std::uint32_t>> gen;
  bool counts = false;  // compute counts and pack them, rather than the simulated bit vector
  unsigned int slots = 0;
};

// MinimalLib::morgan_fp_as_bitvect's options, read from the same JSON keys with the same defaults
// (Code/MinimalLib/common.h).
Morgan morgan(const char *details_json) {
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
  Morgan m;
  m.counts = useCountSimulation && nBits % (2 * kBitsPerSlot) == 0;
  m.slots = m.counts ? nBits / kBitsPerSlot : nBits;
  // Counts fold into `slots`; the simulated bit vector spreads the same slots over `nBits` bits.
  m.gen.reset(MorganFingerprint::getMorganGenerator<std::uint32_t>(
      radius, m.counts ? false : useCountSimulation, useChirality, useBondTypes,
      onlyNonzeroInvariants, includeRedundantEnvironments, nullptr, nullptr,
      m.counts ? m.slots : nBits));
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
  if (fp.counts && static_cast<unsigned int>(nbytes) * 2 != fp.slots) {
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
      if (fp.counts) {
        // Slot k's nibble sits in byte k/2: the even slot in the low half, the odd in the high.
        std::unique_ptr<SparseIntVect<std::uint32_t>> counts{
            fp.gen->getCountFingerprint(*mol)};
        for (const auto &[slot, count] : counts->getNonzeroElements()) {
          const std::uint8_t nibble = kNibble[count < 8 ? count : 8];
          row[slot / 2] |= static_cast<std::uint8_t>(nibble << (4 * (slot % 2)));
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
