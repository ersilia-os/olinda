#!/usr/bin/env bash
# Build olinda/featurizer/rdkit_qmfp.wasm: RDKit MinimalLib's parser and Morgan generator, compiled to
# standalone WebAssembly so olinda can featurize SMILES under wasmtime with no RDKit install.
#
# Runs inside the pinned Emscripten image, with the repository mounted at /work:
#
#     docker run --rm -v "$PWD":/work -w /work emscripten/emsdk:6.0.13 bash wasm/build.sh
#
# The first run downloads Boost and RDKit and compiles RDKit, which takes a while; everything goes
# into wasm/.cache (gitignored) and later runs only relink. Flags follow RDKit's own MinimalLib Docker
# build (Code/MinimalLib/docker), with the JavaScript bindings swapped for plain C exports.
set -euxo pipefail
RDKIT_TAG=Release_2026_09_1  # the RDKit release every olinda featurizer and RDKit.js runtime must share
BOOST=1.87.0
ZLIB=1.3.2
EH="-fwasm-exceptions"
OUT=/work/olinda/featurizer/rdkit_qmfp.wasm
CACHE=/work/wasm/.cache
BOOST_U=${BOOST//./_}

mkdir -p "$CACHE" && cd "$CACHE"

if [ ! -d boost ]; then
  [ -f boost_${BOOST_U}.tar.gz ] || wget -q https://archives.boost.io/release/${BOOST}/source/boost_${BOOST_U}.tar.gz
  tar xzf boost_${BOOST_U}.tar.gz
  (cd boost_${BOOST_U} && ./bootstrap.sh --prefix="$CACHE/boost" --with-libraries=system && ./b2 install)
fi

# zlib for FileParsers (its PNG reader), which MinimalLib's molecule loader links against.
if [ ! -f zlib/lib/libz.a ]; then
  [ -f zlib-${ZLIB}.tar.gz ] || wget -q https://github.com/madler/zlib/releases/download/v${ZLIB}/zlib-${ZLIB}.tar.gz
  tar xzf zlib-${ZLIB}.tar.gz
  (mkdir -p zlib-${ZLIB}/build && cd zlib-${ZLIB}/build && \
    emcmake cmake -DCMAKE_BUILD_TYPE=Release -DZLIB_BUILD_EXAMPLES=OFF \
      -DCMAKE_C_FLAGS="${EH}" -DCMAKE_EXE_LINKER_FLAGS="${EH}" -DCMAKE_INSTALL_PREFIX="$CACHE/zlib" .. && \
    make -j"$(nproc)" && make install)
fi

[ -d rdkit ] || git clone --depth 1 --branch ${RDKIT_TAG} https://github.com/rdkit/rdkit.git

mkdir -p rdkit/build && cd rdkit/build
# Re-run every time: cheap, idempotent, and it picks up option changes without discarding compiled
# libraries.
emcmake cmake -DRDK_BUILD_MINIMAL_LIB=ON -DRDK_BUILD_FREETYPE_SUPPORT=OFF \
  -DZLIB_INCLUDE_DIR="$CACHE/zlib/include" -DZLIB_LIBRARY="$CACHE/zlib/lib/libz.a" \
  -DRDK_BUILD_BOOST_PYTHON_WRAPPERS=OFF -DRDK_BUILD_NANOBIND_WRAPPERS=OFF \
  -DRDK_BUILD_CPP_TESTS=OFF -DRDK_BUILD_INCHI_SUPPORT=OFF -DRDK_BUILD_AVALON_SUPPORT=OFF \
  -DRDK_USE_BOOST_SERIALIZATION=OFF -DRDK_OPTIMIZE_POPCNT=OFF -DRDK_BUILD_THREADSAFE_SSS=OFF \
  -DRDK_BUILD_DESCRIPTORS3D=OFF -DRDK_TEST_MULTITHREADED=OFF \
  -DRDK_BUILD_CHEMDRAW_SUPPORT=OFF -DRDK_BUILD_MAEPARSER_SUPPORT=OFF -DRDK_BUILD_COORDGEN_SUPPORT=OFF \
  -DRDK_BUILD_SLN_SUPPORT=OFF -DRDK_USE_BOOST_IOSTREAMS=OFF \
  -DBoost_DIR="$CACHE/boost/lib/cmake/Boost-${BOOST}" \
  -Dboost_headers_DIR="$CACHE/boost/lib/cmake/boost_headers-${BOOST}" \
  -DCMAKE_CXX_FLAGS="${EH} -O3 -DNDEBUG" \
  -DCMAKE_C_FLAGS="${EH} -O3 -DNDEBUG -DCOMPILE_ANSI_ONLY" ..

# Every static library RDKit_minimal links, plus the two MinimalLib's molecule loader references
# (FileParsers for MolBlocks, RGroupDecomposition for mapped dummies), without the JS executable itself.
make -j"$(nproc)" $(grep -o 'MINIMAL_LIB_LIBRARIES "[^"]*"' ../Code/MinimalLib/CMakeLists.txt | head -1 | sed 's/.*"\(.*\)"/\1/; s/;/ /g') \
  SmilesParse Fingerprints GraphMol RDGeneral DataStructs FileParsers RGroupDecomposition

# Compile flags exactly as CMake gives RDKit_minimal's own sources.
FLAGS=Code/MinimalLib/CMakeFiles/RDKit_minimal.dir/flags.make
DEFS=$(sed -n 's/^CXX_DEFINES = //p' $FLAGS)
INCS=$(sed -n 's/^CXX_INCLUDES = //p' $FLAGS)
em++ ${EH} -O3 -DNDEBUG -std=c++20 ${DEFS} ${INCS} -I../Code/MinimalLib -c /work/wasm/qmfp_wasm.cpp -o "$CACHE/qmfp_wasm.o"
# MinimalLib's JSON option parsing, which mol_from_input calls.
em++ ${EH} -O3 -DNDEBUG -std=c++20 ${DEFS} ${INCS} -I../Code/MinimalLib -c ../Code/MinimalLib/JSONParsers.cpp -o "$CACHE/JSONParsers.o"

# A WASI reactor: no main, the host calls _initialize once to run C++ static constructors. Boost is
# header-only here (its system library is a host build, not WebAssembly), so it is not linked.
em++ ${EH} -O3 "$CACHE/qmfp_wasm.o" "$CACHE/JSONParsers.o" -Wl,--start-group $(find lib -name 'lib*.a' | sort) -Wl,--end-group \
  -sSTANDALONE_WASM=1 -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sUSE_PTHREADS=0 \
  -sEXPORTED_FUNCTIONS=_qmfp_featurize_batch,_qmfp_rdkit_version,_malloc,_free \
  --no-entry -o "$CACHE/rdkit_qmfp.wasm"

# wasmtime runs the standardised exception-handling instructions (exnref); Emscripten emits the legacy
# ones. Translate the finished module, enabling only the features it uses: --all-features would also
# switch on encodings wasmtime cannot parse.
/emsdk/upstream/bin/wasm-opt --enable-exception-handling --enable-reference-types --enable-bulk-memory \
  --enable-bulk-memory-opt --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
  --enable-multivalue --translate-to-exnref "$CACHE/rdkit_qmfp.wasm" -o "$OUT"
ls -la "$OUT"
