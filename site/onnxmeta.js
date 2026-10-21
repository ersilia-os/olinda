// Read ONNX ModelProto.metadata_props (field 14) without a protobuf library.
export function readOnnxMetadata(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const dec = new TextDecoder();
  const varint = (p) => { let x = 0, s = 0, b; do { b = u8[p.i++]; x += (b & 0x7f) * 2 ** s; s += 7; } while (b & 0x80); return x; };
  const fields = (start, end, onField) => {
    const p = { i: start };
    while (p.i < end) {
      const key = varint(p), field = key >>> 3, wire = key & 7;
      if (wire === 0) { varint(p); }
      else if (wire === 1) { p.i += 8; }
      else if (wire === 5) { p.i += 4; }
      else if (wire === 2) { const len = varint(p); onField(field, p.i, p.i + len); p.i += len; }
      else throw new Error(`unsupported wire type ${wire}`);
    }
  };
  const meta = {};
  fields(0, u8.length, (f, a, b) => {
    if (f !== 14) return;                       // ModelProto.metadata_props
    let k = "", v = "";
    fields(a, b, (g, c, d) => { const s = dec.decode(u8.subarray(c, d)); if (g === 1) k = s; else if (g === 2) v = s; });
    meta[k] = v;
  });
  return meta;
}
