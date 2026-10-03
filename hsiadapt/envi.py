"""Minimal ENVI (.hdr + .bil/.bsq/.bip) reader backed by numpy memmap."""
from __future__ import annotations

import os
import re

import numpy as np

_ENVI_DTYPES = {
    1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64,
    12: np.uint16, 13: np.uint32, 14: np.int64, 15: np.uint64,
}


def read_header(hdr_path: str) -> dict:
    with open(hdr_path, encoding="latin-1") as f:
        txt = f.read()
    hdr: dict = {}
    # key = value   or   key = { multi, line, list }
    for m in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", txt, re.M):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if val.startswith("{"):
            val = [s.strip() for s in val[1:-1].replace("\n", " ").split(",") if s.strip()]
        hdr[key] = val
    return hdr


def data_path_for(hdr_path: str) -> str:
    """'x.bil.hdr' -> 'x.bil';  'x.hdr' -> first existing of x.bil/x.bsq/x.bip/x.raw/x."""
    base = hdr_path[:-4]
    if os.path.splitext(base)[1]:
        return base
    for ext in (".bil", ".bsq", ".bip", ".raw", ".img", ""):
        if os.path.exists(base + ext):
            return base + ext
    raise FileNotFoundError(f"No data file next to {hdr_path}")


def open_cube(hdr_path: str):
    """Returns (cube, header). cube is a lazy (bands, lines, samples) view."""
    hdr = read_header(hdr_path)
    samples, lines, bands = int(hdr["samples"]), int(hdr["lines"]), int(hdr["bands"])
    dtype = np.dtype(_ENVI_DTYPES[int(hdr["data type"])])
    if int(hdr.get("byte order", 0)) == 1:
        dtype = dtype.newbyteorder(">")
    offset = int(hdr.get("header offset", 0))
    interleave = str(hdr.get("interleave", "bil")).lower()
    shape = {
        "bil": (lines, bands, samples),
        "bsq": (bands, lines, samples),
        "bip": (lines, samples, bands),
    }[interleave]
    mm = np.memmap(data_path_for(hdr_path), dtype=dtype, mode="r", offset=offset, shape=shape)
    order = {"bil": (1, 0, 2), "bsq": (0, 1, 2), "bip": (2, 0, 1)}[interleave]
    return np.transpose(mm, order), hdr


def wavelengths(hdr: dict) -> list[float] | None:
    wl = hdr.get("wavelength")
    if not wl:
        return None
    try:
        vals = [float(v) for v in wl]
    except ValueError:
        return None
    if str(hdr.get("wavelength units", "nm")).lower().startswith("micro") or max(vals) < 50:
        vals = [v * 1000.0 for v in vals]  # um -> nm
    return vals


def write_cube(path_noext: str, cube: np.ndarray, wl: list[float] | None = None):
    """Write (bands, lines, samples) uint16/float32 cube as ENVI BIL. Used by the synthetic generator."""
    bands, lines, samples = cube.shape
    code = {np.dtype(np.uint16): 12, np.dtype(np.float32): 4, np.dtype(np.int16): 2}[cube.dtype]
    np.ascontiguousarray(np.transpose(cube, (1, 0, 2))).tofile(path_noext + ".bil")
    lines_ = [
        "ENVI",
        f"samples = {samples}", f"lines = {lines}", f"bands = {bands}",
        "header offset = 0", "file type = ENVI Standard", f"data type = {code}",
        "interleave = bil", "byte order = 0",
    ]
    if wl is not None:
        lines_.append("wavelength units = nm")
        lines_.append("wavelength = {" + ", ".join(f"{w:.2f}" for w in wl) + "}")
    with open(path_noext + ".bil.hdr", "w") as f:
        f.write("\n".join(lines_) + "\n")
