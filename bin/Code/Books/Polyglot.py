# This code is a translation to python from pg_key.c and pg_show.c released in the public domain by Michel Van den Bergh
# http://alpha.uhasselt.be/Research/Algebra/Toga

import os
import struct
import sys

import FasterCode

from Code.Base.Constantes import (
    ALL_BEST_MOVES,
    ALL_MOVES,
    FEN_INITIAL,
    FIRST_BEST_MOVE,
    TOP2_FIRST_MOVES,
    TOP3_FIRST_MOVES,
)


class Entry:
    key = 0
    move = 0
    weight = 0
    learn = 0

    def pv(self):
        move = self.move

        f = (move >> 6) & 0o77
        fr = (f >> 3) & 0x7
        ff = f & 0x7
        t = move & 0o77
        tr = (t >> 3) & 0x7
        tf = t & 0x7
        p = (move >> 12) & 0x7
        pv = chr(ff + ord("a")) + chr(fr + ord("1")) + chr(tf + ord("a")) + chr(tr + ord("1"))
        if p:
            pv += " nbrq"[p]

        return {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}.get(pv, pv)


class Polyglot:
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fich = "varied.bin"

    p = Polyglot()
    li = p.lista( fich, fen )

    for entry in li:
        p rint entry.pv(), entry.weight
    """

    def __init__(self, path=None):
        self.path = path
        self.f = None

    def __enter__(self):
        self.f = open(self.path, "rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f:
            self.f.close()
            self.f = None

    def int_from_file(self, length, r):
        cad = self.f.read(length)

        if len(cad) != length:
            return True, 0
        for c in cad:
            r = (r << 8) + c
        return False, r

    def entry_from_file(self):
        entry = Entry()

        r = 0
        ret, r = self.int_from_file(8, r)
        if ret:
            return True, None
        entry.key = r

        ret, r = self.int_from_file(2, r)
        if ret:
            return True, None
        entry.move = r & 0xFFFF

        ret, r = self.int_from_file(2, r)
        if ret:
            return True, None
        entry.weight = r & 0xFFFF

        ret, r = self.int_from_file(4, r)
        if ret:
            return True, None
        entry.learn = r & 0xFFFFFFFF

        return False, entry

    def find_key(self, key):
        first = -1
        try:
            self.f.seek(-16, os.SEEK_END)
        except OSError:
            entry = Entry()
            entry.key = key + 1
            return -1, entry

        last = self.f.tell() // 16
        ret, last_entry = self.entry_from_file()
        while True:
            if last - first == 1:
                return last, last_entry

            middle = (first + last) // 2
            self.f.seek(16 * middle, os.SEEK_SET)
            ret, middle_entry = self.entry_from_file()
            if key <= middle_entry.key:
                last = middle
                last_entry = middle_entry
            else:
                first = middle

    def lista(self, path, fen):
        with open(path, "rb") as self.f:
            return self.xlista(fen)

    def xlista(self, fen):
        key = FasterCode.hash_polyglot8(fen)

        offset, entry = self.find_key(key)
        li = []
        if entry and entry.key == key:

            li.append(entry)

            self.f.seek(16 * (offset + 1), os.SEEK_SET)
            while True:
                ret, entry = self.entry_from_file()
                if ret or (entry.key != key):
                    break

                li.append(entry)

            li.sort(key=lambda x: x.weight, reverse=True)

        return li


def _find_entry_offset(path, key, poly_move):
    """Binary-search for entry with given key and move. Returns byte offset or -1."""
    ENTRY_SIZE = 16
    KEY_FMT = struct.Struct(">Q")
    MOVE_FMT = struct.Struct(">H")
    file_size = os.path.getsize(path)
    num_entries = file_size // ENTRY_SIZE
    if num_entries == 0:
        return -1

    with open(path, "rb") as f:
        lo, hi = 0, num_entries - 1
        first_match = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            f.seek(mid * ENTRY_SIZE)
            ek = KEY_FMT.unpack(f.read(8))[0]
            if ek < key:
                lo = mid + 1
            elif ek > key:
                hi = mid - 1
            else:
                first_match = mid
                hi = mid - 1

        if first_match < 0:
            return -1

        pos = first_match
        while pos < num_entries:
            f.seek(pos * ENTRY_SIZE)
            data = f.read(ENTRY_SIZE)
            if len(data) < ENTRY_SIZE:
                break
            ek = KEY_FMT.unpack(data[:8])[0]
            if ek != key:
                break
            em = MOVE_FMT.unpack(data[8:10])[0]
            if em == poly_move:
                return pos * ENTRY_SIZE
            pos += 1
    return -1


def set_entry_weight(path, key, poly_move, new_weight):
    """Set the weight of an existing entry in-place. Returns True on success."""
    offset = _find_entry_offset(path, key, poly_move)
    if offset < 0:
        return False
    WEIGHT_FMT = struct.Struct(">H")
    with open(path, "r+b") as f:
        f.seek(offset + 10)
        f.write(WEIGHT_FMT.pack(min(max(new_weight, 0), 32767)))
    return True


def delete_entry(path, key, poly_move):
    """Delete an entry by streaming the file without it. Returns True on success."""
    offset = _find_entry_offset(path, key, poly_move)
    if offset < 0:
        return False
    ENTRY_SIZE = 16
    CHUNK = 1024 * 1024
    temp_path = path + ".tmp"
    file_size = os.path.getsize(path)
    with open(path, "rb") as fin, open(temp_path, "wb") as fout:
        # Copy everything before the entry
        remaining = offset
        while remaining > 0:
            chunk = fin.read(min(remaining, CHUNK))
            if not chunk:
                break
            fout.write(chunk)
            remaining -= len(chunk)
        # Skip the entry
        fin.seek(offset + ENTRY_SIZE)
        # Copy everything after
        while True:
            chunk = fin.read(CHUNK)
            if not chunk:
                break
            fout.write(chunk)
    os.replace(temp_path, path)
    return True


def is_polyglot_sorted(path, sample_size=1000):
    """Quick check: read first *sample_size* entries and verify keys are non-decreasing."""
    entry_size = 16
    key_fmt = struct.Struct(">Q")
    prev_key = 0
    try:
        with open(path, "rb") as f:
            for _ in range(sample_size):
                data = f.read(entry_size)
                if len(data) < entry_size:
                    break
                key = key_fmt.unpack(data[:8])[0]
                if key < prev_key:
                    return False
                prev_key = key
    except OSError:
        return True
    return True


def sort_polyglot_file(path):
    """Sort a polyglot .bin book by key (big-endian uint64).

    Uses numpy when available (fastest, ~1.6 GB RAM for 100 M entries),
    otherwise falls back to an external merge-sort that needs far less memory.
    Trailing bytes that don't form a complete 16-byte entry are dropped.
    The file is replaced atomically via a temp file.
    """
    file_size = os.path.getsize(path)
    num_entries = file_size // 16
    if num_entries <= 1:
        return

    temp_path = path + ".sorted.tmp"
    try:
        _sort_with_numpy(path, temp_path, num_entries)
    except Exception:
        _sort_external(path, temp_path, num_entries)
    os.replace(temp_path, path)


def _sort_with_numpy(path, temp_path, num_entries):
    import numpy as np

    dt = np.dtype(
        [
            ("key", ">u8"),
            ("move", ">u2"),
            ("weight", ">u2"),
            ("score", ">u2"),
            ("depth", "u1"),
            ("learn", "u1"),
        ]
    )
    data = np.fromfile(path, dtype=dt, count=num_entries)
    data.sort(order="key")
    data.tofile(temp_path)


def _sort_external(path, temp_path, num_entries):
    """Fallback external merge-sort for systems without numpy."""
    import heapq
    import tempfile

    ENTRY_SIZE = 16
    CHUNK_ENTRIES = 2_000_000  # 32 MB of raw data per chunk

    chunk_paths = []
    td = tempfile.mkdtemp()
    try:
        # Phase 1 – create sorted chunks
        with open(path, "rb") as f:
            ci = 0
            while True:
                raw = f.read(CHUNK_ENTRIES * ENTRY_SIZE)
                if not raw:
                    break
                usable = len(raw) - len(raw) % ENTRY_SIZE
                if usable == 0:
                    break
                raw = raw[:usable]
                mv = memoryview(raw)
                entries = [bytes(mv[i : i + ENTRY_SIZE]) for i in range(0, usable, ENTRY_SIZE)]
                entries.sort()
                cp = os.path.join(td, f"c{ci:04d}.bin")
                with open(cp, "wb") as cf:
                    cf.write(b"".join(entries))
                chunk_paths.append(cp)
                ci += 1

        # Phase 2 – k-way merge
        class _ChunkIter:
            """Buffered reader that yields 16-byte entries."""
            BUF = 1_048_576  # 1 MB read buffer

            def __init__(self, p):
                self._f = open(p, "rb")
                self._buf = b""
                self._off = 0

            def __del__(self):
                self._f.close()

            def __iter__(self):
                return self

            def __next__(self):
                if self._off >= len(self._buf):
                    self._buf = self._f.read(self.BUF)
                    self._off = 0
                    if not self._buf:
                        raise StopIteration
                e = self._buf[self._off : self._off + ENTRY_SIZE]
                self._off += ENTRY_SIZE
                if len(e) < ENTRY_SIZE:
                    raise StopIteration
                return e

        with open(temp_path, "wb") as fout:
            for entry in heapq.merge(*[_ChunkIter(cp) for cp in chunk_paths]):
                fout.write(entry)
    finally:
        for cp in chunk_paths:
            try:
                os.remove(cp)
            except OSError:
                pass
        try:
            os.rmdir(td)
        except OSError:
            pass


class Line:
    def __init__(
        self,
        pol_w,
        pol_b,
        lines,
        mode_white,
        mode_black,
        start_fen,
        dispatch,
        porc_min_white,
        porc_min_black,
        weight_min_white,
        weight_min_black,
    ):
        self.li_pv = []
        self.st_fens_m2 = set()
        self.start_fen = start_fen or FEN_INITIAL
        self.last_fen = self.start_fen

        self.finished = False
        self.pol_w: Polyglot = pol_w
        self.pol_b: Polyglot = pol_b
        self.mode_white = mode_white
        self.mode_black = mode_black

        self.porc_min_white = porc_min_white
        self.porc_min_black = porc_min_black
        self.weight_min_white = weight_min_white
        self.weight_min_black = weight_min_black

        self.dispatch = dispatch

        self.lines = lines
        self.lines.append(self)

    def add_entry(self, xentry: Entry):
        FasterCode.set_fen(self.last_fen)
        pv = xentry.pv()
        FasterCode.make_move(pv)
        new_fen = FasterCode.get_fen()
        new_fenm2 = FasterCode.fen_fenm2(new_fen)
        if new_fenm2 in self.st_fens_m2:
            self.finished = True
            return False
        self.li_pv.append(pv)
        self.st_fens_m2.add(new_fenm2)
        self.last_fen = new_fen
        return True

    def next_level(self, xmax_lines) -> bool:
        if len(self.lines) > xmax_lines:
            return False
        if self.finished:
            return False
        is_white = "w" in self.last_fen
        polyglot = self.pol_w if is_white else self.pol_b
        li_entries = polyglot.xlista(self.last_fen)
        if not li_entries:
            self.finished = True
            return False
        xentry: Entry

        if is_white:
            mode = self.mode_white
            porc = self.porc_min_white
            min_weight = self.weight_min_white
        else:
            mode = self.mode_black
            porc = self.porc_min_black
            min_weight = self.weight_min_black

        if porc:
            tt = sum(xentry.weight for xentry in li_entries)
            if tt == 0:
                self.finished = True
                return False
            li_entries = [xentry for xentry in li_entries if xentry.weight / tt >= porc]
            if not li_entries:
                self.finished = True
                return False

        if min_weight:
            li_entries = [xentry for xentry in li_entries if xentry.weight >= min_weight]

        if mode != ALL_MOVES:
            li_entries.sort(key=lambda x: x.weight, reverse=True)
            if mode == FIRST_BEST_MOVE:
                li_entries = li_entries[:1]
            elif mode == ALL_BEST_MOVES:
                weight1 = li_entries[0].weight
                li_entries = [entry for entry in li_entries if entry.weight == weight1]
            elif mode == TOP2_FIRST_MOVES:
                li_entries = li_entries[:2]
            elif mode == TOP3_FIRST_MOVES:
                li_entries = li_entries[:3]

        for xentry in li_entries[1:]:
            if len(self.lines) >= xmax_lines:
                break
            new_line = Line(
                self.pol_w,
                self.pol_b,
                self.lines,
                self.mode_white,
                self.mode_black,
                self.start_fen,
                self.dispatch,
                0.0,
                0.0,
                0,
                0,
            )
            new_line.li_pv = self.li_pv[:]
            new_line.st_fens_m2 = set(self.st_fens_m2)
            new_line.last_fen = self.last_fen
            if not new_line.add_entry(xentry):
                del self.lines[-1]

            if not self.dispatch(len(self.li_pv) + 1, len(self.lines)):
                return False

        self.add_entry(li_entries[0])
        return True

    def __str__(self):
        return " ".join(self.li_pv)

    def __len__(self):
        return len(self.li_pv)


def dic_modes():
    return {
        FIRST_BEST_MOVE: _("First best move"),
        ALL_BEST_MOVES: _("All best moves"),
        TOP2_FIRST_MOVES: _("Two first moves"),
        TOP3_FIRST_MOVES: _("Three first moves"),
        ALL_MOVES: _("All moves"),
    }


def gen_lines(
    path_pol_w,
    path_pol_b,
    mode_w,
    mode_b,
    max_lines,
    max_depth,
    start_fen,
    dispatch,
    porc_min_white=None,
    porc_min_black=None,
    weight_min_white=None,
    weight_min_black=None,
):
    with Polyglot(path_pol_w) as pol_w, Polyglot(path_pol_b) as pol_b:
        lines = []
        Line(
            pol_w,
            pol_b,
            lines,
            mode_w,
            mode_b,
            start_fen,
            dispatch,
            porc_min_white,
            porc_min_black,
            weight_min_white,
            weight_min_black,
        )

        if max_depth == 0:
            max_depth = sys.maxsize
        if max_lines == 0:
            max_lines = sys.maxsize

        depth = 0
        while depth < max_depth:
            ok = False
            num_lines = len(lines)
            for pos in range(num_lines):
                line = lines[pos]
                if line.next_level(max_lines):
                    ok = True
                else:
                    if not dispatch(None, None):
                        break
            if not ok:
                break
            depth += 1

    return lines
