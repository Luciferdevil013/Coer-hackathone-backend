"""File-type detection and safe content readers."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

TABULAR = {"csv", "tsv", "xlsx", "xlsm", "xls", "json"}
DATABASE = {"db", "sqlite", "sqlite3"}
DOCUMENT = {"pdf", "docx", "pptx", "doc"}
TEXT = {"txt", "md", "log", "env", "cfg", "ini", "yaml", "yml", "sql", "xml", "html", "conf", "properties", "toml"}
ARCHIVE = {"zip", "tar", "gz", "tgz", "7z", "rar", "bz2"}
MEDIA = {"png", "jpg", "jpeg", "gif", "mp4", "mov", "psd", "ai", "mp3", "wav", "svg"}
BACKUP_SUFFIXES = {"bak", "old", "orig", "backup", "tmp"}


def effective_ext(name: str) -> str:
    """'config.env.bak' -> 'env', '.env' -> 'env', 'a.CSV' -> 'csv'."""
    exts = name.lower().split(".")[1:]
    while len(exts) > 1 and exts[-1] in BACKUP_SUFFIXES:
        exts.pop()
    return exts[-1] if exts else ""


def detect_kind(name: str) -> str:
    ext = effective_ext(name)
    if name.lower().startswith(".env"):
        return "text"
    if ext in TABULAR:
        return "tabular"
    if ext in DATABASE:
        return "database"
    if ext in DOCUMENT:
        return "document"
    if ext in TEXT:
        return "text"
    if ext in ARCHIVE:
        return "archive"
    if ext in MEDIA:
        return "media"
    return "binary"


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def read_table(path: Path, max_rows: int, errors: list[str] | None = None) -> pd.DataFrame | None:
    """Load a tabular file as all-string columns (no silent type coercion).
    On failure returns None and, if `errors` is given, appends a human-readable reason."""
    ext = effective_ext(path.name)
    try:
        if ext in ("csv", "tsv"):
            sep = "\t" if ext == "tsv" else None
            opts = dict(dtype=str, sep=sep, engine="python" if sep is None else "c", nrows=max_rows,
                        keep_default_na=False, na_values=[""])
            try:  # utf-8-sig also drops a byte-order mark that would otherwise stick to the first header
                return _clean_headers(pd.read_csv(path, encoding="utf-8-sig", **opts))
            except UnicodeDecodeError:
                # older Windows / Excel exports ('£' as one byte); reading those as UTF-8 manufactures garbled text
                return _clean_headers(pd.read_csv(path, encoding="cp1252", encoding_errors="replace", **opts))
        if ext in ("xlsx", "xlsm", "xls"):
            # first sheet that has data (exports often leave an empty 'Sheet1' in front)
            with pd.ExcelFile(path) as book:
                for sheet in book.sheet_names:
                    df = book.parse(sheet, dtype=str, nrows=max_rows, keep_default_na=False, na_values=[""])
                    if len(df.columns) and len(df):
                        return _clean_headers(df)
            if errors is not None:
                errors.append("The workbook has no sheet with data")
            return None
        if ext == "json":
            with path.open("r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, dict):
                lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
                data = lists[0] if lists else None
            if isinstance(data, list) and data and isinstance(data[0], dict):
                df = pd.json_normalize(data[:max_rows])
                return df.astype(object).where(df.notna(), None).map(lambda x: None if x is None else str(x))
            if errors is not None:
                errors.append("JSON is not a list of records")
            return None
    except Exception as e:
        if errors is not None:
            msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            if ext in ("xlsx", "xlsm", "xls") and re.search(r"cannot be determined|not a zip file|unsupported format|corrupt|BOF", msg, re.I):
                msg = "it is damaged, password-protected or not really an Excel workbook"
            errors.append(f"Could not read the {ext.upper()} file: {msg}")
        return None
    return None


def read_text(path: Path, limit: int = 5_000_000) -> str:
    ext = effective_ext(path.name)
    try:
        if ext == "pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n".join((p.extract_text() or "") for p in reader.pages[:200])[:limit]
        if ext in ("docx", "pptx"):
            with zipfile.ZipFile(path) as z:
                xml = " ".join(
                    z.read(n).decode("utf-8", "replace")
                    for n in z.namelist()
                    if n.endswith(".xml") and ("word/document" in n or "ppt/slides/slide" in n)
                )
            return re.sub(r"<[^>]+>", " ", xml)[:limit]
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except Exception:
        return ""


def zip_listing(path: Path, limit: int = 2000) -> dict:
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            return {
                "members": len(infos),
                "uncompressed_bytes": sum(i.file_size for i in infos),
                "sample_members": [i.filename for i in infos[:12]],
                "member_years": sorted({i.date_time[0] for i in infos[:limit]}),
            }
    except Exception:
        return {}


def sqlite_schema(path: Path) -> dict:
    """Read-only introspection of a SQLite database: tables, columns, row and non-null counts."""
    out: dict = {"tables": {}}
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cur = con.cursor()
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for t in tables:
            info = list(cur.execute(f'PRAGMA table_info("{t}")'))
            cols = [r[1] for r in info]
            pk = [r[1] for r in info if r[5]]
            rows = cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            non_null, nbytes = {}, {}
            for c in cols:
                cnt, size = cur.execute(
                    f'SELECT COUNT("{c}"), COALESCE(SUM(LENGTH(CAST("{c}" AS BLOB))), 0) FROM "{t}" '
                    f'WHERE "{c}" IS NOT NULL AND "{c}" != \'\''
                ).fetchone()
                non_null[c], nbytes[c] = cnt, size
            out["tables"][t] = {"rows": rows, "columns": cols, "pk": pk, "non_null": non_null, "bytes": nbytes}
        con.close()
    except Exception as e:  # corrupt / not sqlite
        out["error"] = str(e)
    return out


def sqlite_table(path: Path, table: str, max_rows: int) -> pd.DataFrame | None:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        df = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {int(max_rows)}', con)
        con.close()
        return df.astype(object).where(df.notna(), None).map(lambda x: None if x is None else str(x))
    except Exception:
        return None
