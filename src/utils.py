# src/utils.py

import re

def normalize_to_ipc(raw: str) -> str | None:
    """
    Convert raw tokens like:
      "120-B", "120B", "376(2)(g)", "Section 302", "IPC_302" 
    into canonical ids like "IPC_120B", "IPC_376_2_G", "IPC_302".

    Returns None for empty or obviously non-statute inputs (e.g., year 1860).
    """
    if not raw:
        return None
    s = str(raw).upper().strip()

    # Remove common leading words
    s = re.sub(r'^(SECTION|SEC|S)\.?\s*', '', s)

    # If already probably canonical (starts with IPC_), normalize and return
    if s.startswith("IPC_") or s.startswith("IPC-") or s.startswith("IPC "):
        s2 = re.sub(r'[^0-9A-Z\_]', '', s.replace('-', '_').replace(' ', '_'))
        return s2 if s2.startswith("IPC_") else f"IPC_{s2.lstrip('IPC_')}"

    # Remove 'IPC' token if present (we'll add it back)
    s = s.replace("IPC", "")
    s = s.strip(" :,.-")

    # If it's a 4-digit year (1700-2100), treat as non-statute (likely year)
    m_year = re.fullmatch(r'(\d{4})', s)
    if m_year:
        val = int(m_year.group(1))
        if 1700 <= val <= 2100:
            return None

    # Normalize patterns like 376(2)(g)
    # turn )(... into underscores and remove stray punctuation
    # Step A: join consecutive parens: ")( " -> ")(" handled later
    s = s.replace(')(', ')(')
    # Replace )(... ) with underscores: e.g., 376(2)(g) -> 376_2_g
    s = re.sub(r'\)\s*\(', '_', s)
    s = s.replace('(', '_').replace(')', '_')

    # Remove problematic chars, keep numbers, letters and underscores
    s = re.sub(r'[^0-9A-Z_]', '', s)
    # collapse multiple underscores
    s = re.sub(r'_+', '_', s).strip('_')

    # Remove stray leading/trailing underscores
    s = s.strip('_')

    # If empty or purely numeric but suspicious (e.g., 1860 already filtered), still allow numeric
    if not s:
        return None

    # Final canonical id
    return f"IPC_{s}"
