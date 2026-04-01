from typing import Optional
import pandas as pd

# ---------------------------------------------------------------------------
# Форматтеры
# ---------------------------------------------------------------------------

def _fmt_ts(sec: Optional[float]) -> str:
    if sec is None or (isinstance(sec, float) and pd.isna(sec)):
        return "--:--.--"
    m, s = divmod(float(sec), 60)
    return f"{int(m):02d}:{s:05.2f}"


def _fmt_dur(sec: Optional[float]) -> str:
    if sec is None or (isinstance(sec, float) and pd.isna(sec)):
        return "    --"
    return f"{float(sec):6.1f}s"