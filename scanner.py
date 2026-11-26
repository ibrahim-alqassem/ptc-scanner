"""
ماسح توافق الزمن والسعر (PTC Scanner)
يطبّق منطق مؤشري "توافق الزمن والسعر" و"صائد الاختراقات" على قائمة أسهم،
ويكتب النتائج في docs/<market>.json لتعرضها صفحة docs/index.html.

التشغيل:  python scanner.py --market us   |   python scanner.py --market sa
أداة مساعدة وليست توصية.
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"

# ===== الإعدادات (نفس قيم المؤشرين) =====
PIV = 5            # حساسية القمم والقيعان
CYC_N = 4          # عدد الدورات لحساب الإيقاع
TOL_PCT = 15.0     # عرض النافذة % من طول الدورة
MIN_TOL = 2
MIN_LEG = 10       # أقل طول موجة لفيبو الزمني
PR_TOL = 1.5 / 100 # قرب السعر من المستوى
VOL_MULT = 1.5     # انفجار الحجم
VP_LEN, VP_BINS, HVN_MULT = 120, 40, 1.3
RS_LEN = 63
TIGHT_LEN, TIGHT_PCT = 10, 10.0
FT_LEN = 252
MIN_PRICE = 5.0
LOW_PRICE = 1.0    # تبويب "أقل من 5$": من 1 إلى 5
LOW_AVGVOL = {"us": 100_000, "sa": 50_000}
LOW_FLOAT = 20e6   # أسهم حرة قليلة
MIN_AVGVOL = {"us": 500_000, "sa": 100_000}   # السوق السعودي أحجامه بالأسهم أقل
FLOAT_TOP = 50     # جلب الأسهم الحرة لأفضل N نتيجة فقط
UPCOMING = 5       # نافذة تبدأ خلال N شموع = "قادمة"
WEEKMASK = "Mon Tue Wed Thu Fri"   # أيام التداول (السعودي: الأحد–الخميس)


# ---------- قوائم الأسهم ----------
NAME_EXCLUDE = re.compile(r"\b(Warrants?|Units?|Rights?|Preferred|Notes? due|Depositary Shares? Representing|Debentures?)\b", re.I)


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ptc-scanner"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def us_universe():
    syms = set()
    sources = [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", 0, 6, 3),   # sym, etf, test
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", 0, 4, 6),
    ]
    for url, i_sym, i_etf, i_test in sources:
        try:
            txt = fetch_text(url)
        except Exception as e:  # noqa
            print("تعذر تحميل قائمة الرموز:", url, e)
            continue
        for line in txt.splitlines()[1:]:
            p = line.split("|")
            if len(p) < 7 or line.startswith("File Creation"):
                continue
            sym, name = p[i_sym].strip(), p[1]
            if p[i_etf].strip() == "Y" or p[i_test].strip() == "Y":
                continue
            if not sym or "$" in sym or NAME_EXCLUDE.search(name):
                continue
            syms.add(sym.replace(".", "-"))
    return sorted(syms)


def read_list(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#")[0].strip()
        if s:
            out.append(s)
    return out


# ---------- تحميل البيانات ----------
def download(tickers, period="2y", chunk=100):
    import yfinance as yf
    frames = {}
    for k in range(0, len(tickers), chunk):
        part = tickers[k:k + chunk]
        for attempt in range(3):
            try:
                df = yf.download(part, period=period, interval="1d", auto_adjust=True, group_by="ticker",
                                 threads=True, progress=False)
                break
            except Exception as e:  # noqa
                print("محاولة فاشلة:", e)
                time.sleep(10 * (attempt + 1))
                df = None
        if df is None or df.empty:
            continue
        for t in part:
            try:
                d = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                d = d.dropna(subset=["Close"])
                if len(d) >= 60:
                    frames[t] = d[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            except Exception:
                pass
        print(f"تم {min(k + chunk, len(tickers))}/{len(tickers)}", flush=True)
        time.sleep(1.5)
    return frames


def get_float(t):
    """يرجع (العدد, المصدر): float = الأسهم الحرة الحقيقية، out = الأسهم القائمة (تقريبي)."""
    import yfinance as yf
    tk = yf.Ticker(t)
    out = None
    for attempt in range(3):
        try:
            info = tk.get_info()
            if info.get("floatShares"):
                return info["floatShares"], "float"
            out = out or info.get("sharesOutstanding")
            if info:
                break
        except Exception:
            time.sleep(3 * (attempt + 1))
    if out:
        return out, "out"
    try:  # بديل أخف: عدد الأسهم القائمة
        v = tk.fast_info.get("shares")
        return (v, "out") if v else (None, None)
    except Exception:
        return None, None


def tv_floats(market):
    """الأسهم الحرة من TradingView نفسه (نفس رقم المؤشر على الشارت) بطلب واحد لكل السوق."""
    url = "https://scanner.tradingview.com/" + ("ksa" if market == "sa" else "america") + "/scan"
    ex = ["TADAWUL"] if market == "sa" else ["NASDAQ", "NYSE", "AMEX"]
    body = {"columns": ["name", "float_shares_outstanding"], "range": [0, 20000],
            "filter": [{"left": "exchange", "operation": "in_range", "right": ex}]}
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                         headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())["data"]
            out = {}
            for row in data:
                name, fl = row["d"][0], row["d"][1]
                if fl:
                    out[f"{name}.SR" if market == "sa" else name.replace(".", "-")] = float(fl)
            print("الأسهم الحرة من TradingView:", len(out))
            return out
        except Exception as e:  # noqa
            print("تعذر جلب الأسهم الحرة من TradingView:", e)
            time.sleep(5 * (attempt + 1))
    return {}


# ---------- التحليل ----------
def pivots(arr, is_low):
    idx = []
    n = len(arr)
    for i in range(PIV, n - PIV):
        w = arr[i - PIV:i + PIV + 1]
        if (is_low and arr[i] == w.min()) or (not is_low and arr[i] == w.max()):
            if not idx or i - idx[-1] > 1:
                idx.append(i)
    return idx


def gap_stats(idx):
    if len(idx) < 3:
        return None, None
    k = min(CYC_N, len(idx) - 1)
    g = np.diff(idx[-(k + 1):])
    return float(g.mean()), float(g.std())


def rhythm_test(idx):
    """كم قاعاً جاء داخل النافذة التي توقعها الإيقاع، مقارنة بالصدفة."""
    hit = n = 0
    ch = 0.0
    for j in range(3, len(idx)):
        m, _ = gap_stats(idx[:j])
        if not m:
            continue
        tol = max(MIN_TOL, round(m * TOL_PCT / 100))
        pred = idx[j - 1] + m
        n += 1
        hit += abs(idx[j] - pred) <= tol
        ch += min(1.0, (2 * tol + 1) / m)
    return hit, n, (ch / n * 100 if n else None)


def next_window(last, m, cur):
    if m is None or last is None or m <= 0:
        return None, None
    tol = max(MIN_TOL, round(m * TOL_PCT / 100))
    c = last + m
    for _ in range(6):
        if c + tol < cur:
            c += m
    return c, tol


def liquidity_targets(h, l, c, v):
    n = min(VP_LEN, len(c))
    h, l, v = h[-n:], l[-n:], v[-n:]
    hi, lo = h.max(), l.min()
    step = (hi - lo) / VP_BINS
    if step <= 0:
        return None, None, False, False, []
    vols = np.zeros(VP_BINS)
    for i in range(n):
        f = int(min(VP_BINS - 1, max(0, math.floor((l[i] - lo) / step))))
        t = int(min(VP_BINS - 1, math.floor((h[i] - lo) / step)))
        vols[f:t + 1] += v[i] / (t - f + 1)
    avg = vols.mean()
    cur = int(min(VP_BINS - 1, max(0, math.floor((c[-1] - lo) / step))))
    mid = lambda k: lo + (k + 0.5) * step  # noqa
    nodes = [mid(k) for k in range(VP_BINS) if vols[k] >= avg * HVN_MULT]

    def peak(k):
        pv = vols[k - 1] if k > 0 else 0
        nx = vols[k + 1] if k < VP_BINS - 1 else 0
        return vols[k] >= pv and vols[k] >= nx

    up = dn = None
    upw = dnw = False
    for k in range(cur + 1, VP_BINS):
        if vols[k] >= avg * HVN_MULT and peak(k):
            up = mid(k)
            break
    if up is None:
        cand = [k for k in range(cur + 1, VP_BINS) if peak(k) and vols[k] >= avg * 0.7]
        if cand:
            up, upw = mid(max(cand, key=lambda k: vols[k])), True
    for k in range(cur - 1, -1, -1):
        if vols[k] >= avg * HVN_MULT and peak(k):
            dn = mid(k)
            break
    if dn is None:
        cand = [k for k in range(cur - 1, -1, -1) if peak(k) and vols[k] >= avg * 0.7]
        if cand:
            dn, dnw = mid(max(cand, key=lambda k: vols[k])), True
    return up, dn, upw, dnw, nodes


def analyze(t, d, bench, float_shares=None, float_src=None):
    o, h, l, c, v = (d[k].values for k in ["Open", "High", "Low", "Close", "Volume"])
    n = len(c)
    cur = n - 1
    vavg20 = v[-20:].mean()
    res = {"t": t, "price": round(float(c[-1]), 2), "chg": round(float((c[-1] / c[-2] - 1) * 100), 2) if n > 1 else 0.0}

    # القمم والقيعان
    lo_i, hi_i = pivots(l, True), pivots(h, False)
    m_lo, sd_lo = gap_stats(lo_i)
    m_hi, _ = gap_stats(hi_i)
    last_lo = lo_i[-1] if lo_i else None
    last_hi = hi_i[-1] if hi_i else None

    # 1) بداية عدّ الحجم: قرب قاع السنة أو قاع السنة حديث
    w = min(FT_LEN, n)
    low52 = l[-w:].min()
    off_low = int((w - 1) - int(np.where(l[-w:] == low52)[0][-1]))
    near_low = c[-1] <= low52 * 1.10 or off_low <= 20
    res["low52"] = round(float(low52), 2)
    res["lowAgo"] = off_low                 # كم شمعة مضت على شمعة بداية عدّ الحجم
    res["atStart"] = bool(off_low <= 3)     # السهم عند شمعة بداية العد (قاع سنة جديد)
    res["fromLow"] = round(float((c[-1] / low52 - 1) * 100), 1)

    # 2) نافذة الإيقاع
    c_lo, tol_lo = next_window(last_lo, m_lo, cur)
    in_lo = c_lo is not None and abs(cur - c_lo) <= tol_lo
    soon_lo = c_lo is not None and not in_lo and 0 < (c_lo - tol_lo) - cur <= UPCOMING
    c_hi, tol_hi = next_window(last_hi, m_hi, cur)

    # 3) فيبو الزمني (بعد موجة صاعدة = نافذة نهاية التصحيح)
    in_fib, fib_r, soon_fib = False, None, False
    up_leg = last_lo is not None and last_hi is not None and last_hi > last_lo
    if up_leg:
        T = last_hi - last_lo
        if T >= MIN_LEG:
            tf = max(MIN_TOL, round(T * 0.1))
            for r in (0.618, 1.0, 1.618):
                cc = last_hi + r * T
                if abs(cur - cc) <= tf:
                    in_fib, fib_r = True, r
                elif 0 < (cc - tf) - cur <= UPCOMING:
                    soon_fib = True

    # 4) دورة الأسهم الحرة
    ft_bars, ft_pct, in_ft, ft_since = None, None, False, None
    if float_shares:
        # نفس منطق المؤشر: عند التساوي نأخذ الأقدم (مثل <= و >= في Pine)
        start = w - 1 - int(np.where(l[-w:] == low52)[0][0])
        if start < 10:  # القاع حديث: العد من القمة
            hi52 = h[-w:].max()
            start = w - 1 - int(np.where(h[-w:] == hi52)[0][0])
        seg = v[-(start + 1):]
        cums = np.cumsum(seg)
        cum = cums[-1]
        ft_pct = cum / float_shares * 100
        mult = math.floor(ft_pct / 100) + 1
        ft_bars = math.ceil((mult * float_shares - cum) / vavg20) if vavg20 > 0 else None
        if mult >= 2:  # كم شمعة مضت على عبور آخر مضاعف (100%، 200%...)
            k = (mult - 1) * float_shares
            ft_since = int(len(cums) - 1 - int(np.argmax(cums >= k)))
        in_ft = (ft_bars is not None and ft_bars <= 3) or (ft_since is not None and ft_since <= 3)
        if ft_bars is not None and ft_bars > 252:
            ft_bars = None  # أبعد من سنة

    # 5) السعر عند دعم
    tgt_up, tgt_dn, upw, dnw, nodes = liquidity_targets(h, l, c, v)
    levels = []
    if up_leg:
        H, L = h[last_hi], l[last_lo]
        for r in (0.382, 0.5, 0.618, 0.786):
            levels.append((H - (H - L) * r, f"فيبو {r}"))
    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().values[-1] if n >= 200 else None
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values[-1]
    ema_ref, ema_nm = (ema200, "EMA200") if ema200 is not None else (ema50, "EMA50")
    levels.append((ema_ref, ema_nm))
    for j in lo_i[-3:]:
        levels.append((l[j], "قاع سابق"))
    for p in nodes:
        levels.append((p, "سيولة"))
    sup, sup_nm = None, []
    for L_, nm in levels:
        if L_ and L_ <= c[-1] * (1 + PR_TOL) and l[-1] <= L_ * (1 + PR_TOL) and c[-1] >= L_ * (1 - PR_TOL):
            if sup is None or abs(c[-1] - L_) < abs(c[-1] - sup):
                sup = L_
            if nm not in sup_nm:
                sup_nm.append(nm)
    at_sup = sup is not None

    # 6) انفجار الحجم
    vr = v[-1] / vavg20 if vavg20 > 0 else 0
    trig = vr >= VOL_MULT and c[-1] > o[-1] and c[-1] > c[-2]

    # انفجار أسهم حرة قليلة: سهم كان هادئاً ثم تداول فجأة كل أسهمه الحرة (أو 3x حجمه)
    turn_today = turn_prior = None
    burst_days, low_burst = 0, False
    if float_shares:
        k = 0
        while k < min(10, n - 22):
            base_k = v[-(21 + k):-(1 + k)].mean()
            if v[-1 - k] >= float_shares or (base_k > 0 and v[-1 - k] >= 3 * base_k):
                k += 1
            else:
                break
        burst_days = k                                   # عدد أيام الانفجار المتتالية حتى اليوم
        base = v[-(20 + k):-k].mean() if k > 0 else v[-21:-1].mean()
        turn_today = v[-1] / float_shares * 100
        turn_prior = base / float_shares * 100          # متوسط التداول اليومي قبل الانفجار
        rng = h[-1] - l[-1]
        green_hi = c[-1] > o[-1] and c[-1] > c[-2] and rng > 0 and (h[-1] - c[-1]) / rng <= 0.3
        low_burst = bool(float_shares < LOW_FLOAT and burst_days >= 1 and turn_prior <= 20 and green_hi)

    # ===== خطة الغد: سلّم المستويات + التفعيل + الوقف + الأهداف =====
    ladder = {}
    for nm, N in (("3D", 3), ("5D", 5), ("10D", 10), ("20D", 20), ("52W", 252)):
        if n >= N:
            ladder[nm] = float(h[-N:].max())
    order = ["3D", "5D", "10D", "20D", "52W"]
    above = [(val, nm) for val, nm in sorted((ladder[k], k) for k in ladder if ladder[k] > c[-1] * 1.001)]
    above.sort(key=lambda x: (round(x[0], 6), order.index(x[1])))  # عند التساوي: الأقصر مدة
    trig, trig_nm = above[0] if above else (float(h[-1]), "قمة اليوم")
    stop4, stop8 = trig * 0.96, trig * 0.92
    nxt = [(val, nm) for val, nm in above if val > trig * 1.02]
    t1, t1_nm = nxt[0] if nxt else ((tgt_up, "هدف السيولة") if tgt_up and tgt_up > trig * 1.02 else (None, None))
    t2 = tgt_up if (tgt_up and t1 and tgt_up > t1 * 1.01) else None
    rr = (t1 - trig) / (trig - stop4) if t1 else None
    hi52 = ladder.get("52W")
    under_major = bool(hi52 and c[-1] < hi52 <= c[-1] * 1.05)
    cs = pd.Series(c)
    e6, e20, e50 = (cs.ewm(span=k, adjust=False).mean().values for k in (6, 20, 50))
    e200v = cs.ewm(span=200, adjust=False).mean().values[-1] if n >= 200 else None
    ema_stack = bool(e200v is not None and e6[-1] > e20[-1] > e50[-1] > e200v)
    macd = cs.ewm(span=12, adjust=False).mean().values - cs.ewm(span=26, adjust=False).mean().values
    msig = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    macd_pos = bool(macd[-1] > msig[-1])
    # المراحل الخمس (نفس المؤشر)
    brk_today = bool(n > 6 and c[-1] > h[-6:-1].max() and vr >= 1.2 and macd_pos)
    stages = [
        ("قاعدة", bool(c[-1] <= low52 * 1.3)),
        ("انكماش", None),  # يُحسب بعد حساب الانكماش أدناه
        ("دورة قريبة", bool((ft_bars is not None and 1 <= ft_bars <= 8) or (ft_since is not None and ft_since <= 1))),
        ("اختراق بحجم", brk_today),
        ("تأكيد", bool(c[-1] > e20[-1] and macd_pos)),
    ]
    # اختبار صادق لطريقة الاختراق: إغلاق فوق أعلى 5 أيام + حجم ≥1.2x + MACD إيجابي → هل لمس +4% قبل -4% خلال 10 أيام؟
    bo_win = bo_n = 0
    hi5 = pd.Series(h).rolling(5).max().shift(1).values
    va20 = pd.Series(v).rolling(20).mean().shift(1).values
    for i in range(60, n - 1):
        if not (c[i] > hi5[i] and va20[i] > 0 and v[i] >= 1.2 * va20[i] and macd[i] > msig[i]):
            continue
        ent, res_ = c[i], None
        for j in range(i + 1, min(n, i + 11)):
            if l[j] <= ent * 0.96:
                res_ = False
                break
            if h[j] >= ent * 1.04:
                res_ = True
                break
        if res_ is not None:
            bo_n += 1
            bo_win += int(res_)

    # إضافات
    rs = None
    if isinstance(bench, float):  # مرجع بديل: متوسط أداء أسهم السوق نفسه
        k = min(RS_LEN, n - 1)
        rs = (c[-1] / c[-1 - k] - 1) - bench
    elif bench is not None and len(bench) > RS_LEN:
        b = bench.reindex(d.index).ffill().values
        k = min(RS_LEN, n - 1)
        if b[-1 - k] and b[-1]:
            rs = (c[-1] / c[-1 - k] - 1) - (b[-1] / b[-1 - k] - 1)
    tl = l[-TIGHT_LEN:].min()
    tight = (h[-TIGHT_LEN:].max() - tl) / tl * 100 if tl > 0 else None
    hit, nn, chance = rhythm_test(lo_i)
    stages[1] = ("انكماش", bool(tight is not None and tight <= TIGHT_PCT))

    score = int(near_low) + int(in_lo) + int(in_fib) + int(in_ft) + int(at_sup) + int(trig)
    dt = d.index[-1]
    def date_at(bars):  # بعد N يوم تداول، مع تخطي عطلة نهاية الأسبوع
        d0 = np.datetime64(dt.date(), "D")
        return pd.Timestamp(np.busday_offset(d0, int(round(bars)), roll="forward", weekmask=WEEKMASK)).strftime("%d %b")

    res.update({
        "score": score,
        "nearLow": bool(near_low), "inLo": bool(in_lo), "inFib": bool(in_fib), "inFt": bool(in_ft),
        "atSup": bool(at_sup), "trig": bool(trig),
        "soon": bool(soon_lo or soon_fib),
        "fibR": fib_r,
        "loWin": None if c_lo is None else ("الآن" if in_lo else date_at(c_lo - cur)),
        "rhythm": None if m_lo is None else round(m_lo),
        "hiWin": None if c_hi is None else date_at(c_hi - cur),
        "ftBars": ft_bars, "ftPct": None if ft_pct is None else round(ft_pct),
        "ftDate": None if ft_bars is None else date_at(ft_bars),
        "ftSince": ft_since,
        "ftFloat": None if not float_shares else round(float_shares / 1e6, 1),
        "ftSrc": float_src,
        "sup": None if sup is None else round(float(sup), 2), "supNm": " + ".join(sup_nm),
        "up": None if tgt_up is None else round(float(tgt_up), 2), "upWeak": upw,
        "dn": None if tgt_dn is None else round(float(tgt_dn), 2), "dnWeak": dnw,
        "rs": None if rs is None else round(rs * 100, 1),
        "tight": None if tight is None else round(float(tight), 1),
        "vr": round(float(vr), 2),
        "turnToday": None if turn_today is None else round(float(turn_today)),
        "turnPrior": None if turn_prior is None else round(float(turn_prior), 1),
        "burstDays": burst_days, "lowBurst": low_burst,
        "plan": {"trig": round(trig, 2), "trigNm": trig_nm, "stop4": round(stop4, 2), "stop8": round(stop8, 2),
                 "t1": None if t1 is None else round(float(t1), 2), "t1Nm": t1_nm,
                 "t2": None if t2 is None else round(float(t2), 2), "rr": None if rr is None else round(rr, 1)},
        "ladder": {k: round(val, 2) for k, val in ladder.items()},
        "underMajor": under_major, "emaStack": ema_stack, "macdPos": macd_pos,
        "stages": [ok for _, ok in stages], "miss": [nm for nm, ok in stages if not ok],
        "boWin": bo_win, "boN": bo_n,
        "rhHit": hit, "rhN": nn, "rhChance": None if chance is None else round(chance),
        "date": dt.strftime("%Y-%m-%d"),
    })
    return res


# ---------- التشغيل ----------
def run(market):
    global WEEKMASK
    WEEKMASK = "Sun Mon Tue Wed Thu" if market == "sa" else "Mon Tue Wed Thu Fri"
    names = {}
    if market == "sa":
        for line in read_list(ROOT / "saudi.txt"):
            code, _, nm = line.partition(" ")
            names[f"{code}.SR"] = nm.strip()
        tickers = list(names)
        bench_sym = "^TASI.SR"
    else:
        tickers = us_universe()
        bench_sym = "QQQ"
    watch = [w for w in read_list(ROOT / "watchlist.txt") if (w.endswith(".SR") == (market == "sa"))]
    tickers = sorted(set(tickers) | set(watch))
    print(f"عدد الرموز: {len(tickers)}")

    data = download(tickers + [bench_sym])
    bench = data.pop(bench_sym, None)
    bench = bench["Close"] if bench is not None else None
    if bench is None or len(bench) <= RS_LEN:
        rets = [d["Close"].iloc[-1] / d["Close"].iloc[-1 - min(RS_LEN, len(d) - 1)] - 1 for d in data.values() if len(d) > 2]
        bench = float(np.median(rets)) if rets else None
        print("المؤشر المرجعي غير متوفر، استخدام وسيط أداء السوق:", bench)

    rows, low_items = [], []
    for t, d in data.items():
        try:
            px, av = d["Close"].iloc[-1], d["Volume"].iloc[-20:].mean()
            liquid = px >= MIN_PRICE and av >= MIN_AVGVOL[market]
            if liquid or t in watch:
                rows.append((t, d, analyze(t, d, bench)))
            elif LOW_PRICE <= px < MIN_PRICE and (av >= LOW_AVGVOL[market] or d["Volume"].iloc[-1] >= 10 * LOW_AVGVOL[market]):
                low_items.append((t, d))
        except Exception as e:  # noqa
            print("خطأ في", t, e)


    # الأسهم الحرة: من TradingView لكل الأسهم (مطابقة للمؤشر)، وياهو احتياطاً فقط إذا تعذر TradingView
    rows.sort(key=lambda x: (x[2]["score"], x[2]["soon"]), reverse=True)
    tvf = tv_floats(market)
    for i, (t, d, r) in enumerate(rows):
        if tvf:
            fs, src = tvf.get(t), "tv"
        elif i < FLOAT_TOP or t in watch or r["atStart"]:
            fs, src = get_float(t)
            time.sleep(0.4)
        else:
            fs = None
        if fs:
            try:
                rows[i] = (t, d, analyze(t, d, bench, fs, src))
            except Exception:
                pass
    results = [r for _, _, r in rows]
    for r in results:
        r["watch"] = r["t"] in watch
        if names.get(r["t"]):
            r["name"] = names[r["t"]]
    results.sort(key=lambda r: (r["score"], r["soon"], r["watch"]), reverse=True)
    top = [r for r in results if not r["watch"] and (r["score"] >= 2 or r["soon"])][:400]
    ids = {r["t"] for r in top}
    start = [r for r in results if r["atStart"] and not r["watch"] and r["t"] not in ids]  # عند بداية عدّ الحجم تظهر دائماً
    ids |= {r["t"] for r in start}
    near = [r for r in results if not r["watch"] and r["t"] not in ids and r.get("ftSrc") != "out"
            and r["ftBars"] is not None and r["ftBars"] <= 8]  # دورة قريبة تظهر دائماً (لاصطياد ما قبل الانطلاق)
    keep = top + start + near + [r for r in results if r["watch"]]  # قائمة المراقبة تظهر دائماً
    keep.sort(key=lambda r: (r["score"], r["soon"], r["watch"]), reverse=True)
    print("الأسهم الحرة متوفرة لـ", sum(1 for r in results if r.get("ftSrc") in ("tv", "float")), "سهم، وتقريبية (القائمة) لـ",
          sum(1 for r in results if r.get("ftSrc") == "out"))

    # ===== تبويب "أقل من 5$" =====
    low = []
    for t, d in low_items:
        fs = tvf.get(t) if tvf else None
        try:
            r = analyze(t, d, bench, fs, "tv" if fs else None)
        except Exception:
            continue
        if names.get(t):
            r["name"] = names[t]
        r["watch"] = False
        cyc_ok = (r["ftBars"] is not None and 1 <= r["ftBars"] <= 5) or (r["ftSince"] is not None and r["ftSince"] <= 1)
        r["burstCyc"] = bool(r["lowBurst"] and cyc_ok)
        low.append(r)
    low.sort(key=lambda r: (r["burstCyc"], r["lowBurst"], r["score"], r["turnToday"] or 0), reverse=True)
    low_keep = [r for r in low if r["lowBurst"] or r["atStart"] or r["score"] >= 2 or r["soon"]
                or (r["ftBars"] is not None and r["ftBars"] <= 8)][:500]
    (DOCS / f"{market}_low.json").parent.mkdir(exist_ok=True)
    (DOCS / f"{market}_low.json").write_text(json.dumps({
        "market": market, "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "scanned": len(low), "rows": low_keep}, ensure_ascii=False), encoding="utf-8")
    print(f"أقل من 5$: {len(low)} سهم، انفجار: {sum(r['lowBurst'] for r in low)}، انفجار + دورة: {sum(r['burstCyc'] for r in low)}")

    out = {
        "market": market,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "scanned": len(results),
        "rows": keep,
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / f"{market}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"تم: {len(results)} سهم محلل، {len(keep)} في النتائج")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "sa"], required=True)
    run(ap.parse_args().market)
