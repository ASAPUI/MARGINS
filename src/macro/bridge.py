# src/macro/bridge.py
"""
author:Essabri ali rayan
version:1.3
add macro data to match the unstabability of the world
"""

import os
import asyncio
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from cachetools import TTLCache
import httpx

from .signals import MacroSignal, AnomalyEvent

logger = logging.getLogger(__name__)


# ── Tier 3: Static baseline (always available) ─────────────────────────────────
BASELINE_CII: Dict[str, float] = {
    "Syria": 95, "Yemen": 93, "Somalia": 91, "Sudan": 88, "DRC": 86,
    "CAR": 84, "Mali": 82, "Chad": 80, "Niger": 78, "Ethiopia": 75,
    "Myanmar": 74, "Haiti": 73, "Afghanistan": 72, "Iraq": 70,
    "Libya": 69, "Venezuela": 67, "Ukraine": 65, "Lebanon": 64,
    "Pakistan": 62, "Nigeria": 60, "Mozambique": 58, "Cuba": 58,
    "Iran": 56, "North Korea": 55, "Israel": 55, "Russia": 50,
    "Palestine": 78, "Zimbabwe": 65, "Belarus": 52, "Armenia": 48,
    "Azerbaijan": 46, "Egypt": 47, "Turkey": 42, "Colombia": 45,
    "Mexico": 48, "Brazil": 40, "India": 38, "China": 35,
    "Saudi Arabia": 38, "Serbia": 35, "Argentina": 42,
    "United States": 25, "France": 22, "Italy": 24, "Spain": 22,
    "UK": 20, "South Korea": 25, "Japan": 15, "Germany": 18,
    "Canada": 14, "Australia": 14,
}

# GDELT country name → our baseline key mapping
GDELT_COUNTRY_MAP = {
    "united states": "United States", "russia": "Russia", "china": "China",
    "ukraine": "Ukraine", "israel": "Israel", "iran": "Iran",
    "iraq": "Iraq", "syria": "Syria", "yemen": "Yemen",
    "afghanistan": "Afghanistan", "pakistan": "Pakistan",
    "north korea": "North Korea", "nigeria": "Nigeria",
    "ethiopia": "Ethiopia", "myanmar": "Myanmar", "sudan": "Sudan",
    "somalia": "Somalia", "libya": "Libya", "venezuela": "Venezuela",
    "mexico": "Mexico", "brazil": "Brazil", "india": "India",
    "turkey": "Turkey", "egypt": "Egypt", "lebanon": "Lebanon",
    "france": "France", "germany": "Germany", "united kingdom": "UK",
    "uk": "UK", "italy": "Italy", "spain": "Spain",
}


@dataclass
class BridgeConfig:
    wm_base: str = "https://api.worldmonitor.app"
    fred_key: Optional[str] = None
    risk_ttl: int = 300
    brief_ttl: int = 600
    timeout: float = 8.0
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        return cls(
            wm_base=os.getenv("WORLDMONITOR_API_URL", "https://api.worldmonitor.app"),
            fred_key=os.getenv("FRED_API_KEY"),
            risk_ttl=int(os.getenv("WORLDMONITOR_RISK_TTL", "300")),
            brief_ttl=int(os.getenv("WORLDMONITOR_BRIEF_TTL", "600")),
            timeout=float(os.getenv("WORLDMONITOR_TIMEOUT", "8.0")),
            enabled=os.getenv("WORLDMONITOR_ENABLED", "true").lower() == "true",
        )


class MacroBridge:
    """
    Three-tier macro intelligence bridge.

    WorldMonitor (primary) → FRED+GDELT (enrichment) → Static baseline (fallback)
    """

    # WorldMonitor versioned endpoint paths (from source repo /api/ directory)
    WM_ENDPOINTS = {
        "conflict":      "/conflict/v1",
        "intelligence":  "/intelligence/v1",
        "economic":      "/economic/v1",
        "market":        "/market/v1",
        "risk":          "/api/risk-scores",    # legacy flat endpoint
    }

    # GDELT GKG — no auth, free
    GDELT_GKG_URL = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        "?query=conflict%20OR%20war%20OR%20protest%20OR%20crisis%20OR%20military"
        "&mode=artlist&maxrecords=250&timespan=1440&format=json"
    )

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig.from_env()
        self._cache: TTLCache = TTLCache(maxsize=20, ttl=self.config.risk_ttl)
        self._brief_cache: TTLCache = TTLCache(maxsize=1, ttl=self.config.brief_ttl)
        self._last_successful: Optional[datetime] = None
        self._lock: Optional[asyncio.Lock] = None
        self._wm_available: Optional[bool] = None  # None = untested

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    async def _get(self, url: str, params: Dict = None, label: str = "") -> Optional[Any]:
        """GET with timeout, redirect follow, JSON content-type guard."""
        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout,
                follow_redirects=True,
                headers={"Accept": "application/json",
                         "User-Agent": "GoldOption-MacroBridge/1.0"}
            ) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "json" not in ct:
                    logger.debug(f"{label}: non-JSON response ({ct})")
                    return None
                return resp.json()
        except Exception as e:
            logger.debug(f"{label} fetch error: {e}")
            return None

    # ── Tier 1: WorldMonitor ───────────────────────────────────────────────────

    async def _fetch_wm_conflict(self) -> Dict[str, float]:
        """
        Pull conflict/v1 from WorldMonitor.
        Expected shape (from source repo): list of conflict events with country + severity.
        Returns country → conflict_score (0-100).
        """
        cached = self._cache.get("wm_conflict")
        if cached is not None:
            return cached

        data = await self._get(
            f"{self.config.wm_base}{self.WM_ENDPOINTS['conflict']}",
            label="WM conflict/v1"
        )
        if not data:
            return {}

        scores: Dict[str, float] = {}
        # Handle list of conflict events
        events = data if isinstance(data, list) else data.get("events", data.get("data", []))
        for ev in events:
            if not isinstance(ev, dict):
                continue
            country = ev.get("country") or ev.get("location", {}).get("country", "")
            severity = ev.get("severity") or ev.get("score") or ev.get("intensity", 0)
            if country and severity is not None:
                # Accumulate: multiple events in same country → higher score
                prev = scores.get(country, 0)
                scores[country] = min(100, prev + float(severity) * 10)

        if scores:
            self._cache["wm_conflict"] = scores
            self._wm_available = True
            self._last_successful = datetime.utcnow()
            logger.info(f"WorldMonitor conflict/v1: {len(scores)} countries")

        return scores

    async def _fetch_wm_intelligence(self) -> Dict[str, Any]:
        """Pull intelligence/v1 — hotspot and anomaly data."""
        cached = self._cache.get("wm_intel")
        if cached is not None:
            return cached

        data = await self._get(
            f"{self.config.wm_base}{self.WM_ENDPOINTS['intelligence']}",
            label="WM intelligence/v1"
        )
        if not data:
            return {}

        self._cache["wm_intel"] = data
        self._wm_available = True
        return data

    async def _fetch_wm_risk_scores(self) -> Dict[str, float]:
        """Pull legacy /api/risk-scores endpoint."""
        cached = self._cache.get("wm_risk")
        if cached is not None:
            return cached

        data = await self._get(
            f"{self.config.wm_base}{self.WM_ENDPOINTS['risk']}",
            label="WM /api/risk-scores"
        )
        if not data:
            return {}

        scores: Dict[str, float] = {}
        # Flat dict shape: {"Syria": 92.3, ...}
        if isinstance(data, dict):
            raw = data.get("scores") or data.get("countries") or data.get("data") or data
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, (int, float)):
                        scores[k] = float(v)
                    elif isinstance(v, dict):
                        s = v.get("score") or v.get("cii") or v.get("value")
                        if s is not None:
                            scores[k] = float(s)
            elif isinstance(raw, list):
                for item in raw:
                    c = item.get("country") or item.get("code")
                    s = item.get("score") or item.get("cii")
                    if c and s is not None:
                        scores[c] = float(s)
        elif isinstance(data, list):
            for item in data:
                c = item.get("country") or item.get("code")
                s = item.get("score") or item.get("cii")
                if c and s is not None:
                    scores[c] = float(s)

        if scores:
            self._cache["wm_risk"] = scores
            self._wm_available = True
            self._last_successful = datetime.utcnow()
            logger.info(f"WorldMonitor risk-scores: {len(scores)} countries")

        return scores

    async def _try_worldmonitor(self) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Attempt all WorldMonitor endpoints concurrently.
        Returns (cii_scores, anomaly_list).
        """
        conflict_task = self._fetch_wm_conflict()
        risk_task = self._fetch_wm_risk_scores()
        intel_task = self._fetch_wm_intelligence()

        conflict, risk, intel = await asyncio.gather(
            conflict_task, risk_task, intel_task, return_exceptions=True
        )

        if isinstance(conflict, Exception):
            conflict = {}
        if isinstance(risk, Exception):
            risk = {}
        if isinstance(intel, Exception):
            intel = {}

        # Merge: risk-scores takes priority, enrich with conflict signals
        merged: Dict[str, float] = dict(risk)
        for country, score in conflict.items():
            if country not in merged:
                merged[country] = score
            else:
                # Blend: 70% CII score + 30% conflict signal
                merged[country] = merged[country] * 0.7 + score * 0.3

        # Extract anomalies from intelligence endpoint
        anomalies: List[Dict] = []
        if isinstance(intel, dict):
            raw_anomalies = (intel.get("anomalies") or
                             intel.get("hotspots") or
                             intel.get("alerts") or [])
            for a in raw_anomalies:
                if isinstance(a, dict):
                    anomalies.append({
                        "region": a.get("country") or a.get("region", "unknown"),
                        "type": a.get("type", "instability"),
                        "zscore": float(a.get("zscore") or a.get("score", 2.0)),
                        "timestamp": datetime.utcnow().isoformat(),
                    })

        return merged, anomalies

    # ── Tier 2: GDELT GKG ─────────────────────────────────────────────────────

    async def _fetch_gdelt_scores(self) -> Dict[str, float]:
        """
        GDELT GKG: article-level tone and conflict mentions per country.
        Returns country → velocity_score (0-100), enriches baseline.
        """
        cached = self._cache.get("gdelt")
        if cached is not None:
            return cached

        data = await self._get(self.GDELT_GKG_URL, label="GDELT GKG")
        if not data:
            return {}

        mentions: Dict[str, int] = {}
        for art in data.get("articles", []):
            src = art.get("sourcecountry", "").lower().strip()
            matched = GDELT_COUNTRY_MAP.get(src)
            if matched:
                mentions[matched] = mentions.get(matched, 0) + 1

        if not mentions:
            return {}

        max_m = max(mentions.values(), default=1)
        scores = {
            c: math.log1p(n) / math.log1p(max_m) * 100
            for c, n in mentions.items()
        }

        self._cache["gdelt"] = scores
        logger.info(f"GDELT GKG: {len(scores)} countries with mention data")
        return scores

    async def _fetch_fred_gold_vol(self) -> Optional[float]:
        """
        FRED: Gold volatility proxy via GVZ index or VIX if FRED key set.
        Returns annualised vol as a float (e.g. 18.5 = 18.5%).
        """
        if not self.config.fred_key:
            return None

        data = await self._get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "GVZCLS",  # CBOE Gold ETF Volatility Index
                "api_key": self.config.fred_key,
                "file_type": "json",
                "limit": "1",
                "sort_order": "desc",
            },
            label="FRED GVZ"
        )
        if not data:
            return None
        try:
            val = data["observations"][0]["value"]
            return float(val) if val != "." else None
        except Exception:
            return None
    async def _fetch_fred_real_rate(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Fetch 10Y real rate from FRED (DFII10) or compute as DGS10 - T5YIE.
        Returns (current_rate, delta_30d) where delta is change over 30 days (annualized).
        """
        if not self.config.fred_key:
            return None, None
        
        try:
            from fredapi import Fred
            fred = Fred(api_key=self.config.fred_key)
            
            # Try DFII10 (10Y TIPS yield = real rate) first
            try:
                series = fred.get_series('DFII10', observation_start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
                if series is not None and len(series) >= 2:
                    current = float(series.iloc[-1]) / 100  # Convert % to decimal
                    prev_30d = float(series.iloc[-30]) / 100 if len(series) >= 30 else float(series.iloc[0]) / 100
                    delta = current - prev_30d  # Already annualized
                    return current, delta
            except Exception:
                pass
            
            # Fallback: DGS10 - T5YIE (nominal - inflation = real)
            try:
                nominal = fred.get_series('DGS10', observation_start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
                breakeven = fred.get_series('T5YIE', observation_start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
                
                if nominal is not None and breakeven is not None and len(nominal) >= 2 and len(breakeven) >= 2:
                    real_current = (float(nominal.iloc[-1]) - float(breakeven.iloc[-1])) / 100
                    real_prev = (float(nominal.iloc[-30]) - float(breakeven.iloc[-30])) / 100 if len(nominal) >= 30 and len(breakeven) >= 30 else \
                                (float(nominal.iloc[0]) - float(breakeven.iloc[0])) / 100
                    delta = real_current - real_prev
                    return real_current, delta
            except Exception:
                pass
                
        except Exception as e:
            logger.debug(f"FRED real rate fetch error: {e}")
        
        return None, None

    async def _fetch_dxy(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Fetch DXY (Dollar Index) from yfinance.
        Returns (current_price, delta_30d_pct) where delta is % change over 30 days.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker("DX-Y.NYB")
            hist = ticker.history(period="3mo")
            
            if hist.empty or len(hist) < 30:
                return None, None
            
            current = float(hist["Close"].iloc[-1])
            price_30d_ago = float(hist["Close"].iloc[-30])
            delta_pct = (current - price_30d_ago) / price_30d_ago
            
            return current, delta_pct
            
        except Exception as e:
            logger.debug(f"DXY fetch error: {e}")
            return None, None
    # ── CII computation ────────────────────────────────────────────────────────

    def _build_cii(
        self,
        wm_scores: Dict[str, float],
        gdelt_scores: Dict[str, float],
        gold_vol: Optional[float]
    ) -> Dict[str, float]:
        """
        Merge all signals into final CII scores.

        Weights:
          baseline_risk  40%  (static fragility)
          wm_live        35%  (WorldMonitor if available, else 0)
          gdelt_velocity 20%  (GDELT mention frequency)
          vol_pressure    5%  (gold VIX as global risk proxy)
        """
        all_countries = set(BASELINE_CII) | set(wm_scores) | set(gdelt_scores)
        # Global vol boost: GVZ > 20 adds up to 5 points across all countries
        vol_boost = min(5.0, max(0.0, (gold_vol - 15) * 0.5)) if gold_vol else 0.0

        final: Dict[str, float] = {}
        for country in all_countries:
            baseline = BASELINE_CII.get(country, 30.0)
            wm = wm_scores.get(country)
            gdelt = gdelt_scores.get(country, 0.0)

            if wm is not None:
                # Full blend: all three tiers
                score = baseline * 0.40 + wm * 0.35 + gdelt * 0.20 + vol_boost
            else:
                # No WM data: baseline + GDELT only
                score = baseline * 0.70 + gdelt * 0.25 + vol_boost

            final[country] = round(min(100.0, max(0.0, score)), 1)

        return final

    def _compute_anomalies(self, cii: Dict[str, float]) -> List[Dict]:
        """
        Z-score anomaly detection — only flags countries with ELEVATED instability.
        Stable/low-scoring countries are intentionally excluded: a low CII score
        is not an anomaly worth surfacing in a gold risk dashboard.
        Threshold: z >= 1.5 AND raw score >= 55 (genuinely elevated, not just
        low relative to war zones).
        """
        if len(cii) < 5:
            return []
        vals = list(cii.values())
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) or 1.0

        anomalies = []
        for country, score in cii.items():
            z = (score - mean) / stdev
            # Only surface high-risk outliers — ignore stable country outliers
            if z >= 1.5 and score >= 55:
                anomalies.append({
                    "region": country,
                    "type": "instability",
                    "zscore": round(z, 2),
                    "timestamp": datetime.utcnow().isoformat(),
                })
        return sorted(anomalies, key=lambda x: x["zscore"], reverse=True)

    def _generate_brief(self, cii: Dict[str, float], anomalies: List[Dict]) -> str:
        """
        Generate a concise world brief from CII data — no external API needed.
        Summarises top risk countries, active anomalies, and gold risk implications.
        """
        if not cii:
            return ""

        sorted_countries = sorted(cii.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_countries[:5]
        top5_avg = sum(v for _, v in top5) / len(top5)

        critical = [(c, s) for c, s in sorted_countries if s >= 80]
        high      = [(c, s) for c, s in sorted_countries if 65 <= s < 80]

        risk_label = (
            "EXTREME" if top5_avg >= 80 else
            "HIGH"    if top5_avg >= 65 else
            "ELEVATED" if top5_avg >= 50 else
            "MODERATE"
        )

        lines = []
        lines.append(f"Global risk assessment: {risk_label} (CII Top-5 avg: {top5_avg:.1f}/100).")

        if critical:
            names = ", ".join(c for c, _ in critical[:4])
            lines.append(f"Critical instability zones: {names}.")

        if high:
            names = ", ".join(c for c, _ in high[:3])
            lines.append(f"High-risk watch list: {names}.")

        if anomalies:
            hot = anomalies[0]["region"]
            z   = anomalies[0]["zscore"]
            lines.append(
                f"Strongest signal: {hot} shows elevated instability "
                f"(z={z:.2f} above regional baseline)."
            )

        # Gold implication
        if top5_avg >= 70:
            lines.append(
                "Gold implication: Heightened geopolitical stress historically "
                "supports safe-haven demand. Merton jump probability elevated."
            )
        elif top5_avg >= 50:
            lines.append(
                "Gold implication: Moderate geopolitical risk — watch for "
                "escalation signals that could trigger safe-haven flows."
            )
        else:
            lines.append(
                "Gold implication: Subdued geopolitical risk environment. "
                "Gold likely to trade on macro/USD factors near-term."
            )

        return " ".join(lines)

    # ── Public interface ───────────────────────────────────────────────────────

    async def get_signals(self) -> MacroSignal:
        """
        Fetch and return MacroSignal using all available tiers.
        """
        if not self.config.enabled:
            return MacroSignal(cii_scores=dict(BASELINE_CII),
                             cii_top5_avg=50.0, cii_max=0.0, is_fallback=True)

        async with self._get_lock():
            try:
                # Run all tiers concurrently
                wm_task = self._try_worldmonitor()
                gdelt_task = self._fetch_gdelt_scores()
                fred_task = self._fetch_fred_gold_vol()
                real_rate_task = self._fetch_fred_real_rate()
                dxy_task = self._fetch_dxy()

                (wm_scores, wm_anomalies), gdelt_scores, gold_vol, real_rate_result, dxy_result = \
                    await asyncio.gather(wm_task, gdelt_task, fred_task, 
                                       real_rate_task, dxy_task,
                                       return_exceptions=True)

                if isinstance(wm_scores, Exception) or isinstance(wm_scores, tuple):
                    wm_scores, wm_anomalies = {}, []
                if isinstance(gdelt_scores, Exception):
                    gdelt_scores = {}
                if isinstance(gold_vol, Exception):
                    gold_vol = None
                
                # Extract real rate and DXY data with fallback to 0
                real_rate_delta = 0.0
                dxy_delta = 0.0
                
                if isinstance(real_rate_result, tuple) and len(real_rate_result) == 2:
                    _, real_rate_delta = real_rate_result
                    if real_rate_delta is None:
                        real_rate_delta = 0.0
                        
                if isinstance(dxy_result, tuple) and len(dxy_result) == 2:
                    _, dxy_delta = dxy_result
                    if dxy_delta is None:
                        dxy_delta = 0.0

                cii = self._build_cii(wm_scores, gdelt_scores, gold_vol)

                anomalies = wm_anomalies if wm_anomalies else self._compute_anomalies(cii)
                is_fallback = not wm_scores and not gdelt_scores

                self._cache["scores"] = cii

                if not is_fallback:
                    self._last_successful = datetime.utcnow()

                brief = self._generate_brief(cii, anomalies)
                signal = MacroSignal.from_raw_api_data(
                    risk_scores=cii,
                    anomalies=anomalies,
                    brief=brief or None,
                    is_fallback=is_fallback
                )
                
                # Set financial market indicators
                signal.real_rate_delta = real_rate_delta
                signal.dxy_delta = dxy_delta

                return signal

            except Exception as e:
                logger.error(f"get_signals critical error: {e}", exc_info=True)
                cii = dict(BASELINE_CII)
                anoms = self._compute_anomalies(cii)
                brief = self._generate_brief(cii, anoms)
                self._cache["scores"] = cii
                return MacroSignal.from_raw_api_data(
                    risk_scores=cii,
                    anomalies=anoms,
                    brief=brief or None,
                    is_fallback=True
                )

    async def get_brief(self) -> Optional[str]:
        """
        World brief — tries WorldMonitor intelligence/v1 first,
        falls back to locally generated brief from CII data.
        """
        cached = self._brief_cache.get("brief")
        if cached:
            return cached

        # Try WorldMonitor first
        data = await self._get(
            f"{self.config.wm_base}{self.WM_ENDPOINTS['intelligence']}",
            label="WM brief"
        )
        if isinstance(data, dict):
            wm_brief = (data.get("brief") or data.get("summary")
                        or data.get("digest") or data.get("text"))
            if wm_brief and isinstance(wm_brief, str) and len(wm_brief) > 20:
                self._brief_cache["brief"] = wm_brief
                return wm_brief

        # Fall back to locally generated brief from cached CII scores
        cii = self._cache.get("scores") or dict(BASELINE_CII)
        anoms = self._compute_anomalies(cii)
        local_brief = self._generate_brief(cii, anoms)
        if local_brief:
            self._brief_cache["brief"] = local_brief
            return local_brief
        return None

    def get_signals_sync(self):
        import concurrent.futures

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.get_signals())
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result()

    def get_brief_sync(self) -> Optional[str]:
        import concurrent.futures

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.get_brief())
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result()

    def is_healthy(self) -> bool:
        # Static baseline always available → never fully unhealthy
        if self._last_successful:
            return (datetime.utcnow() - self._last_successful) < timedelta(
                seconds=self.config.risk_ttl * 2
            )
        # Haven't fetched yet — return True so status shows Live on first load
        return True

    def get_source_status(self) -> Dict[str, str]:
        """Returns which tiers are active — shown in UI."""
        return {
            "WorldMonitor": "live" if self._cache.get("wm_risk") or
                                      self._cache.get("wm_conflict") else "offline",
            "GDELT":        "live" if self._cache.get("gdelt") else "pending",
            "FRED":         "live" if self.config.fred_key else "no key (optional)",
            "Baseline":     "always on",
        }