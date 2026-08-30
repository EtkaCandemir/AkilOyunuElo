"""Ince, stateless FastAPI sarmalayici — AO European Elo hesap cekirdegi icin.

Bu dosya `src/ao_elo/*` icindeki model kodunu HICBIR SEKILDE degistirmez,
SADECE cagirir (AGENTS.md: "Kullanici acikca production aktivasyonu
istemedikce production contract'i ... degistirilmez"). Sadece
`AOEuropeanEloConfig.active()` ve `DynamicEloConfig.calibrated_v2()` — yani
zaten aktif olan production davranisi — HTTP uzerinden erisilebilir kilinir.

akiloyunu-api (Laravel) tum kalicilik/state'i kendi tarafinda tutar (bkz. o
repodaki LIVE_DATA_INGESTION.md ve proje notlari); bu servis her istekte
gelen state'ten AYNI ANDA gecici bir `SeasonState` kurup hesabi yapar ve
sonucu doner — hicbir sey diskte/hafizada tutmaz.

Bilinen sinirlama: `qualifier_to_main_carry` aktif configde `1.0` (no-op)
oldugu icin sezon boyunca `qualification_participants`/`qualification_
carry_applied` setlerini TASIMAMANIN hesap sonucuna etkisi yoktur — bu
deger degisirse (config guncellenirse) bu sarmalayicinin da guncellenmesi
gerekir.

Calistirma:
    pip install -r requirements.txt
    uvicorn service.app:app --host 0.0.0.0 --port 8000

Health kontrolu: GET /health
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ao_elo.config import AOEuropeanEloConfig
from ao_elo.pipeline import compute_ao_first_elo
from ao_elo.dynamic import (
    DynamicEloConfig,
    MatchFixture,
    MatchInput,
    SeasonState,
    TeamRating,
    lock_prediction,
    update_match,
)

try:
    from ao_elo.production_prediction import ProductionPredictionService

    # Satiri append-only ledger'a yazilabilecek JSON-guvenli degerlere
    # cevirmek icin AYNI normalizasyonu (numpy/pandas skalarlarini native
    # Python'a cevirir, NaN'i None yapar) kullaniyoruz — prediction_ledger.py
    # zaten test edilmis, kopyalanip driftlenmesin diye dogrudan import edildi.
    from ao_elo.prediction_ledger import _normalize as _ledger_normalize

    _PRODUCTION_PREDICTION_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 - eksik ML bagimliligi servisi DUSURMEMELI
    ProductionPredictionService = None  # type: ignore[assignment]
    _ledger_normalize = None  # type: ignore[assignment]
    _PRODUCTION_PREDICTION_IMPORT_ERROR = f"IMPORT_FAILED:{type(exc).__name__}:{exc}"


# ---------------------------------------------------------------------------
# Aktif config'ler — AGENTS.md geregi burada ASLA ozellestirilmez/degistirilmez.
# ---------------------------------------------------------------------------
_AO_FIRST_CONFIG = AOEuropeanEloConfig.active()
_DYNAMIC_CONFIG = DynamicEloConfig.calibrated_v2()

_CONTRACT_PATH = _REPO_ROOT / "contracts" / "ao_european_elo_v2_production.json"

if ProductionPredictionService is not None:
    _PRODUCTION_PREDICTION_SERVICE = ProductionPredictionService.from_contract(
        _CONTRACT_PATH, allow_degraded_fallback=True
    )
else:
    _PRODUCTION_PREDICTION_SERVICE = None


# ---------------------------------------------------------------------------
# Laravel (football-data.org konvansiyonu) <-> AO ic sozlesmesi eslemeleri.
# Bu eslemeler AO'nun KENDI kodundaki sabit degerlerden (COMPETITIONS,
# STAGE_MULTIPLIERS, QUALIFICATION_ROUND_KEYS) turetildi.
# ---------------------------------------------------------------------------
_COMPETITION_TO_AO = {"CL": "UCL", "UEL": "UEL", "UECL": "UECL"}

_STAGE_TO_AO = {
    "Q1": ("QUALIFYING", "1st Qualifying Round"),
    "Q2": ("QUALIFYING", "2nd Qualifying Round"),
    "Q3": ("QUALIFYING", "3rd Qualifying Round"),
    "LEAGUE_STAGE": ("LEAGUE", "League Stage"),
    "LAST_16": ("ROUND_OF_16", "Round of 16"),
    "QUARTER_FINALS": ("QUARTERFINAL", "Quarter-final"),
    "SEMI_FINALS": ("SEMIFINAL", "Semi-final"),
    "FINAL": ("FINAL", "Final"),
}


def _map_competition(code: str) -> str:
    try:
        return _COMPETITION_TO_AO[code]
    except KeyError as exc:
        raise HTTPException(422, f"Bilinmeyen competition kodu: {code}") from exc


def _map_stage(stage: str, group_name: Optional[str]) -> tuple[str, str, bool]:
    """`(ao_stage, ao_round, is_knockout)` doner.

    `PLAYOFFS`, akiloyunu-api tarafinda AYNI stage degeriyle iki farkli formati
    (qualifying play-off VE lig-fazi-sonrasi knockout play-off) temsil ediyor
    — bkz. akiloyunu-api proje notu (2026-08-29 ampirik dogrulama: ikisi de
    2 bacakli, ama AO'nun K-carpani icin qualifying/knockout ayrimi onemli).
    `group_name` doluysa qualifying play-off, bossa knockout play-off'tur.
    """
    if stage == "PLAYOFFS":
        if group_name:
            return "QUALIFYING", "Qualifying Play-off Round", True
        return "KNOCKOUT_PLAYOFF", "Knockout Play-off Round", True
    try:
        ao_stage, ao_round = _STAGE_TO_AO[stage]
    except KeyError as exc:
        raise HTTPException(422, f"Bilinmeyen stage: {stage}") from exc
    is_knockout = ao_stage != "LEAGUE"
    return ao_stage, ao_round, is_knockout


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Pydantic semalari
# ---------------------------------------------------------------------------
class TeamStateIn(BaseModel):
    team_id: str
    power_elo: float
    achievement_reserve: float = 0.0
    progression_bonus_ucl: float = 0.0
    progression_bonus_uel: float = 0.0
    progression_bonus_uecl: float = 0.0


class FixtureIn(BaseModel):
    match_id: str
    season: str
    kickoff_utc: datetime
    competition: str = Field(description="CL | UEL | UECL")
    stage: str = Field(
        description="Q1 | Q2 | Q3 | PLAYOFFS | LEAGUE_STAGE | LAST_16 | QUARTER_FINALS | SEMI_FINALS | FINAL"
    )
    group_name: Optional[str] = Field(
        default=None,
        description="PLAYOFFS'un qualifying/knockout ayrimi icin (competition_matches.group_name ile ayni)",
    )
    tie_id: Optional[str] = None
    is_tie_decider: bool = False
    is_single_match_tie: bool = False
    is_neutral: bool = False
    home: TeamStateIn
    away: TeamStateIn

    def to_ao(self) -> tuple[str, str, str, bool]:
        ao_competition = _map_competition(self.competition)
        ao_stage, ao_round, is_knockout = _map_stage(self.stage, self.group_name)
        return ao_competition, ao_stage, ao_round, is_knockout

    def season_state(self) -> SeasonState:
        ratings = {
            team.team_id: TeamRating(
                team_id=team.team_id,
                team_name=team.team_id,
                # ao_first_elo yalniz season-baslangici carry hesabinda (bu
                # istekte KULLANILMAYAN _prepare_qualification_transition
                # dalinda) devreye girer — burada guvenli bir yer tutucu.
                ao_first_elo=team.power_elo,
                power_elo=team.power_elo,
                achievement_reserve=team.achievement_reserve,
                progression_bonus_ucl=team.progression_bonus_ucl,
                progression_bonus_uel=team.progression_bonus_uel,
                progression_bonus_uecl=team.progression_bonus_uecl,
            )
            for team in (self.home, self.away)
        }
        return SeasonState(
            season=self.season,
            ratings=ratings,
            processed_match_ids=frozenset(),
            processed_tie_ids=frozenset(),
            open_ties={},
            qualification_participants=frozenset(),
            qualification_carry_applied=frozenset(),
            last_event_utc=None,
            last_match_id=None,
            model_version=_DYNAMIC_CONFIG.model_version,
            config_id=_DYNAMIC_CONFIG.config_id,
        )


class PredictResponse(BaseModel):
    match_id: str
    home_probability: float
    draw_probability: float
    away_probability: float
    expected_home_score: float
    prediction_status: str
    fallback_reason: Optional[str] = None
    model_version: str
    config_id: str
    contract_sha256: str
    # AkilOyunuElo/data/prediction_ledger/*.jsonl'in
    # REQUIRED_PREDICTION_FIELDS'inin (prediction_ledger.py) ihtiyac duydugu,
    # ama yukaridaki dar alanlarda YER ALMAYAN kalan hepsi: ao_*/current_ml_*/
    # ao_poisson_* olasilik kirilimi, artifact_manifest_sha256, ml_artifact_sha256,
    # domestic_state_sha256, competition/round/stage/format_type, home_club_id/
    # away_club_id, kickoff_utc, generated_at_utc, season vb. Model
    # hesaplamasini DEGISTIRMEZ — production_prediction.py'nin ZATEN urettigi
    # PRODUCTION_PREDICTION_LOG_COLUMNS satirini oldugu gibi disari acar.
    # Production servisi calismadiysa (FALLBACK_CURRENT_AO oncesi hicbir
    # ensemble satiri uretilmediyse) None kalir — caller boyle bir satiri
    # append-only ledger'a EKLEMEMELI (eksik kanit).
    ledger_fields: Optional[dict] = None


class SettleRequest(FixtureIn):
    home_goals: int
    away_goals: int
    decided_on_penalties: bool = False
    advanced_team_id: Optional[str] = None
    xg_home: Optional[float] = None
    xg_away: Optional[float] = None
    xg_analysis_eligible: bool = False


class SettleResponse(BaseModel):
    match_id: str
    home_power_elo: float
    away_power_elo: float
    home_live_elo: float
    away_live_elo: float
    power_delta: float
    expected_home_score: float
    goal_difference_multiplier: Optional[float] = None
    xg_applied: bool = False
    progression_bonus_recipient_id: Optional[str] = None
    progression_bonus_added: float = 0.0
    progression_bonus_competition_pre: Optional[float] = None
    progression_bonus_competition_post: Optional[float] = None
    progression_bonus_competition_cap: Optional[float] = None
    model_version: str
    config_id: str


class BootstrapSeasonRequest(BaseModel):
    teams: list[dict]
    country_coefficients: list[dict]
    domestic_context: list[dict]
    club_european_points: list[dict]


class BootstrapSeasonResponse(BaseModel):
    model_version: str
    ratings: list[dict]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AkilOyunuElo Service", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "dynamic_model_version": _DYNAMIC_CONFIG.model_version,
        "dynamic_config_id": _DYNAMIC_CONFIG.config_id,
        "production_prediction_available": bool(
            _PRODUCTION_PREDICTION_SERVICE is not None
            and _PRODUCTION_PREDICTION_SERVICE.runtime is not None
        ),
        "production_prediction_contract_sha256": (
            _PRODUCTION_PREDICTION_SERVICE.contract_sha256
            if _PRODUCTION_PREDICTION_SERVICE is not None
            else None
        ),
        "production_prediction_load_error": (
            _PRODUCTION_PREDICTION_IMPORT_ERROR
            if _PRODUCTION_PREDICTION_SERVICE is None
            else _PRODUCTION_PREDICTION_SERVICE.load_error
        ),
    }


@app.post("/bootstrap-season", response_model=BootstrapSeasonResponse)
def bootstrap_season(payload: BootstrapSeasonRequest) -> BootstrapSeasonResponse:
    if not payload.teams:
        raise HTTPException(422, "teams bos olamaz")
    try:
        output = compute_ao_first_elo(
            pd.DataFrame(payload.teams),
            pd.DataFrame(payload.country_coefficients),
            pd.DataFrame(payload.domestic_context),
            pd.DataFrame(payload.club_european_points),
            _AO_FIRST_CONFIG,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc

    model_version = str(output["model_version"].iloc[0])
    return BootstrapSeasonResponse(
        model_version=model_version,
        ratings=output.to_dict(orient="records"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: FixtureIn) -> PredictResponse:
    ao_competition, ao_stage, ao_round, is_knockout = payload.to_ao()
    state = payload.season_state()
    kickoff = _ensure_utc(payload.kickoff_utc)

    fixture = MatchFixture(
        match_id=payload.match_id,
        season=payload.season,
        kickoff_utc=kickoff,
        competition=ao_competition,
        round=ao_round,
        home_team_id=payload.home.team_id,
        away_team_id=payload.away.team_id,
        is_neutral=payload.is_neutral,
        tie_id=payload.tie_id if is_knockout else None,
        is_knockout=is_knockout,
        is_tie_decider=payload.is_tie_decider if is_knockout else False,
        stage=ao_stage if is_knockout else None,
        is_single_match_tie=payload.is_single_match_tie if is_knockout else False,
    )

    try:
        locked = lock_prediction(
            state,
            fixture,
            _DYNAMIC_CONFIG,
            generated_at_utc=datetime.now(timezone.utc),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    ao_probs = (
        locked.home_win_probability,
        locked.draw_probability,
        locked.away_win_probability,
    )

    final_probs = ao_probs
    status = "FALLBACK_CURRENT_AO"
    fallback_reason: Optional[str] = None
    model_version = _DYNAMIC_CONFIG.model_version
    contract_sha256 = "UNAVAILABLE"
    ledger_fields: Optional[dict] = None

    if _PRODUCTION_PREDICTION_SERVICE is not None:
        contract_sha256 = _PRODUCTION_PREDICTION_SERVICE.contract_sha256
        if _PRODUCTION_PREDICTION_SERVICE.runtime is not None:
            frame = pd.DataFrame(
                [
                    {
                        "match_id": payload.match_id,
                        "season": payload.season,
                        "kickoff_utc": kickoff,
                        "competition": ao_competition,
                        "round": ao_round,
                        "stage": ao_stage,
                        "format_type": "KNOCKOUT" if is_knockout else "LEAGUE",
                        "is_neutral": payload.is_neutral,
                        "home_club_id": payload.home.team_id,
                        "away_club_id": payload.away.team_id,
                        "ao_home_probability": ao_probs[0],
                        "ao_draw_probability": ao_probs[1],
                        "ao_away_probability": ao_probs[2],
                    }
                ]
            )
            try:
                result_frame = _PRODUCTION_PREDICTION_SERVICE.predict(
                    frame, generated_at_utc=datetime.now(timezone.utc)
                )
                row = result_frame.iloc[0]
                final_probs = (
                    float(row["home_probability"]),
                    float(row["draw_probability"]),
                    float(row["away_probability"]),
                )
                status = str(row["prediction_status"])
                reason = str(row["fallback_reason"])
                fallback_reason = reason or None
                model_version = str(row["prediction_model_version"])
                if _ledger_normalize is not None:
                    ledger_fields = {
                        key: _ledger_normalize(value)
                        for key, value in row.to_dict().items()
                    }
            except Exception as exc:  # noqa: BLE001 - "the contract is total": hicbir feature/state sorunu servisi dusurmez
                fallback_reason = f"WRAPPER_ERROR:{type(exc).__name__}"
        else:
            fallback_reason = _PRODUCTION_PREDICTION_SERVICE.load_error
    else:
        fallback_reason = _PRODUCTION_PREDICTION_IMPORT_ERROR

    return PredictResponse(
        match_id=payload.match_id,
        home_probability=final_probs[0],
        draw_probability=final_probs[1],
        away_probability=final_probs[2],
        expected_home_score=locked.expected_home_score,
        prediction_status=status,
        fallback_reason=fallback_reason,
        model_version=model_version,
        config_id=_DYNAMIC_CONFIG.config_id,
        contract_sha256=contract_sha256,
        ledger_fields=ledger_fields,
    )


@app.post("/settle", response_model=SettleResponse)
def settle(payload: SettleRequest) -> SettleResponse:
    ao_competition, ao_stage, ao_round, is_knockout = payload.to_ao()
    state = payload.season_state()
    kickoff = _ensure_utc(payload.kickoff_utc)

    match = MatchInput(
        match_id=payload.match_id,
        season=payload.season,
        kickoff_utc=kickoff,
        competition=ao_competition,
        round=ao_round,
        home_team_id=payload.home.team_id,
        away_team_id=payload.away.team_id,
        home_goals=payload.home_goals,
        away_goals=payload.away_goals,
        is_neutral=payload.is_neutral,
        decided_on_penalties=payload.decided_on_penalties,
        tie_id=payload.tie_id if is_knockout else None,
        is_knockout=is_knockout,
        is_tie_decider=payload.is_tie_decider if is_knockout else False,
        advanced_team_id=payload.advanced_team_id,
        stage=ao_stage if is_knockout else None,
        xg_home=payload.xg_home,
        xg_away=payload.xg_away,
        xg_analysis_eligible=payload.xg_analysis_eligible,
        is_single_match_tie=payload.is_single_match_tie if is_knockout else False,
    )

    try:
        _, update = update_match(state, match, _DYNAMIC_CONFIG)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    return SettleResponse(
        match_id=payload.match_id,
        home_power_elo=update.home_power_post,
        away_power_elo=update.away_power_post,
        home_live_elo=update.home_live_post,
        away_live_elo=update.away_live_post,
        power_delta=update.power_delta,
        expected_home_score=update.expected_home_score,
        goal_difference_multiplier=update.goal_multiplier,
        xg_applied=update.xg_applied,
        progression_bonus_recipient_id=update.progression_bonus_recipient_id,
        progression_bonus_added=update.progression_bonus_added,
        progression_bonus_competition_pre=update.progression_bonus_competition_pre,
        progression_bonus_competition_post=update.progression_bonus_competition_post,
        progression_bonus_competition_cap=update.progression_bonus_competition_cap,
        model_version=update.model_version,
        config_id=update.config_id,
    )
