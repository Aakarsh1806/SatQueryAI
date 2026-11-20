"""Rules-first routing. Scores describe heuristic routing, not model certainty."""
import re
from typing import Literal

from pydantic import Field

from ..schemas.contracts import Contract


class AgentInputError(ValueError):
    """A validation message suitable for display in the UI."""


class RoutingDecision(Contract):
    task: Literal["qa", "caption", "grounding", "change_caption", "classification"]
    model_id: Literal["rs_vlm", "grounding", "change", "fusion"]
    confidence_of_routing: float = Field(ge=0, le=1)
    reasons: list[str]


CHANGE = re.compile(r"\b(changes?|changed|changing|before\s*(?:and|/|vs\.?|to)?\s*after|compare|comparison|difference|differences)\b", re.I)
CAPTION = re.compile(r"\b(describe|caption|summary|summari[sz]e)\b", re.I)
GROUND = re.compile(r"\b(where|locate|find|highlight|show|detect)\b", re.I)


def detect_task(modalities, query, *, llm_fallback=None, enable_llm_fallback=False):
    """Return a decision or raise AgentInputError; LLM can refine default QA only.

    An enabled hook receives query text and returns qa/caption/grounding (or
    None). It cannot override deterministic image-combination or keyword rules.
    """
    if not isinstance(query, str) or not query.strip():
        raise AgentInputError("Enter a question or a caption request.")
    if not 1 <= len(modalities) <= 2:
        raise AgentInputError("Upload one or two images; this request supports at most two images.")
    if any(modality not in {"optical", "sar"} for modality in modalities):
        raise AgentInputError("Each image modality must be optical or sar.")
    if all(modality == "sar" for modality in modalities):
        raise AgentInputError("SAR-only input is not supported; provide an optical image and a SAR image for land-cover classification.")
    if len(modalities) == 2:
        if set(modalities) == {"optical", "sar"}:
            return RoutingDecision(task="classification", model_id="fusion", confidence_of_routing=1,
                reasons=["An optical+SAR pair selects the land-cover classifier; it cannot provide free-form answers."])
        return RoutingDecision(task="change_caption", model_id="change", confidence_of_routing=1,
            reasons=["Two optical images select Chg2Cap (A=before, B=after).",
                     "This model produces a description of changes, not a direct answer to change questions or a change map."])
    if CHANGE.search(query):
        raise AgentInputError("This task needs two images: upload before and after optical images for change analysis.")
    if CAPTION.search(query):
        return RoutingDecision(task="caption", model_id="rs_vlm", confidence_of_routing=.98,
                               reasons=["An explicit describe/caption/summarize request selects the fixed caption prompt."])
    if GROUND.search(query):
        return RoutingDecision(task="grounding", model_id="grounding", confidence_of_routing=.95,
                               reasons=["A location/detection keyword selects grounding (boxes only, no masks)."])
    reasons = ["One optical image with no caption/location keyword defaults to short-phrase QA."]
    if enable_llm_fallback and llm_fallback is not None:
        try:
            task = llm_fallback(query)
            if task in {"qa", "caption", "grounding"}:
                return RoutingDecision(task=task, model_id="grounding" if task == "grounding" else "rs_vlm",
                    confidence_of_routing=.6, reasons=["Optional LLM fallback refined the otherwise default QA route."])
            reasons.append("LLM fallback supplied no supported task; retained rules-based QA.")
        except Exception:
            reasons.append("LLM fallback failed; retained rules-based QA.")
    return RoutingDecision(task="qa", model_id="rs_vlm", confidence_of_routing=.8, reasons=reasons)
