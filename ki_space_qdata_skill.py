from __future__ import annotations

import os
from typing import Any, Dict

import requests

try:
    from pymirokai.decorators.skill import ParameterDescription, skill
    from pymirokai.enums import AccessLevel
    from pymirokai.robot import Robot
except Exception:  # pragma: no cover
    ParameterDescription = None  # type: ignore[assignment]
    AccessLevel = None  # type: ignore[assignment]

    def skill(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def _wrap(fn: Any):
            return fn

        return _wrap

    Robot = Any  # type: ignore[misc,assignment]


DEFAULT_PROXY_URL = os.environ.get("KI_SPACE_PROXY_URL", "https://mcp-proxy-vg3s.onrender.com/query")
DEFAULT_TEAM_ID = os.environ.get("KI_SPACE_TEAM_ID", "4577")
DEFAULT_LOCALE = os.environ.get("KI_SPACE_LOCALE", "fr_FR")


@skill(
    access_level=AccessLevel.USER if AccessLevel else None,
    can_be_triggered_by_llm=True,
    verbal_descriptions={
        "fr": [
            "petit déjeuner",
            "petit dejeuner",
            "petit déj",
            "petit dej",
            "restaurant",
            "dîner",
            "piscine",
            "spa",
            "parking",
            "wifi",
            "code wifi",
            "salle de sport",
            "annulation",
            "réservation",
            "tarif",
            "prix",
            "chambre",
        ],
        "en": [
            "breakfast",
            "breakfast time",
            "breakfast price",
            "breakfast menu",
            "restaurant",
            "dinner",
            "pool",
            "spa",
            "parking",
            "wifi",
            "gym",
            "cancellation",
            "booking",
            "price",
            "room",
        ],
    },
    parameters=[
        ParameterDescription(name="query", description="User question about the hotel.") if ParameterDescription else None,
        ParameterDescription(name="timeout", description="Timeout in seconds for the external API call.") if ParameterDescription else None,
    ]
    if ParameterDescription
    else [],
)
async def ki_space_qdata(
    robot: Robot,
    query: str,
    timeout: int = 10,
    team_id: str = DEFAULT_TEAM_ID,
    locale: str = DEFAULT_LOCALE,
    proxy_url: str = DEFAULT_PROXY_URL,
) -> Dict[str, Any]:
    if not query:
        return {"llm_output": "Je n'ai pas compris votre question."}

    try:
        resp = requests.post(
            proxy_url,
            json={
                "team_id": team_id,
                "query": query,
                "locale": locale,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        return {"llm_output": "Le service met trop de temps à répondre. Pouvez-vous répéter votre question ?"}
    except Exception:
        return {"llm_output": "Je rencontre un problème technique de connexion au service d'information."}

    if data.get("success") is True and isinstance(data.get("response"), str):
        return {"llm_output": data["response"]}

    if data.get("fallback_to_chatgpt") is True:
        return {
            "llm_output": "Je n'ai pas trouvé l'information dans la base. Utilise tes connaissances générales pour répondre à l'utilisateur.",
            "fallback_to_chatgpt": True,
        }

    return {"llm_output": "Je n'ai pas trouvé cette information."}


if __name__ == "__main__":
    import asyncio

    async def _test():
        out = await ki_space_qdata(
            robot=None,
            query="Quels sont les horaires du petit déjeuner ?",
            timeout=10,
            proxy_url=DEFAULT_PROXY_URL,
        )
        print(out)

    asyncio.run(_test())
