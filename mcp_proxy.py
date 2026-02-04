"""
QuickText MCP Proxy Server
Bridges Mirokai robot (HTTP) to QuickText MCP server (SSE persistent connection)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from mcp.client.session import ClientSession
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCPSession:
    """Manages persistent MCP connection using official client"""
    
    def __init__(self):
        self.base_url = "https://mcp-servers.quicktext.im/mcp"
        self.token = "qt_QZuKapb07QWqVrqHQPlio00IcuPpxfETAwllBOYf_S4"
        self.client: Optional[ClientSession] = None
        self.initialized = False
        self._transport_cm = None
        self._read_stream = None
        self._write_stream = None
        self._get_session_id = None
        self._lock = asyncio.Lock()
        
    async def start(self) -> bool:
        """Initialize MCP session using official client"""
        try:
            # Use Streamable HTTP transport (server-side chooses JSON + SSE under the hood).
            from mcp.client.streamable_http import streamable_http_client
            from mcp.client.sse import create_mcp_http_client

            headers = {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json, text/event-stream",
            }
            http_client = create_mcp_http_client(headers=headers)

            self._transport_cm = streamable_http_client(
                self.base_url,
                http_client=http_client,
                terminate_on_close=False,
            )

            # IMPORTANT: this is a long-running server, so we must keep the transport/session open.
            self._read_stream, self._write_stream, self._get_session_id = await self._transport_cm.__aenter__()
            self.client = ClientSession(self._read_stream, self._write_stream)
            await self.client.__aenter__()
            await self.client.initialize()

            logger.info("✅ MCP initialized with persistent streams (streamable_http)")
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"❌ MCP init error: {e}")
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool using official client"""
        if not self.client or not self.initialized:
            raise HTTPException(500, "MCP not initialized")

        async with self._lock:
            try:
                res = await self.client.call_tool(tool_name, arguments)
                # mcp returns pydantic models (e.g. CallToolResult). Convert to plain JSON-serializable dict.
                if hasattr(res, "model_dump"):
                    return res.model_dump(by_alias=True)
                return res
            except Exception:
                # If the SSE stream dropped, reconnect once and retry.
                self.initialized = False
                await self.stop()
                started = await self.start()
                if not started or not self.client:
                    raise HTTPException(500, "MCP reconnect failed")
                res = await self.client.call_tool(tool_name, arguments)
                if hasattr(res, "model_dump"):
                    return res.model_dump(by_alias=True)
                return res
    
    async def stop(self):
        """Close session gracefully"""
        self.initialized = False
        if self.client is not None:
            try:
                await self.client.__aexit__(None, None, None)
            finally:
                self.client = None

        if self._transport_cm is not None:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            finally:
                self._transport_cm = None
                self._read_stream = None
                self._write_stream = None
                self._get_session_id = None

        logger.info("🔌 MCP session closed")


# Global MCP session instance
mcp_session = MCPSession()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    # Startup
    logger.info("🚀 Starting QuickText MCP Proxy...")
    success = await mcp_session.start()
    if not success:
        logger.error("⚠️  Failed to initialize MCP session")
    yield
    # Shutdown
    await mcp_session.stop()


# FastAPI app
app = FastAPI(
    title="QuickText MCP Proxy",
    version="1.0.0",
    description="REST API proxy for QuickText MCP server",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# Pydantic models
class QueryRequest(BaseModel):
    team_id: str
    query: str
    locale: str = "fr_FR"


class QueryResponse(BaseModel):
    success: bool
    response: str
    dialog_id: Optional[str] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    fallback_to_chatgpt: bool = False


# ============ ENDPOINTS ============

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok" if mcp_session.initialized else "degraded",
        "mcp_initialized": mcp_session.initialized,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Main endpoint for Mirokai robot queries
    
    Example:
    POST /query
    {
        "team_id": "4577",
        "query": "quels sont les horaires du petit déjeuner?",
        "locale": "fr_FR"
    }
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"📝 Query received: '{request.query}' for team {request.team_id}")
        
        # 1. Detect intent
        dialog_id = detect_intent(request.query)
        logger.info(f"🎯 Detected dialog: {dialog_id}")
        
        # 2. Call MCP
        dialog_request = {
            "team_id": request.team_id,
            "dialogs": dialog_id
        }
        
        result = await mcp_session.call_tool("get-dialog-configuration", dialog_request)
        
        # 3. Parse MCP response
        config_data = parse_mcp_response(result)
        
        # 4. Format for voice
        logger.info(f"📦 FULL MCP Response: {json.dumps(result, indent=2)}")
        logger.info(f"📦 Parsed Config Data: {json.dumps(config_data, indent=2)}")
        vocal_response = format_for_voice(config_data, request.locale, dialog_id)
        
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Query completed in {elapsed:.0f}ms")
        
        return QueryResponse(
            success=True,
            response=vocal_response,
            dialog_id=dialog_id,
            latency_ms=int(elapsed)
        )
        
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        
        return QueryResponse(
            success=False,
            response="Désolé, je n'ai pas trouvé cette information dans notre base de données.",
            error=str(e),
            fallback_to_chatgpt=True
        )


@app.get("/test-mcp")
async def test_mcp():
    """Test MCP connection with simple call"""
    try:
        result = await mcp_session.call_tool(
            "get-hotel-settings",
            {"teams": "4577"}
        )
        return {
            "success": True,
            "mcp_response": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/dialogs/search")
async def search_dialogs(keywords: list[str], property_kind: int = 1):
    """Search dialogs by keywords"""
    try:
        result = await mcp_session.call_tool(
            "get-dialogs-list",
            {
                "property_kind": property_kind,
                "description": True,
                "keywords": True
            }
        )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/test-dialog/{dialog_id}")
async def test_dialog(dialog_id: str, team_id: str = "4577"):
    """Test a specific dialog ID directly"""
    try:
        result = await mcp_session.call_tool(
            "get-dialog-configuration",
            {"team_id": team_id, "dialogs": dialog_id}
        )
        return {
            "success": True,
            "dialog_id": dialog_id,
            "raw_response": result
        }
    except Exception as e:
        return {
            "success": False,
            "dialog_id": dialog_id,
            "error": str(e)
        }


# ============ HELPER FUNCTIONS ============

def detect_intent(query: str) -> str:
    """
    Map user query to dialog ID via keyword matching
    Based on real MCP dialogs discovered through testing
    """
    query_lower = query.lower()

    # ===== BREAKFAST (12-xx) - PRIORITY #1 =====
    # Real MCP dialogs: 12-01=time, 12-02=rates, 12-05=menu
    breakfast_keywords = [
        "petit déjeuner", "petit-déjeuner", "petit dejeuner", "petit dej", "petit déj", "breakfast",
        "desayuno", "colazione", "frühstück", "fruhstuck"
    ]
    if any(k in query_lower for k in breakfast_keywords):
        # Menu - what food is served
        if any(k in query_lower for k in [
            "menu", "carte", "quoi", "propose", "options", "contenu", "buffet", "plat", "disponible", "servi", "servis",
            "menú", "menu del", "menù", "menü"
        ]):
            return "12-05"
        # Rates - price information  
        if any(k in query_lower for k in [
            "prix", "tarif", "coût", "combien", "rate", "price", "cost", "cher", "payant", "gratuit",
            "precio", "prezzo", "preis", "costo"
        ]):
            return "12-02"
        # Time - opening hours
        if any(k in query_lower for k in [
            "horaire", "horaires", "heure", "heures", "quand", "time", "opening", "hours", "ouvre", "ferme", "matin",
            "horario", "horarios", "orari", "öffnungszeiten", "uhrzeit"
        ]):
            return "12-01"
        # Default breakfast
        return "12-01"

    # ===== PARKING (17-02) - PRIORITY #2 =====
    parking_keywords = ["parking", "voiture", "garer", "stationnement", "place", "aparcamiento", "parcheggio", "parkplatz"]
    if any(k in query_lower for k in parking_keywords):
        return "17-02"

    # ===== ROOMS (14-01 / 10-01) - PRIORITY #3 =====
    # Use 14-01 for room assignment/number/location; otherwise booking/general.
    room_keywords = ["chambre", "room", "suite", "lit", "beds", "bed"]
    if any(k in query_lower for k in room_keywords):
        if any(k in query_lower for k in ["numéro", "numero", "number", "assign", "attribuée", "ou est", "où est", "where is", "ma chambre", "my room"]):
            return "14-01"
        return "10-01"

    # ===== WIFI/INTERNET (12-04) - PRIORITY #4 =====
    wifi_keywords = ["wifi", "wi-fi", "internet", "connexion", "réseau", "reseau", "web", "code wifi", "contraseña", "password", "passwort", "conexión", "connessione"]
    if any(k in query_lower for k in wifi_keywords):
        return "12-04"

    # ===== RESTAURANT (18-xx) - PRIORITY #5 =====
    restaurant_keywords = ["restaurant", "dîner", "diner", "dinner", "manger", "repas", "déjeuner", "dejeuner", "restaurante", "ristorante"]
    if any(k in query_lower for k in restaurant_keywords):
        return "18-01"  # Restaurant general

    # ===== SWIMMING POOL (17-61/17-62) - PRIORITY #6 =====
    pool_keywords = ["piscine", "pool", "nager", "baignade", "nage", "piscina", "schwimmbad"]
    if any(k in query_lower for k in pool_keywords):
        if any(k in query_lower for k in [
            "horaire", "horaires", "heure", "heures", "quand", "time", "opening", "hours",
            "horario", "horarios", "orari", "öffnungszeiten", "uhrzeit"
        ]):
            return "17-62"  # Pool hours
        return "17-61"  # Pool general

    # ===== SPA (17-06) - PRIORITY #7 =====
    spa_keywords = ["spa", "massage", "bien-être", "bien etre", "relaxation", "soins", "wellness", "benessere"]
    if any(k in query_lower for k in spa_keywords):
        return "17-06"

    # ===== GYM/FITNESS (17-07) - PRIORITY #8 =====
    gym_keywords = ["gym", "fitness", "sport", "muscu", "entraînement", "entrainement", "salle de sport", "gimnasio", "palestra", "fitnessraum"]
    if any(k in query_lower for k in gym_keywords):
        return "17-07"

    # ===== BOOKING (10-xx) - PRIORITY #9 =====
    booking_keywords = ["réservation", "réserver", "booking", "reserve", "résa", "resa", "reserva", "prenotazione", "reservierung"]
    if any(k in query_lower for k in booking_keywords):
        return "10-01"  # Booking general

    # ===== CANCELLATION (10-08) - PRIORITY #10 =====
    cancel_keywords = ["annulation", "annuler", "cancel", "cancellation", "cancelación", "annullamento", "stornierung"]
    if any(k in query_lower for k in cancel_keywords):
        return "10-08"

    # ===== PRICING (10-03) - PRIORITY #11 =====
    price_keywords = ["prix", "tarif", "coût", "combien", "rate", "price", "cost", "cher", "gratuit"]
    if any(k in query_lower for k in price_keywords):
        return "10-03"

    # ===== NEARBY (19-xx) - PRIORITY #12 =====
    nearby_keywords = ["proximité", "près", "autour", "nearby", "close", "aux alentours"]
    if any(k in query_lower for k in nearby_keywords):
        return "19-01"  # Location general

    # Default fallback
    return "10-01"


def parse_mcp_response(mcp_result: Dict[str, Any]) -> Dict[str, Any]:
    """Parse MCP JSON-RPC response"""
    if not isinstance(mcp_result, dict):
        return {}

    if "error" in mcp_result:
        error_msg = mcp_result["error"].get("message", "Unknown error")
        raise Exception(f"MCP error: {error_msg}")

    if mcp_result.get("isError") is True:
        raise Exception("MCP tool returned an error")

    structured = mcp_result.get("structuredContent")
    if isinstance(structured, dict) and structured:
        return structured

    content = mcp_result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    entry = parsed[0]
                    configs = entry.get("configs")
                    if isinstance(configs, list) and configs:
                        fields = configs[0].get("fields")
                        if isinstance(fields, dict):
                            return {"raw": fields, "info": entry.get("info"), "configs": configs}
                return {"raw": parsed}
            except Exception:
                return {"raw_text": text}

    if "result" in mcp_result and isinstance(mcp_result["result"], dict):
        return mcp_result["result"]

    return {}


def format_for_voice(config_data: Dict[str, Any], locale: str, dialog_id: str) -> str:
    """
    Convert MCP configuration to natural French sentence for TTS
    
    Note: This is simplified - actual implementation depends on 
    real MCP response structure which we'll discover through logging
    """
    
    if not config_data:
        return "Je n'ai pas trouvé cette information dans notre système."
    
    # Log full response for debugging
    logger.debug(f"Formatting config for dialog {dialog_id}: {json.dumps(config_data, indent=2)}")
    
    # Extract common structured data if present
    def _pick_localized(items: Any) -> Optional[str]:
        if not isinstance(items, list):
            return None
        lang = "fr"
        if locale:
            lang = locale.split("_")[0].lower()
            if lang not in {"fr", "en", "es", "it", "de"}:
                lang = "en"
        for item in items:
            if isinstance(item, dict) and item.get(lang):
                return str(item.get(lang)).strip()
        for item in items:
            if isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return None

    def _humanize_token(token: str) -> str:
        mapping = {
            "coffee": "café",
            "fruit_juice": "jus de fruits",
            "tea": "thé",
            "pastries": "viennoiseries",
            "baguette": "baguette",
            "toasts": "tartines",
            "boiled_eggs": "oeufs durs",
            "cereals_lower_case": "céréales",
            "fresh_fruit": "fruits frais",
            "salad": "salade",
            "cheese": "fromages",
            "whole_milk": "lait",
            "butter": "beurre",
            "yogurts": "yaourts",
            "sausage": "saucisses",
            "bacon": "bacon",
            "cold_meat": "charcuterie",
            "jams": "confitures",
            "honey": "miel",
            "oat_milk": "lait d'avoine",
        }
        return mapping.get(token, token.replace("_", " "))

    lang = locale.split("_")[0].lower() if locale else "en"
    if lang not in {"fr", "en", "es", "it", "de"}:
        lang = "en"

    phrases = {
        "breakfast_time": {
            "fr": "Le petit-déjeuner est servi de {from_time} à {to_time}{place}.",
            "en": "Breakfast is served from {from_time} to {to_time}{place}.",
            "es": "El desayuno se sirve de {from_time} a {to_time}{place}.",
            "it": "La colazione è servita dalle {from_time} alle {to_time}{place}.",
            "de": "Frühstück wird von {from_time} bis {to_time} serviert{place}.",
        },
        "parking": {
            "fr": "Le parking est disponible. Pour plus d'infos: {contact}.",
            "en": "Parking is available. For more info: {contact}.",
            "es": "Hay aparcamiento disponible. Para más info: {contact}.",
            "it": "Il parcheggio è disponibile. Per informazioni: {contact}.",
            "de": "Parkplätze sind verfügbar. Für weitere Infos: {contact}.",
        },
        "pool_hours": {
            "fr": "{name} est ouverte de {from_time} à {to_time}.",
            "en": "{name} is open from {from_time} to {to_time}.",
            "es": "{name} abre de {from_time} a {to_time}.",
            "it": "{name} è aperta dalle {from_time} alle {to_time}.",
            "de": "{name} ist von {from_time} bis {to_time} geöffnet.",
        },
    }

    raw = config_data.get("raw") if isinstance(config_data, dict) else None
    if isinstance(raw, dict):
        # Breakfast time (12-01): keys like from/to/serveIn
        if dialog_id == "12-01":
            time_from = raw.get("from") or raw.get("start") or raw.get("startTime")
            time_to = raw.get("to") or raw.get("end") or raw.get("endTime")
            serve_in = _pick_localized(raw.get("serveIn")) or raw.get("location")
            if time_from and time_to:
                place = f" au {serve_in}" if (serve_in and lang == "fr") else (f" at {serve_in}" if serve_in else "")
                return phrases["breakfast_time"][lang].format(from_time=time_from, to_time=time_to, place=place)

        # Breakfast rates (12-02): prefer localized text if available
        if dialog_id == "12-02":
            message = raw.get("message")
            if isinstance(message, dict):
                lang = locale.split("_")[0].lower() if locale else "en"
                localized = message.get(lang) or message.get("en")
                if isinstance(localized, str) and localized.strip():
                    return localized.strip()
            lang = locale.split("_")[0].lower() if locale else "en"
            localized = raw.get(lang) or raw.get("en")
            if isinstance(localized, str) and localized.strip():
                return localized.strip()
            if isinstance(raw.get("text"), str):
                return raw["text"].strip()

        # Breakfast menu (12-05): use types/drinks if available
        if dialog_id == "12-05":
            types = raw.get("breakfastTypes")
            drinks = raw.get("drinks")
            breads = raw.get("breadAndCakes")
            parts = []
            if isinstance(types, list) and types:
                types_text = ", ".join(_humanize_token(t) if lang == "fr" else t.replace("_", " ") for t in types)
                label = "Types proposés" if lang == "fr" else "Types"
                parts.append(f"{label}: {types_text}")
            if isinstance(drinks, list) and drinks:
                drinks_text = ", ".join(_humanize_token(d) if lang == "fr" else d.replace("_", " ") for d in drinks)
                label = "Boissons" if lang == "fr" else "Drinks"
                parts.append(f"{label}: {drinks_text}")
            if isinstance(breads, list) and breads:
                breads_text = ", ".join(_humanize_token(b) if lang == "fr" else b.replace("_", " ") for b in breads)
                label = "Pains/viennoiseries" if lang == "fr" else "Bread & pastries"
                parts.append(f"{label}: {breads_text}")
            if parts:
                return " / ".join(parts) + "."

        # Parking (17-02): contact info if available
        if dialog_id == "17-02":
            settings = raw.get("notificationSettings")
            if isinstance(settings, list) and settings:
                item = settings[0]
                phone = item.get("phone")
                email = item.get("email")
                if phone and email:
                    joiner = "ou" if lang == "fr" else "or"
                    return phrases["parking"][lang].format(contact=f"{phone} {joiner} {email}")
                if phone:
                    return phrases["parking"][lang].format(contact=phone)
                if email:
                    return phrases["parking"][lang].format(contact=email)
            return "Un parking est disponible. Demandez les détails à l'accueil."

        # Pool general (17-61): extract pool details
        if dialog_id == "17-61":
            tabs = raw.get("poolsTabs")
            if isinstance(tabs, list) and tabs:
                tab = tabs[0]
                name = _pick_localized(tab.get("swimmingPoolName")) or ("la piscine" if lang == "fr" else "the pool")
                pool_type = tab.get("poolType")
                heated = tab.get("isHeated")
                temp = tab.get("heatedPoolTemperature", {}).get("value") if isinstance(tab.get("heatedPoolTemperature"), dict) else None
                open_public = tab.get("hasOpenToPublic")
                price = tab.get("openToPublicPriceValue", {}).get("value") if isinstance(tab.get("openToPublicPriceValue"), dict) else None
                unit = tab.get("openToPublicPriceValue", {}).get("unit") if isinstance(tab.get("openToPublicPriceValue"), dict) else None
                parts = []
                if pool_type:
                    if lang == "fr" and pool_type == "indoor" and "piscine intérieure" in str(name).lower():
                        parts.append(str(name))
                    else:
                        pt = "intérieure" if (lang == "fr" and pool_type == "indoor") else pool_type
                        parts.append(f"{name} ({pt})")
                else:
                    parts.append(str(name))
                if heated == "yes" and temp:
                    parts.append(f"chauffée à {temp}°" if lang == "fr" else f"heated to {temp}°")
                if open_public == "yes" and price:
                    label = "ouverte au public" if lang == "fr" else "open to public"
                    if unit:
                        parts.append(f"{label}: {price} {unit}")
                    else:
                        parts.append(f"{label}: {price}")
                return " / ".join(parts) + "."

        # Pool hours (17-62): extract opening times
        if dialog_id == "17-62":
            pool = raw.get("pool")
            if isinstance(pool, dict) and pool:
                first_key = next(iter(pool))
                p = pool.get(first_key, {})
                name = p.get("tab_name") or ("la piscine" if lang == "fr" else "the pool")
                times = p.get("openEveryDayTime")
                if isinstance(times, list) and times:
                    t0 = times[0]
                    time_from = t0.get("openEveryDayTimeFrom")
                    time_to = t0.get("openEveryDayTimeTo")
                    if time_from and time_to:
                        return phrases["pool_hours"][lang].format(name=name, from_time=time_from, to_time=time_to)

        # Generic localized text
        lang = locale.split("_")[0].lower() if locale else "en"
        localized = raw.get(lang) or raw.get("en")
        if isinstance(localized, str) and localized.strip():
            return localized.strip()

    # Generic fallback based on dialog type
    dialog_responses = {
        "12-01": "Le petit-déjeuner est servi le matin. La réception peut vous confirmer les horaires exacts.",
        "12-02": "Le petit-déjeuner est disponible. La réception peut vous confirmer les tarifs.",
        "12-05": "Le petit-déjeuner propose plusieurs options. La réception peut détailler le menu.",
        "18-01": "Notre restaurant est ouvert pour le service. Contactez la réception pour plus d'informations.",
        "17-05": "La piscine est disponible pour nos clients. Demandez les horaires à la réception.",
        "17-06": "Le spa propose différents services. La réception peut vous renseigner.",
        "17-02": "Un parking est disponible. Demandez les détails à l'accueil.",
        "10-08": "Pour les conditions d'annulation, veuillez contacter notre service de réservation.",
        "10-01": "Pour toute information sur les réservations, contactez notre équipe.",
        "17-07": "La salle de sport est accessible. Renseignez-vous à la réception pour les horaires.",
        "12-04": "Le wifi est disponible dans l'hôtel. Les codes d'accès sont fournis à la réception.",
    }
    
    # Try to extract meaningful info from config_data
    # This will be refined once we see actual MCP response structure
    
    return dialog_responses.get(dialog_id, "Information disponible à la réception de l'hôtel.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
