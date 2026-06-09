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
import os
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
        self.token = os.environ.get("MCP_TOKEN", "")
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

        # Special case: Hotel presentation (combines multiple info)
        if dialog_id == "PRESENTATION":
            try:
                settings_result = await mcp_session.call_tool("get-hotel-settings", {"teams": request.team_id})
                hotel_info = parse_hotel_settings(settings_result)
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                return QueryResponse(
                    success=True,
                    response=hotel_info,
                    dialog_id="PRESENTATION",
                    latency_ms=int(elapsed)
                )
            except Exception as e:
                logger.error(f"Presentation error: {e}")
                return QueryResponse(
                    success=True,
                    response="Bienvenue au Ki Space Val d'Europe, un hôtel 4 étoiles situé près de Disneyland Paris. Nous proposons 274 chambres, un restaurant, une piscine intérieure, un spa et une salle de sport.",
                    dialog_id="PRESENTATION",
                    latency_ms=int((datetime.now() - start_time).total_seconds() * 1000)
                )

        # 2. Call MCP
        dialog_request = {
            "teams": request.team_id,
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
            {"teams": team_id, "dialogs": dialog_id}
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

    # ===== CHECK-IN (11-01) - PRIORITY #1 =====
    checkin_keywords = [
        "check-in", "checkin", "check in", "arrivée", "arrivee", "heure d'arrivée",
        "enregistrement", "à quelle heure on arrive", "hora de llegada", "einchecken", "arrivo"
    ]
    if any(k in query_lower for k in checkin_keywords):
        return "11-01"

    # ===== CHECK-OUT (11-11) - PRIORITY #2 =====
    checkout_keywords = [
        "check-out", "checkout", "check out", "départ", "depart", "heure de départ",
        "libérer", "liberer", "quitter", "hora de salida", "auschecken", "partenza"
    ]
    if any(k in query_lower for k in checkout_keywords):
        return "11-11"

    # ===== BREAKFAST (12-xx) - PRIORITY #3 =====
    breakfast_keywords = [
        "petit déjeuner", "petit-déjeuner", "petit dejeuner", "petit dej", "petit déj", "breakfast",
        "desayuno", "colazione", "frühstück", "fruhstuck"
    ]
    if any(k in query_lower for k in breakfast_keywords):
        if any(k in query_lower for k in [
            "menu", "carte", "quoi", "propose", "options", "contenu", "buffet", "plat", "disponible", "servi", "servis",
            "menú", "menu del", "menù", "menü"
        ]):
            return "12-05"
        if any(k in query_lower for k in [
            "prix", "tarif", "coût", "combien", "rate", "price", "cost", "cher", "payant", "gratuit",
            "precio", "prezzo", "preis", "costo"
        ]):
            return "12-02"
        if any(k in query_lower for k in [
            "horaire", "horaires", "heure", "heures", "quand", "time", "opening", "hours", "ouvre", "ferme", "matin",
            "horario", "horarios", "orari", "öffnungszeiten", "uhrzeit"
        ]):
            return "12-01"
        return "12-01"

    # ===== PARKING (13-01) - PRIORITY #4 =====
    parking_keywords = [
        "parking", "voiture", "garer", "stationnement", "place de parking", "garage",
        "véhicule", "vehicule", "car park", "aparcamiento", "parcheggio", "parkplatz",
        "borne électrique", "borne electrique", "electric car", "recharge", "charging"
    ]
    if any(k in query_lower for k in parking_keywords):
        return "13-01"

    # ===== DISNEYLAND (FORCE_FALLBACK) - PRIORITY #16.5 =====
    disney_keywords = [
        "disney", "disneyland", "parc", "attraction", "billet", "ticket",
        "activité", "activités", "y a til", "il y a", "disney paris",
        "walt disney", "mickey", "minnie", "princesse", "château", "fermeture", "ouverture", "horaires"
    ]
    if any(k in query_lower for k in disney_keywords):
        return "FORCE_FALLBACK"  # Force fallback intelligent

    # ===== PETS/ANIMALS (16-10) - PRIORITY #5 =====
    pets_keywords = [
        "animal", "animaux", "chien", "chat", "pet", "pets", "dog", "cat",
        "mascota", "animale", "haustier", "hund", "katze", "cane", "gatto"
    ]
    if any(k in query_lower for k in pets_keywords):
        return "16-10"

    # ===== LUGGAGE (11-21) - PRIORITY #6 =====
    luggage_keywords = [
        "bagage", "bagages", "valise", "valises", "consigne", "luggage", "suitcase",
        "equipaje", "bagaglio", "gepäck", "koffer"
    ]
    if any(k in query_lower for k in luggage_keywords):
        return "11-21"

    # ===== WIFI/INTERNET (13-11/13-12) - PRIORITY #7 =====
    wifi_keywords = [
        "wifi", "wi-fi", "internet", "connexion", "réseau", "reseau", "web",
        "contraseña", "password", "passwort", "conexión", "connessione"
    ]
    if any(k in query_lower for k in wifi_keywords):
        if any(k in query_lower for k in ["code", "mot de passe", "password", "accès", "acces", "connect"]):
            return "13-12"
        return "13-11"

    # ===== RESTAURANT (18-01) - PRIORITY #8 =====
    restaurant_keywords = [
        "restaurant", "dîner", "diner", "dinner", "manger", "repas", "déjeuner", "dejeuner",
        "restaurante", "ristorante", "lunch", "souper"
    ]
    if any(k in query_lower for k in restaurant_keywords):
        return "18-01"

    # ===== SWIMMING POOL (17-61/17-62) - PRIORITY #9 =====
    pool_keywords = ["piscine", "pool", "nager", "baignade", "nage", "piscina", "schwimmbad", "swimming"]
    if any(k in query_lower for k in pool_keywords):
        if any(k in query_lower for k in [
            "horaire", "horaires", "heure", "heures", "quand", "time", "opening", "hours",
            "horario", "horarios", "orari", "öffnungszeiten", "uhrzeit"
        ]):
            return "17-62"
        return "17-61"

    # ===== SPA (17-10) - PRIORITY #10 =====
    spa_keywords = [
        "spa", "massage", "bien-être", "bien etre", "relaxation", "soins", "wellness",
        "benessere", "hammam", "sauna", "jacuzzi"
    ]
    if any(k in query_lower for k in spa_keywords):
        return "17-10"

    # ===== GYM/FITNESS (17-31) - PRIORITY #11 =====
    gym_keywords = [
        "gym", "fitness", "sport", "muscu", "musculation", "entraînement", "entrainement",
        "salle de sport", "gimnasio", "palestra", "fitnessraum", "exercise"
    ]
    if any(k in query_lower for k in gym_keywords):
        return "17-31"

    # ===== CANCELLATION (10-05) - PRIORITY #12 =====
    cancel_keywords = [
        "annulation", "annuler", "cancel", "cancellation", "cancelación", "annullamento",
        "stornierung", "politique d'annulation", "conditions d'annulation", "rembours"
    ]
    if any(k in query_lower for k in cancel_keywords):
        return "10-05"

    # ===== ROOMS (14-01 / 10-01) - PRIORITY #13 =====
    room_keywords = ["chambre", "room", "suite", "lit", "beds", "bed", "hébergement", "hebergement"]
    if any(k in query_lower for k in room_keywords):
        if any(k in query_lower for k in ["numéro", "numero", "number", "assign", "attribuée", "où est", "where is", "ma chambre", "my room"]):
            return "14-01"
        return "10-01"

    # ===== LOCATION/ADDRESS (19-01) - PRIORITY #14 =====
    location_keywords = [
        "adresse", "address", "où", "localisation", "location", "situé", "situe",
        "comment venir", "how to get", "dirección", "indirizzo", "adresse"
    ]
    if any(k in query_lower for k in location_keywords):
        return "19-01"

    # ===== HOTEL PRESENTATION - PRIORITY #15 =====
    presentation_keywords = [
        "présente", "presente", "présentation", "presentation", "parle-moi de", "parle moi de",
        "c'est quoi", "describe", "about the hotel", "tell me about", "qu'est-ce que",
        "hôtel", "hotel", "établissement", "etablissement"
    ]
    if any(k in query_lower for k in presentation_keywords):
        return "PRESENTATION"

    # ===== BOOKING (10-01) - PRIORITY #16 =====
    booking_keywords = [
        "réservation", "réserver", "booking", "reserve", "résa", "resa",
        "reserva", "prenotazione", "reservierung", "book"
    ]
    if any(k in query_lower for k in booking_keywords):
        return "10-01"

    # ===== PRICING (10-03) - PRIORITY #17 =====
    price_keywords = ["prix", "tarif", "coût", "combien", "rate", "price", "cost", "cher", "gratuit"]
    if any(k in query_lower for k in price_keywords):
        return "10-03"

    # ===== NEARBY (19-01) - PRIORITY #18 =====
    nearby_keywords = ["proximité", "près", "autour", "nearby", "close", "aux alentours", "around"]
    if any(k in query_lower for k in nearby_keywords):
        return "19-01"

    # Default fallback
    return "10-01"


def parse_hotel_settings(settings_result: Dict[str, Any]) -> str:
    """Parse hotel settings to create a presentation text"""
    try:
        content = settings_result.get("content", [])
        if isinstance(content, list) and content:
            text = content[0].get("text", "")
            if text:
                data = json.loads(text)
                if isinstance(data, list) and data:
                    hotel = data[0]
                    info = hotel.get("information", {})

                    name = info.get("name", "l'hôtel")
                    stars = info.get("stars", 4)
                    rooms = info.get("rooms", "")
                    address = info.get("address", {}).get("location", "")
                    checkin = info.get("checkin", "15:00")
                    checkout = info.get("checkout", "11:00")

                    parts = [f"Bienvenue au {name}"]
                    if stars:
                        parts.append(f"un hôtel {stars} étoiles")
                    if address:
                        parts.append(f"situé au {address}")
                    if rooms:
                        parts.append(f"Nous disposons de {rooms} chambres")

                    extras = []
                    extras.append("un restaurant")
                    extras.append("une piscine intérieure chauffée")
                    extras.append("un spa")
                    extras.append("une salle de sport")

                    parts.append("avec " + ", ".join(extras))
                    parts.append(f"Check-in à partir de {checkin}, check-out avant {checkout}")

                    return ". ".join(parts) + "."
    except Exception as e:
        logger.error(f"Error parsing hotel settings: {e}")

    return "Bienvenue au Ki Space Val d'Europe, un hôtel 4 étoiles situé près de Disneyland Paris. Nous proposons 274 chambres, un restaurant, une piscine intérieure, un spa et une salle de sport. Check-in à 15h00, check-out à 11h00."


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

        # ===== CHECK-IN (11-01) =====
        if dialog_id == "11-01":
            message = raw.get("message")
            if isinstance(message, dict):
                localized = message.get(lang) or message.get("fr") or message.get("en")
                if isinstance(localized, str) and localized.strip():
                    return localized.strip()

        # ===== CHECK-OUT (11-11) =====
        if dialog_id == "11-11":
            message = raw.get("message")
            if isinstance(message, dict):
                localized = message.get(lang) or message.get("fr") or message.get("en")
                if isinstance(localized, str) and localized.strip():
                    return localized.strip()

        # ===== PARKING (13-01) =====
        if dialog_id == "13-01":
            message_yes = raw.get("messageYes")
            if isinstance(message_yes, dict):
                localized = message_yes.get(lang) or message_yes.get("fr") or message_yes.get("en")
                if isinstance(localized, str) and localized.strip():
                    return localized.strip()

        # ===== CANCELLATION (10-05) =====
        if dialog_id == "10-05":
            message = raw.get("message")
            if isinstance(message, dict):
                localized = message.get(lang) or message.get("fr") or message.get("en")
                if isinstance(localized, str) and localized.strip():
                    return localized.strip()

        # ===== PETS (16-10) =====
        if dialog_id == "16-10":
            has_pets = raw.get("hasPets")
            if has_pets == "yes":
                fee = raw.get("petFeeUnit", {})
                fee_value = fee.get("value") if isinstance(fee, dict) else None
                fee_unit = fee.get("unit", "EUR") if isinstance(fee, dict) else "EUR"
                max_pets = raw.get("maxPetPerRoom", {}).get("value") if isinstance(raw.get("maxPetPerRoom"), dict) else None
                max_weight = raw.get("maxWeightValue", {}).get("value") if isinstance(raw.get("maxWeightValue"), dict) else None

                parts = ["Les animaux sont acceptés dans l'hôtel" if lang == "fr" else "Pets are welcome at the hotel"]
                if fee_value:
                    parts.append(f"supplément de {fee_value} {fee_unit} par animal" if lang == "fr" else f"fee of {fee_value} {fee_unit} per pet")
                if max_pets:
                    parts.append(f"maximum {max_pets} par chambre" if lang == "fr" else f"max {max_pets} per room")
                if max_weight:
                    parts.append(f"poids maximum {max_weight} kg" if lang == "fr" else f"max weight {max_weight} kg")
                return ", ".join(parts) + "."
            else:
                return "Les animaux ne sont pas acceptés dans l'hôtel." if lang == "fr" else "Pets are not allowed at the hotel."

        # ===== LUGGAGE (11-21) =====
        if dialog_id == "11-21":
            has_luggage = raw.get("hasStoreLuggage")
            if has_luggage == "yes":
                return "Nous pouvons garder vos bagages avant le check-in ou après le check-out." if lang == "fr" else "We can store your luggage before check-in or after check-out."
            return "La consigne à bagages n'est pas disponible." if lang == "fr" else "Luggage storage is not available."

        # ===== WIFI (13-11) =====
        if dialog_id == "13-11":
            has_wifi = raw.get("hasWifi")
            if has_wifi == "yesWifi":
                standard = raw.get("standardWifi")
                hi_speed = raw.get("hiSpeedWifi")
                speed = raw.get("hiSpeedValue", {}).get("value") if isinstance(raw.get("hiSpeedValue"), dict) else None

                parts = ["Le WiFi est disponible" if lang == "fr" else "WiFi is available"]
                if standard == "yesFree" or hi_speed == "yesFreeHispeed":
                    parts.append("gratuitement" if lang == "fr" else "for free")
                if speed:
                    parts.append(f"vitesse jusqu'à {speed} Mbit/s" if lang == "fr" else f"speed up to {speed} Mbit/s")
                return ", ".join(parts) + "."
            return "Le WiFi n'est pas disponible." if lang == "fr" else "WiFi is not available."

        # ===== WIFI CODE (13-12) =====
        if dialog_id == "13-12":
            code_type = raw.get("wifiCodeTypes")
            if code_type == "wifiCodeAtFrontDesk":
                return "Le code WiFi est disponible à la réception." if lang == "fr" else "WiFi code is available at the front desk."
            return "Demandez le code WiFi à la réception." if lang == "fr" else "Ask for the WiFi code at the front desk."

        # ===== SPA (17-10) =====
        if dialog_id == "17-10":
            has_spa = raw.get("hasSpa")
            if has_spa == "yes":
                tabs = raw.get("spasTabs")
                if isinstance(tabs, list) and tabs:
                    spa = tabs[0]
                    name = spa.get("spaName", "Le spa")
                    facilities = spa.get("facilities", [])
                    guest_price = spa.get("freeForGuestPriceValue", {})
                    guest_fee = guest_price.get("value") if isinstance(guest_price, dict) else None
                    age_limit = spa.get("ageLimitValue", {}).get("value") if isinstance(spa.get("ageLimitValue"), dict) else None

                    parts = [f"{name.strip()} propose" if lang == "fr" else f"{name.strip()} offers"]
                    if facilities:
                        facilities_fr = {"jacuzzi": "jacuzzi", "massage": "massages", "pool": "piscine", "steamBath": "hammam", "sauna": "sauna", "solarium": "solarium"}
                        fac_text = ", ".join(facilities_fr.get(f, f) for f in facilities[:4])
                        parts.append(fac_text)
                    if guest_fee:
                        parts.append(f"tarif clients: {guest_fee} EUR/heure" if lang == "fr" else f"guest rate: {guest_fee} EUR/hour")
                    if age_limit:
                        parts.append(f"âge minimum {age_limit} ans" if lang == "fr" else f"minimum age {age_limit}")
                    return ", ".join(parts) + "."
            return "Le spa n'est pas disponible." if lang == "fr" else "Spa is not available."

        # ===== GYM (17-31) =====
        if dialog_id == "17-31":
            has_gym = raw.get("hasFitnessGym")
            if has_gym == "yes":
                times = raw.get("openEveryDayTime")
                free_guests = raw.get("hasFreeForGuests")
                location = _pick_localized(raw.get("fitnessGymLocation"))

                parts = ["La salle de sport est ouverte" if lang == "fr" else "The gym is open"]
                if isinstance(times, list) and times:
                    t = times[0]
                    parts.append(f"de {t.get('from')} à {t.get('to')}" if lang == "fr" else f"from {t.get('from')} to {t.get('to')}")
                if free_guests == "yes":
                    parts.append("gratuite pour les clients" if lang == "fr" else "free for guests")
                if location:
                    parts.append(f"située au {location}" if lang == "fr" else f"located at {location}")
                return ", ".join(parts) + "."
            return "La salle de sport n'est pas disponible." if lang == "fr" else "Gym is not available."

        # ===== RESTAURANT (18-01) =====
        if dialog_id == "18-01":
            has_restaurant = raw.get("hasRestaurant")
            if has_restaurant:
                tabs = raw.get("restaurantsTabs")
                if isinstance(tabs, list) and tabs:
                    resto = tabs[0]
                    name = resto.get("restaurantName", "Le restaurant")
                    kitchen = resto.get("restaurantKitchen", "")
                    lunch_from = resto.get("restaurantLunchFromMonday")
                    lunch_to = resto.get("restaurantLunchToMonday")
                    dinner_from = resto.get("restaurantDinnerFromMonday")
                    dinner_to = resto.get("restaurantDinnerToMonday")

                    parts = [f"{name.strip()}" if lang == "fr" else f"{name.strip()}"]
                    if kitchen:
                        kitchen_fr = {"international": "cuisine internationale", "french": "cuisine française"}
                        parts.append(kitchen_fr.get(kitchen, kitchen))
                    if lunch_from and lunch_to:
                        parts.append(f"déjeuner {lunch_from}-{lunch_to}" if lang == "fr" else f"lunch {lunch_from}-{lunch_to}")
                    if dinner_from and dinner_to:
                        parts.append(f"dîner {dinner_from}-{dinner_to}" if lang == "fr" else f"dinner {dinner_from}-{dinner_to}")
                    return ", ".join(parts) + "."
            return "Information restaurant disponible à la réception." if lang == "fr" else "Restaurant info available at the front desk."

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
                localized = message.get(lang) or message.get("fr") or message.get("en")
                if isinstance(localized, str) and localized.strip():
                    return localized.strip()
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

        # Generic localized message (works for many dialogs)
        message = raw.get("message")
        if isinstance(message, dict):
            localized = message.get(lang) or message.get("fr") or message.get("en")
            if isinstance(localized, str) and localized.strip():
                return localized.strip()

        # Generic localized text
        localized = raw.get(lang) or raw.get("fr") or raw.get("en")
        if isinstance(localized, str) and localized.strip():
            return localized.strip()

    # Generic fallback based on dialog type
    dialog_responses = {
        "11-01": "Le check-in est à 15h00. La réception peut vous renseigner sur les arrivées anticipées.",
        "11-11": "Le check-out est à 11h00. Demandez à la réception pour un départ tardif.",
        "11-21": "Nous pouvons garder vos bagages. Renseignez-vous à la réception.",
        "12-01": "Le petit-déjeuner est servi le matin. La réception peut vous confirmer les horaires exacts.",
        "12-02": "Le petit-déjeuner est disponible. La réception peut vous confirmer les tarifs.",
        "12-05": "Le petit-déjeuner propose plusieurs options. La réception peut détailler le menu.",
        "13-01": "Un parking est disponible. Demandez les détails et tarifs à l'accueil.",
        "13-11": "Le WiFi est disponible dans l'hôtel.",
        "13-12": "Le code WiFi est disponible à la réception.",
        "16-10": "Pour les informations sur les animaux, contactez la réception.",
        "17-10": "Le spa propose différents services. La réception peut vous renseigner.",
        "17-31": "La salle de sport est accessible. Renseignez-vous à la réception pour les horaires.",
        "17-61": "La piscine est disponible pour nos clients. Demandez les horaires à la réception.",
        "17-62": "La piscine est ouverte. Demandez les horaires exacts à la réception.",
        "18-01": "Notre restaurant est ouvert pour le service. Contactez la réception pour plus d'informations.",
        "10-05": "Pour les conditions d'annulation, veuillez contacter notre service de réservation.",
        "10-01": "Pour toute information sur les réservations, contactez notre équipe.",
        "19-01": "L'hôtel est situé à Serris, près de Disneyland Paris. Demandez l'adresse exacte à la réception.",
    }

    return dialog_responses.get(dialog_id, "Information disponible à la réception de l'hôtel.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
