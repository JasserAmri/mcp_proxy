"""
Ki Space Custom Skill for Mirokai Robot
Queries QuickText MCP Proxy for hotel information
"""
import requests
from typing import Dict

class KiSpaceSkill:
    """
    Custom skill for Ki Space hotel (Val d'Europe)
    Connects to QuickText MCP proxy for hotel data
    """
    
    # Trigger sentences - robot activates skill when these are detected
    trigger_sentences = [
        # French
        "petit déjeuner",
        "breakfast",
        "restaurant",
        "dîner",
        "diner",
        "piscine",
        "pool",
        "spa",
        "massage",
        "parking",
        "voiture",
        "gym",
        "fitness",
        "sport",
        "wifi",
        "internet",
        "chambre",
        "room",
        "annulation",
        "annuler",
        "réservation",
        "prix",
        "tarif",
        "coût",
        # Add more as needed
    ]
    
    def __init__(self):
        """Initialize skill with proxy configuration"""
        # TODO: Replace with actual deployed proxy URL
        self.proxy_url = "http://localhost:8000/query"  # Change to https://your-proxy.railway.app/query
        self.team_id = "4577"  # Ki Space hotel ID
        self.timeout = 3  # 3 seconds timeout
        self.locale = "fr_FR"
    
    def execute(self, parameters: Dict) -> Dict:
        """
        Execute skill when trigger detected
        
        Parameters:
            parameters (dict): Contains 'query' from voice input
            
        Returns:
            dict: {
                'success': bool,
                'message': str  # Text for robot to speak
            }
        """
        query = parameters.get("query", "")
        
        if not query:
            return {
                "success": False,
                "message": "Je n'ai pas compris votre question."
            }
        
        try:
            # Call QuickText proxy
            response = requests.post(
                self.proxy_url,
                json={
                    "team_id": self.team_id,
                    "query": query,
                    "locale": self.locale
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success"):
                    # Return voice response
                    return {
                        "success": True,
                        "message": data["response"]
                    }
                else:
                    # MCP failed, fallback to ChatGPT
                    if data.get("fallback_to_chatgpt"):
                        return {
                            "success": False,
                            "message": "Laissez-moi chercher cette information autrement..."
                        }
                    else:
                        return {
                            "success": False,
                            "message": "Je n'ai pas trouvé cette information."
                        }
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except requests.Timeout:
            # Timeout - use fallback
            return {
                "success": False,
                "message": "Le service met trop de temps à répondre, laissez-moi chercher autrement."
            }
            
        except requests.ConnectionError:
            # Proxy unreachable
            return {
                "success": False,
                "message": "Je rencontre un problème de connexion au service d'information."
            }
            
        except Exception as e:
            # Generic error
            print(f"KiSpaceSkill error: {e}")
            return {
                "success": False,
                "message": "Je rencontre un problème technique, laissez-moi essayer autrement."
            }


# ============ USAGE EXAMPLE ============

if __name__ == "__main__":
    """Test skill locally"""
    
    skill = KiSpaceSkill()
    
    # Test queries
    test_queries = [
        "Quels sont les horaires du petit déjeuner?",
        "Vous avez une piscine?",
        "Le parking est-il gratuit?",
        "Quelle est la politique d'annulation?",
        "Le wifi fonctionne-t-il dans les chambres?"
    ]
    
    print("Testing Ki Space Skill...\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        result = skill.execute({"query": query})
        print(f"Success: {result['success']}")
        print(f"Response: {result['message']}")
        print("-" * 60)
