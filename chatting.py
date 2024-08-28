# Import des bibliothèques nécessaires
import faiss# Biblio pour la recherche dans les espaces de grandes dimensions
import numpy as np # Biblio pour les opérations numériques sur les tableaux
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI# Intégration de Google Generative AI pour embeddings et génération de texte

# Clé API Google Generative AI 
GOOGLE_API_KEY = 'AIzaSyAD35w9sxvo7DTnL85e0F5BsBsTH60g8xY'

# Initialiser les embeddings et le modèle de langage Google Generative AI
# 'GoogleGenerativeAIEmbeddings' crée des vecteurs d'embeddings pour les textes donnés
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=GOOGLE_API_KEY)

EMBEDDING_DIM = 768  # Dimension des embeddings

# Classe pour gérer la mémoire du chatbot avec historique de recherche
class ChatbotMemory:
    def __init__(self, embeddings, embedding_dim):
        """
        Initialise la mémoire du chatbot.

        Args:
        embeddings: Objet de type GoogleGenerativeAIEmbeddings pour générer des embeddings.
        embedding_dim: La dimension des vecteurs d'embedding.
        """
        self.embeddings = embeddings
        self.index = faiss.IndexFlatIP(embedding_dim)  # Initialise un index FAISS
        self.memory = {}  # Stocke les messages par utilisateur
        self.search_history = {}  # Stocke l'historique de recherche par utilisateur

    def remember(self, user_id, message):
        """
        Stocke un message pour un utilisateur et ajoute son embedding à l'index FAISS.

        Args:
        user_id: Identifiant unique de l'utilisateur.
        message: Le message texte de l'utilisateur.
        """
        if user_id not in self.memory:
            self.memory[user_id] = []
        self.memory[user_id].append(message)

        # Obtenir l'embedding pour le message et l'indexer avec FAISS
        embedding = self.embeddings.embed_query(message)
        embedding = np.array([embedding]).astype('float32')
        self.index.add(embedding)

    def recall(self, user_id, query, k=5):
        """
        Recherche les messages similaires en mémoire à un message de requête donné.

        Args:
        user_id: Identifiant unique de l'utilisateur.
        query: Le message de requête pour lequel on cherche des similitudes.
        k: Nombre de résultats les plus proches à récupérer (par défaut 5).

        Returns:
        Une liste des messages similaires.
        """
        # Ajouter la requête à l'historique de recherche
        if user_id not in self.search_history:
            self.search_history[user_id] = []
        self.search_history[user_id].append(query)

        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array([query_embedding]).astype('float32')

        # Rechercher les k embeddings les plus similaires dans l'index FAISS
        D, I = self.index.search(query_embedding, k)

        # Récupérer les messages correspondants aux indices I
        similar_messages = [self.memory[user_id][i] for i in I[0] if i < len(self.memory[user_id])]
        return similar_messages

    def get_search_history(self, user_id):
        """
        Récupère l'historique de recherche pour un utilisateur donné.

        Args:
        user_id: Identifiant unique de l'utilisateur.

        Returns:
        Une liste des requêtes de recherche précédentes de l'utilisateur.
        """
        return self.search_history.get(user_id, [])

# Instancier la mémoire du chatbot
memory = ChatbotMemory(embeddings, EMBEDDING_DIM)

# Fonction principale du chatbot avec recherche de similarité et génération de texte
def chatbot(user_id, message):
    """
    Gère les interactions du chatbot en stockant les messages et en générant des réponses.

    Args:
    user_id: Identifiant unique de l'utilisateur.
    message: Le message texte de l'utilisateur.

    Returns:
    Une réponse générée par le modèle de langage.
    """
    memory.remember(user_id, message)  # Stocker le message de l'utilisateur
    history = memory.recall(user_id, message)  # Récupérer les messages similaires précédents
    
    # Créer une invite de chat basée sur l'historique et le message actuel
    prompt = f"History: {' '.join(history)}\nUser: {message}"
    
    # Générer une réponse basée sur le contexte fourni
    response = llm.generate(prompts=[prompt])  # Notez que nous passons une liste de chaînes de caractères
    
    # Récupérer le texte généré par le modèle
    generated_text = response.generations[0][0].text
    
    return generated_text

# Exemple d'utilisation du chatbot
user_id = "user123"
message = "who is Kylian Mbappé?"
response = chatbot(user_id, message)
print("Chatbot:", response)

# Récupérer et afficher l'historique de recherche de l'utilisateur
search_history = memory.get_search_history(user_id)
print("Historique de recherche:", search_history)
