from fastapi import FastAPI, HTTPException, Request, Form, Depends, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
import json
import re
import markdown
from bs4 import BeautifulSoup

# Initialize FastAPI app
app = FastAPI(title="East Africa Travel Assistant")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define East African countries
EAST_AFRICAN_COUNTRIES = [
    "Kenya", "Tanzania", "Uganda", "Rwanda", "Burundi", 
    "South Sudan", "Ethiopia", "Somalia", "Djibouti", "Eritrea"
]

# Create a class for the request body
class TravelQuery(BaseModel):
    query: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

# Memory storage for conversations
# In a production app, you would use a proper database
conversation_history = {}

# Track previously given responses to avoid repetition
response_cache = {}

# Groq API configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"  # You can adjust based on Groq's available models

# Image Search API configuration with Serper API
IMAGE_SEARCH_API_KEY = "c54f8a5c85fa60faa2b21639fef95c15b05a3678"
IMAGE_SEARCH_API_URL = "https://serpapi.com/search"

# System prompt to define the assistant's behavior
SYSTEM_PROMPT = """
You are a knowledgeable and helpful travel assistant specializing exclusively in East African travel.
The East African region includes: Kenya, Tanzania, Uganda, Rwanda, Burundi, South Sudan, Ethiopia, Somalia, Djibouti, and Eritrea.

FORMAT YOUR RESPONSES:
1. Use Markdown formatting to make your responses visually appealing
2. Include clear section headers with ## for main sections and ### for subsections
3. Use bullet points and numbered lists where appropriate
4. Structure longer responses with logical sections: introduction, details, and conclusion
5. Include a "Quick Tips" section for practical advice when relevant
6. Highlight important information using **bold** text

CONTENT GUIDELINES:
- Provide detailed, accurate information about East African travel topics
- Focus ONLY on travel-related information for East Africa
- For any question not related to East African travel, politely pivot to suggest East African travel alternatives
- Avoid repetition - check previous responses and provide new information
- Include specific, actionable advice rather than generic statements
- Suggest relevant links to official tourism websites, parks, or reputable travel resources
- Mention specific tour operators, accommodations or services when appropriate

If a user is interested in a specific country or activity, remember this preference and tailor future responses to it.

Always maintain a warm, helpful tone while providing structured, well-organized information.
"""

# Function to check content similarity to avoid repetition
def is_content_similar(new_content: str, old_content: str, threshold: float = 0.6) -> bool:
    # Extract meaningful words (ignoring common words)
    def extract_keywords(text):
        text = text.lower()
        words = re.findall(r'\b[a-z]{4,}\b', text)  # Get words with 4+ letters
        # Remove common words
        stopwords = {'about', 'above', 'after', 'again', 'against', 'all', 'and', 'any', 'are', 'because', 
                    'been', 'before', 'being', 'below', 'between', 'both', 'but', 'cannot', 'could', 'did', 
                    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 
                    'have', 'having', 'here', 'how', 'into', 'itself', 'just', 'more', 'most', 'not', 'now', 
                    'off', 'once', 'only', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 
                    'should', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'then', 
                    'there', 'these', 'they', 'this', 'those', 'through', 'under', 'until', 'very', 'was', 
                    'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 
                    'would', 'your', 'yours', 'yourself', 'yourselves'}
        return [w for w in words if w not in stopwords]
    
    # Calculate Jaccard similarity
    new_keywords = set(extract_keywords(new_content))
    old_keywords = set(extract_keywords(old_content))
    
    if not new_keywords or not old_keywords:
        return False
    
    intersection = len(new_keywords.intersection(old_keywords))
    union = len(new_keywords.union(old_keywords))
    
    similarity = intersection / union
    return similarity > threshold

# Function to check if the query relates to East Africa
def is_east_africa_related(query: str) -> bool:
    query_lower = query.lower()
    
    # Check if any East African country is mentioned
    for country in EAST_AFRICAN_COUNTRIES:
        if country.lower() in query_lower:
            return True
    
    # Check for major cities, landmarks, or other East African terms
    east_africa_terms = [
        "nairobi", "dar es salaam", "kampala", "kigali", "addis ababa", "mogadishu",
        "serengeti", "masai mara", "kilimanjaro", "zanzibar", "victoria falls",
        "safari", "east africa", "swahili", "nile", "lake victoria"
    ]
    
    for term in east_africa_terms:
        if term in query_lower:
            return True
    
    # If this is a follow-up question with no specific location, we should still accept it
    if any(term in query_lower for term in ["what about", "how about", "tell me more", "and also", "what else", "then", "also"]):
        return True
    
    return False

# Get user ID from cookies or create a new one
async def get_user_id(user_id: Optional[str] = Cookie(None)):
    if not user_id:
        return str(uuid.uuid4())
    return user_id

# Get conversation ID from cookies or create a new one
async def get_conversation_id(conversation_id: Optional[str] = Cookie(None)):
    if not conversation_id:
        return str(uuid.uuid4())
    return conversation_id

# Function to get conversation history for a user
def get_conversation_history(conversation_id: str) -> List[Dict[str, Any]]:
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []
    return conversation_history[conversation_id]

# Function to add a message to conversation history
def add_to_conversation_history(conversation_id: str, role: str, content: str):
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []
    
    conversation_history[conversation_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # Limit history size (optional)
    if len(conversation_history[conversation_id]) > 20:
        conversation_history[conversation_id] = conversation_history[conversation_id][-20:]

# Function to extract topics from query
def extract_topics(query: str) -> List[str]:
    # Simple keyword extraction
    topics = []
    
    # Check for countries
    for country in EAST_AFRICAN_COUNTRIES:
        if country.lower() in query.lower():
            topics.append(country.lower())
    
    # Check for common travel topics
    travel_topics = {
        "safari": ["safari", "wildlife", "animals", "national park", "reserve", "game drive"],
        "beach": ["beach", "coast", "island", "zanzibar", "ocean", "sea", "snorkeling", "diving"],
        "mountain": ["mountain", "hiking", "trekking", "kilimanjaro", "mount kenya", "rwenzori"],
        "culture": ["culture", "tribe", "maasai", "local", "tradition", "history", "people"],
        "food": ["food", "cuisine", "eat", "restaurant", "meal", "dish", "drink", "coffee"],
        "accommodation": ["hotel", "lodge", "camp", "resort", "stay", "accommodation", "hostel"],
        "transportation": ["transport", "flight", "bus", "train", "car rental", "taxi", "uber"],
        "visa": ["visa", "entry", "passport", "border", "immigration", "requirements"],
        "cost": ["cost", "price", "budget", "expensive", "cheap", "money", "currency"],
        "safety": ["safety", "safe", "danger", "security", "crime", "health", "vaccine", "malaria"]
    }
    
    for category, keywords in travel_topics.items():
        if any(keyword in query.lower() for keyword in keywords):
            topics.append(category)
    
    return topics

# Function to enhance response with formatting and links
def enhance_response(response_text: str) -> str:
    # Check if response is already in markdown
    if not any(marker in response_text for marker in ['##', '**', '- ', '1. ']):
        # Convert to a simple markdown structure
        lines = response_text.split('\n')
        enhanced_text = []
        
        in_list = False
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                enhanced_text.append(line)
                continue
                
            # Check if this might be a header
            if i == 0 or (i > 0 and not lines[i-1].strip()):
                if not line.endswith('?') and not line.endswith('.') and len(line) < 60:
                    enhanced_text.append(f"## {line}")
                    continue
            
            # Check if this might be a list item
            if line.startswith('- ') or re.match(r'^\d+\.', line):
                in_list = True
                enhanced_text.append(line)
            elif in_list and line.startswith(' '):
                enhanced_text.append(line)
            else:
                in_list = False
                # Look for potential emphasis
                words = line.split()
                if len(words) > 3:
                    important_words = ["must", "best", "top", "recommend", "popular", "famous", 
                                      "important", "essential", "crucial", "key"]
                    for i, word in enumerate(words):
                        if word.lower() in important_words and i < len(words) - 1:
                            phrase = " ".join(words[i:i+min(3, len(words)-i)])
                            line = line.replace(phrase, f"**{phrase}**")
                enhanced_text.append(line)
        
        response_text = '\n'.join(enhanced_text)
    
    # Add travel resource links if not present
    if "http" not in response_text:
        # Extract topics to determine relevant links
        soup = BeautifulSoup(markdown.markdown(response_text), 'html.parser')
        content_text = soup.get_text()
        
        # Determine which country is being discussed
        country_mentioned = None
        for country in EAST_AFRICAN_COUNTRIES:
            if country.lower() in content_text.lower():
                country_mentioned = country
                break
        
        # Add contextual links
        if country_mentioned:
            links_section = f"\n\n### Helpful Resources for {country_mentioned}\n"
            
            if country_mentioned == "Kenya":
                links_section += "- [Kenya Tourism Board](https://magicalkenya.com) - Official travel information\n"
                links_section += "- [Kenya Wildlife Service](https://www.kws.go.ke) - National parks and reserves\n"
            elif country_mentioned == "Tanzania":
                links_section += "- [Tanzania Tourism Board](https://www.tanzaniatourism.go.tz) - Official travel guide\n"
                links_section += "- [Tanzania National Parks](https://www.tanzaniaparks.go.tz) - Information on all national parks\n"
            elif country_mentioned == "Uganda":
                links_section += "- [Uganda Tourism Board](https://utb.go.ug) - Official tourism portal\n"
                links_section += "- [Uganda Wildlife Authority](https://www.ugandawildlife.org) - Wildlife and national parks\n"
            elif country_mentioned == "Rwanda":
                links_section += "- [Visit Rwanda](https://www.visitrwanda.com) - Official travel information\n"
                links_section += "- [Rwanda Development Board](https://rdb.rw/visit-rwanda) - Tourism resources\n"
            elif country_mentioned == "Ethiopia":
                links_section += "- [Ethiopia Tourism](https://www.moct.gov.et) - Official tourism information\n"
                links_section += "- [Ethiopian Airlines](https://www.ethiopianairlines.com) - National carrier with tourism packages\n"
            else:
                links_section += f"- [Lonely Planet: {country_mentioned}](https://www.lonelyplanet.com/search?q={country_mentioned.lower()}) - Travel guides and information\n"
                links_section += f"- [Wikitravel: {country_mentioned}](https://wikitravel.org/en/{country_mentioned}) - Community travel guide\n"
            
            response_text += links_section
    
    return response_text

# Function to check if response contains repetitive information
def avoid_repetition(conversation_id: str, response_text: str) -> str:
    # Get previous responses
    history = get_conversation_history(conversation_id)
    previous_responses = [msg["content"] for msg in history if msg["role"] == "assistant"]
    
    if not previous_responses:
        return response_text
    
    # Check if new response is similar to any previous response
    for prev_response in previous_responses:
        if is_content_similar(response_text, prev_response):
            # If similar, add a note to avoid repetition
            note = "\n\n**Note:** I've shared information about this topic before. Here's some additional perspective:"
            
            # Extract topics to provide alternative information
            soup = BeautifulSoup(markdown.markdown(response_text), 'html.parser')
            content_text = soup.get_text()
            
            # Determine which country is being discussed
            country_mentioned = None
            for country in EAST_AFRICAN_COUNTRIES:
                if country.lower() in content_text.lower():
                    country_mentioned = country
                    break
            
            if country_mentioned:
                alternatives = {
                    "Kenya": ["Try exploring the less-visited northern regions like Samburu or Lake Turkana.", 
                             "Consider combining your visit with community-based tourism experiences."],
                    "Tanzania": ["Beyond the popular northern circuit, southern Tanzania offers uncrowded wildlife viewing.", 
                                "The southern highlands around Iringa offer beautiful landscapes and cultural experiences."],
                    "Uganda": ["Eastern Uganda's Mount Elgon region offers hiking and unique coffee experiences.", 
                              "The Karamoja region provides authentic cultural immersion opportunities."],
                    "Rwanda": ["While gorilla trekking is popular, Akagera National Park offers excellent safari experiences.", 
                              "Lake Kivu provides beautiful beaches and water activities."],
                    "Ethiopia": ["The Danakil Depression is one of Earth's most unique landscapes.", 
                                "The ancient city of Harar offers a glimpse into Ethiopia's diverse cultural heritage."]
                }
                
                if country_mentioned in alternatives:
                    note += f"\n\n### Alternative Experiences in {country_mentioned}\n"
                    for alt in alternatives[country_mentioned]:
                        note += f"- {alt}\n"
                
            return note + "\n\n" + response_text
    
    return response_text

# Function to call the Groq API with conversation history
async def query_groq_api(travel_query: str, conversation_id: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Get conversation history for context
    history = get_conversation_history(conversation_id)
    
    # Construct messages with history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add up to 10 most recent message exchanges
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the current query
    messages.append({"role": "user", "content": travel_query})
    
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GROQ_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data["choices"][0]["message"]["content"]
            
            # Enhance the response
            enhanced_response = enhance_response(response_text)
            
            # Check for repetition
            final_response = avoid_repetition(conversation_id, enhanced_response)
            
            return final_response
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Groq API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error querying Groq API: {str(e)}")

# Function to search for images using Serper API
async def search_images(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    # Add "East Africa" to query to improve relevance unless already included
    if "east africa" not in query.lower():
        search_query = f"{query} East Africa"
    else:
        search_query = query
        
    params = {
        "q": search_query,
        "engine": "google_images",
        "api_key": IMAGE_SEARCH_API_KEY,
        "num": limit
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(IMAGE_SEARCH_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant image data from response
            images = []
            if "images_results" in data:
                for img in data["images_results"][:limit]:
                    images.append({
                        "thumbnail": img.get("thumbnail"),
                        "original": img.get("original"),
                        "title": img.get("title", ""),
                        "source": img.get("source", ""),
                        "source_url": img.get("source_page", "")
                    })
            return images
        except Exception as e:
            print(f"Error searching images: {str(e)}")
            return []

# Function to extract relevant travel links based on query content
def extract_relevant_links(query: str, response: str) -> List[Dict[str, Any]]:
    links = []
    
    # Determine topics from query and response
    topics = set(extract_topics(query + " " + response))
    
    # General East Africa travel resources
    general_links = [
        {
            "url": "https://www.eastafricatourismplatform.org/",
            "title": "East Africa Tourism Platform",
            "description": "Official platform for regional tourism information"
        },
        {
            "url": "https://www.lonelyplanet.com/east-africa",
            "title": "Lonely Planet: East Africa",
            "description": "Comprehensive travel guides and tips"
        }
    ]
    
    # Country-specific resources
    country_links = {
        "kenya": [
            {
                "url": "https://magicalkenya.com/",
                "title": "Magical Kenya",
                "description": "Official Kenya tourism portal"
            },
            {
                "url": "https://www.kws.go.ke/",
                "title": "Kenya Wildlife Service",
                "description": "Information on national parks and wildlife conservation"
            }
        ],
        "tanzania": [
            {
                "url": "https://www.tanzaniatourism.go.tz/",
                "title": "Tanzania Tourism",
                "description": "Official Tanzania tourism website"
            },
            {
                "url": "https://www.tanzaniaparks.go.tz/",
                "title": "Tanzania National Parks",
                "description": "Information on Tanzania's national parks"
            }
        ],
        "uganda": [
            {
                "url": "https://utb.go.ug/",
                "title": "Visit Uganda",
                "description": "Official tourism portal for Uganda"
            },
            {
                "url": "https://www.ugandawildlife.org/",
                "title": "Uganda Wildlife Authority",
                "description": "Information on Uganda's wildlife and national parks"
            }
        ],
        "rwanda": [
            {
                "url": "https://www.visitrwanda.com/",
                "title": "Visit Rwanda",
                "description": "Official tourism website for Rwanda"
            }
        ],
        "ethiopia": [
            {
                "url": "https://www.ethiopia.travel/",
                "title": "Ethiopia Tourism",
                "description": "Official tourism website for Ethiopia"
            }
        ]
    }
    
    # Topic-specific resources
    topic_links = {
        "safari": [
            {
                "url": "https://www.safaribookings.com/east-africa",
                "title": "SafariBookings: East Africa",
                "description": "Compare and book safari tours"
            }
        ],
        "beach": [
            {
                "url": "https://www.zanzibartourism.go.tz/",
                "title": "Zanzibar Tourism",
                "description": "Official tourism website for Zanzibar"
            }
        ],
        "mountain": [
            {
                "url": "https://www.kilimanjaroparks.go.tz/",
                "title": "Mount Kilimanjaro National Park",
                "description": "Information on climbing Kilimanjaro"
            }
        ],
        "visa": [
            {
                "url": "https://www.visiteastafrica.org/visa-information",
                "title": "East Africa Visa Information",
                "description": "Information on East African tourist visas"
            }
        ],
        "safety": [
            {
                "url": "https://www.gov.uk/foreign-travel-advice",
                "title": "UK Foreign Travel Advice",
                "description": "Travel safety information for East African countries"
            }
        ]
    }
    
    # Add general links
    links.extend(general_links)
    
    # Add country-specific links
    for country in EAST_AFRICAN_COUNTRIES:
        if country.lower() in topics and country.lower() in country_links:
            links.extend(country_links[country.lower()])
    
    # Add topic-specific links
    for topic in topics:
        if topic in topic_links:
            links.extend(topic_links[topic])
    
    # Return a maximum of 5 most relevant links
    return links[:5]

# Main route for the web interface
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user_id: str = Depends(get_user_id), conversation_id: str = Depends(get_conversation_id)):
    # Get conversation history
    history = get_conversation_history(conversation_id)
    
    response = templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "conversation_history": history,
            "user_id": user_id,
            "conversation_id": conversation_id
        }
    )
    
    # Set cookies if they don't exist
    if not request.cookies.get("user_id"):
        response.set_cookie(key="user_id", value=user_id)
    if not request.cookies.get("conversation_id"):
        response.set_cookie(key="conversation_id", value=conversation_id)
    
    return response

# Route to handle API queries
@app.post("/query")
async def handle_query(travel_query: TravelQuery, user_id: str = Depends(get_user_id), conversation_id: str = Depends(get_conversation_id)):
    # Override user_id and conversation_id if provided in the request
    if travel_query.user_id:
        user_id = travel_query.user_id
    if travel_query.conversation_id:
        conversation_id = travel_query.conversation_id
    
    query = travel_query.query.strip()
    
    # Skip processing for empty queries
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Add user message to conversation history
    add_to_conversation_history(conversation_id, "user", query)
    
    # Check if the query is related to East Africa
    is_related = is_east_africa_related(query)
    
    if not is_related:
        # For non-East Africa queries, gently redirect
        response_text = f"""
## East Africa Travel Focus

I specialize in travel information for East African countries including Kenya, Tanzania, Uganda, Rwanda, and others in the region.

It seems your question about "{query}" may not be directly related to East African travel. 

**Would you like me to suggest some popular travel destinations or experiences in East Africa instead?** 

I'd be happy to help with:
- Safari experiences in the Serengeti or Masai Mara
- Mountain trekking options like Kilimanjaro 
- Beach destinations along the Indian Ocean
- Cultural experiences with local communities
- Travel planning tips for the region

Just let me know what aspects of East African travel interest you!
"""
    else:
        # Query the Groq API for East Africa related queries
        response_text = await query_groq_api(query, conversation_id)
    
    # Add assistant response to conversation history
    add_to_conversation_history(conversation_id, "assistant", response_text)
    
    # Get images related to the query if it's East Africa related
    images = []
    if is_related:
        images = await search_images(query)
    
    # Extract relevant links
    links = extract_relevant_links(query, response_text)
    
    return {
        "response": response_text,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "images": images,
        "links": links,
        "timestamp": datetime.now().isoformat()
    }

# Route to reset conversation
@app.post("/reset")
async def reset_conversation(request: Request, user_id: str = Depends(get_user_id)):
    # Generate a new conversation ID
    new_conversation_id = str(uuid.uuid4())
    
    # Return response with new conversation ID
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="conversation_id", value=new_conversation_id)
    
    return response

# Route to get conversation history
@app.get("/history/{conversation_id}")
async def get_history(conversation_id: str):
    history = get_conversation_history(conversation_id)
    return {"history": history}

# Function to generate HTML template for the chat interface
def generate_chat_interface():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>East Africa Travel Assistant</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <link rel="stylesheet" href="/static/css/styles.css">
    </head>
    <body class="bg-gray-100 min-h-screen">
        <div class="container mx-auto p-4">
            <header class="bg-green-700 text-white p-4 rounded-lg shadow-md mb-6">
                <h1 class="text-2xl font-bold">East Africa Travel Assistant</h1>
                <p class="text-sm">Your expert guide to Kenya, Tanzania, Uganda, Rwanda, and more</p>
            </header>
            
            <div class="flex flex-col md:flex-row gap-6">
                <!-- Chat section -->
                <div class="w-full md:w-2/3 bg-white rounded-lg shadow-md p-4">
                    <div id="chat-container" class="h-96 overflow-y-auto mb-4 p-2">
                        <div class="mb-4">
                            <div class="bg-green-100 p-3 rounded-lg inline-block max-w-3xl">
                                <p class="text-sm text-green-800">
                                    Hello! I'm your East Africa Travel Assistant. I can help you with information about:
                                </p>
                                <ul class="list-disc pl-5 text-sm text-green-800 mt-2">
                                    <li>Safari destinations like Serengeti and Masai Mara</li>
                                    <li>Beach getaways in Zanzibar and the East African coast</li>
                                    <li>Mountain trekking including Kilimanjaro</li>
                                    <li>Cultural experiences and local cuisine</li>
                                    <li>Travel planning, safety, and visa information</li>
                                </ul>
                                <p class="text-sm text-green-800 mt-2">
                                    What would you like to know about travel in East Africa?
                                </p>
                            </div>
                        </div>
                        <!-- Chat messages will be inserted here -->
                    </div>
                    
                    <form id="query-form" class="flex flex-col space-y-2">
                        <textarea id="query-input" class="w-full p-2 border border-gray-300 rounded-lg" 
                                rows="3" placeholder="Ask me about travel in East Africa..."></textarea>
                        <div class="flex justify-between">
                            <button type="button" id="reset-btn" class="bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600">
                                New Conversation
                            </button>
                            <button type="submit" class="bg-green-700 text-white px-4 py-2 rounded-lg hover:bg-green-800">
                                Send
                            </button>
                        </div>
                    </form>
                </div>
                
                <!-- Info section -->
                <div class="w-full md:w-1/3">
                    <!-- Image gallery -->
                    <div id="image-gallery" class="bg-white rounded-lg shadow-md p-4 mb-6 hidden">
                        <h2 class="text-lg font-semibold mb-2 text-green-800">Related Images</h2>
                        <div id="images-container" class="grid grid-cols-2 gap-2">
                            <!-- Images will be inserted here -->
                        </div>
                    </div>
                    
                    <!-- Helpful links -->
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <h2 class="text-lg font-semibold mb-2 text-green-800">Helpful Resources</h2>
                        <div id="links-container">
                            <ul class="list-disc pl-5 space-y-2">
                                <li><a href="https://www.magicalkenya.com" target="_blank" class="text-blue-600 hover:underline">Magical Kenya</a></li>
                                <li><a href="https://www.tanzaniatourism.go.tz" target="_blank" class="text-blue-600 hover:underline">Tanzania Tourism</a></li>
                                <li><a href="https://www.visituganda.com" target="_blank" class="text-blue-600 hover:underline">Visit Uganda</a></li>
                                <li><a href="https://www.visitrwanda.com" target="_blank" class="text-blue-600 hover:underline">Visit Rwanda</a></li>
                                <li><a href="https://www.safaribookings.com" target="_blank" class="text-blue-600 hover:underline">Safari Bookings</a></li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <script src="/static/js/script.js"></script>
    </body>
    </html>
    """

# Create static/js/script.js 
def generate_script_js():
    return """
    document.addEventListener('DOMContentLoaded', function() {
        const queryForm = document.getElementById('query-form');
        const queryInput = document.getElementById('query-input');
        const chatContainer = document.getElementById('chat-container');
        const resetBtn = document.getElementById('reset-btn');
        const imageGallery = document.getElementById('image-gallery');
        const imagesContainer = document.getElementById('images-container');
        const linksContainer = document.getElementById('links-container');
        
        // Function to add a message to the chat
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'mb-4 ' + (isUser ? 'text-right' : '');
            
            const bubble = document.createElement('div');
            bubble.className = isUser 
                ? 'bg-blue-100 p-3 rounded-lg inline-block max-w-3xl'
                : 'bg-green-100 p-3 rounded-lg inline-block max-w-3xl';
            
            if (isUser) {
                // For user messages, just add the text
                bubble.textContent = content;
            } else {
                // For assistant messages, render the markdown
                bubble.innerHTML = marked.parse(content);
                // Add target="_blank" to all links
                bubble.querySelectorAll('a').forEach(link => {
                    link.setAttribute('target', '_blank');
                });
            }
            
            messageDiv.appendChild(bubble);
            chatContainer.appendChild(messageDiv);
            
            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Function to display images
        function displayImages(images) {
            if (images && images.length > 0) {
                imageGallery.classList.remove('hidden');
                imagesContainer.innerHTML = '';
                
                images.forEach(image => {
                    const imgDiv = document.createElement('div');
                    imgDiv.className = 'aspect-w-16 aspect-h-9';
                    
                    const img = document.createElement('img');
                    img.src = image.thumbnail;
                    img.alt = image.title;
                    img.className = 'object-cover rounded w-full h-32';
                    
                    const link = document.createElement('a');
                    link.href = image.source_url;
                    link.target = '_blank';
                    link.appendChild(img);
                    
                    imgDiv.appendChild(link);
                    imagesContainer.appendChild(imgDiv);
                });
            } else {
                imageGallery.classList.add('hidden');
            }
        }
        
        // Function to display links
        function displayLinks(links) {
            if (links && links.length > 0) {
                linksContainer.innerHTML = '<ul class="list-disc pl-5 space-y-2"></ul>';
                const linksList = linksContainer.querySelector('ul');
                
                links.forEach(link => {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.href = link.url;
                    a.className = 'text-blue-600 hover:underline';
                    a.target = '_blank';
                    a.textContent = link.title;
                    
                    li.appendChild(a);
                    
                    if (link.description) {
                        const desc = document.createElement('p');
                        desc.className = 'text-xs text-gray-600 mt-1';
                        desc.textContent = link.description;
                        li.appendChild(desc);
                    }
                    
                    linksList.appendChild(li);
                });
            }
        }
        
        // Handle form submission
        queryForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const query = queryInput.value.trim();
            if (!query) return;
            
            // Add user message to chat
            addMessage(query, true);
            
            // Clear input
            queryInput.value = '';
            
            // Show loading indicator
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'mb-4';
            loadingDiv.innerHTML = `
                <div class="bg-green-100 p-3 rounded-lg inline-block">
                    <p class="text-green-800">Thinking...</p>
                </div>
            `;
            chatContainer.appendChild(loadingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            try {
                // Send query to API
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: query
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to get response');
                }
                
                const data = await response.json();
                
                // Remove loading indicator
                chatContainer.removeChild(loadingDiv);
                
                // Add assistant response to chat
                addMessage(data.response);
                
                // Display images if available
                displayImages(data.images);
                
                // Display links if available
                displayLinks(data.links);
                
            } catch (error) {
                console.error('Error:', error);
                
                // Remove loading indicator
                chatContainer.removeChild(loadingDiv);
                
                // Show error message
                addMessage('Sorry, I encountered an error. Please try again.');
            }
        });
        
        // Handle reset button
        resetBtn.addEventListener('click', async function() {
            try {
                const response = await fetch('/reset', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    // Reload the page to start fresh
                    window.location.reload();
                }
            } catch (error) {
                console.error('Error resetting conversation:', error);
            }
        });
        
        // Load conversation history
        async function loadHistory() {
            try {
                // Get conversation ID from cookie
                const cookies = document.cookie.split(';').reduce((acc, cookie) => {
                    const [key, value] = cookie.trim().split('=');
                    acc[key] = value;
                    return acc;
                }, {});
                
                const conversationId = cookies['conversation_id'];
                
                if (!conversationId) return;
                
                const response = await fetch(`/history/${conversationId}`);
                if (!response.ok) return;
                
                const data = await response.json();
                
                if (data.history && data.history.length > 0) {
                    // Clear default welcome message
                    chatContainer.innerHTML = '';
                    
                    // Add messages from history
                    data.history.forEach(msg => {
                        addMessage(msg.content, msg.role === 'user');
                    });
                }
            } catch (error) {
                console.error('Error loading history:', error);
            }
        }
        
        // Load history when page loads
        loadHistory();
    });
    """

# Create static/css/styles.css
def generate_styles_css():
    return """
    /* Custom styles for the East Africa Travel Assistant */
    
    /* Override some markdown styles */
    h2, h3, h4 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #276749; /* text-green-800 */
    }
    
    h2 {
        font-size: 1.25rem;
    }
    
    h3 {
        font-size: 1.125rem;
    }
    
    ul, ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    
    p {
        margin-bottom: 0.5rem;
    }
    
    /* Image hover effect */
    #images-container img {
        transition: transform 0.2s;
    }
    
    #images-container img:hover {
        transform: scale(1.05);
    }
    
    /* Chat container styling */
    #chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    #chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    #chat-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 3px;
    }
    
    #chat-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Add aspect ratio for image gallery */
    .aspect-w-16 {
        position: relative;
        padding-bottom: 56.25%;
    }
    
    .aspect-w-16 > * {
        position: absolute;
        height: 100%;
        width: 100%;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        #images-container {
            grid-template-columns: repeat(1, 1fr);
        }
    }
    """

# Main execution
def setup_app():
    # Generate templates directory and index.html
    import os
    
    # Create directories if they don't exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    # Write template files
    with open("templates/index.html", "w") as f:
        f.write(generate_chat_interface())
    
    with open("static/js/script.js", "w") as f:
        f.write(generate_script_js())
    
    with open("static/css/styles.css", "w") as f:
        f.write(generate_styles_css())

if __name__ == "__main__":
    import uvicorn
    
    # Setup app files
    setup_app()
    
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8000)