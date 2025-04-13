from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import re
import html
from dotenv import load_dotenv
import markdown
import bleach
import time
import json
import logging
import hashlib
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("elan_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("elan_server")

# Load environment variables
load_dotenv()

GROQCLOUD_API_KEY = os.getenv("GROQCLOUD_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("CSE_ID")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

app = Flask(__name__)
CORS(app)

# Add creator information
CREATOR = "Pranay Tadakamalla"

# Cache for recent responses to avoid duplicates
response_cache = {}
# Cache for web search results to reduce API calls
web_search_cache = {}
# Cache expiration time (in seconds)
CACHE_EXPIRATION = 3600  # 1 hour

def markdown_to_html(text):
    """Convert markdown to HTML with proper formatting for headings, italics, etc."""
    # Process the markdown to HTML with the 'extra' extension to allow raw HTML
    # Also add 'nl2br' to convert newlines to <br> tags
    html_content = markdown.markdown(text, extensions=['extra', 'nl2br'])
    
    # Use bleach to allow specific HTML tags but sanitize potentially harmful ones
    allowed_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'ul', 'ol', 
                   'li', 'strong', 'em', 'b', 'i', 'br', 'code', 'pre', 'hr',
                   'div', 'span', 'table', 'thead', 'tbody', 'tr', 'th', 'td']
    allowed_attributes = {
        'a': ['href', 'title', 'target'],
        'img': ['src', 'alt', 'title', 'width', 'height'],
        '*': ['class', 'id', 'style']
    }
    
    clean_html = bleach.clean(html_content, 
                             tags=allowed_tags, 
                             attributes=allowed_attributes, 
                             strip=True)
    
    return clean_html

def fetch_from_groq(user_message):
    """Fetch response from Groq API with improved error handling and retries"""
    cache_key = f"groq_{hashlib.md5(user_message.encode()).hexdigest()}"
    if cache_key in response_cache:
        cache_entry = response_cache[cache_key]
        if time.time() - cache_entry['timestamp'] < CACHE_EXPIRATION:
            logger.info("Using cached Groq response")
            return cache_entry['response']
    
    headers = {
        "Authorization": f"Bearer {GROQCLOUD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = f"""
    You are ELAN, an AI Assistant specializing in Sports, Fitness, Mental Well-Being, Meditation, Yoga, and Games.
    ELAN was created by {CREATOR}.
    
    IMPORTANT GUIDELINES:
    1. Always provide COMPLETE answers. Never cut off mid-sentence.
    2. If you're listing items or explaining steps, always complete the full list or all steps.
    3. Use proper HTML formatting with headings (<h1> for main headings, <h2> for subheadings)
    4. Use <em> for emphasis and <strong> for important points
    5. Create well-structured responses with clear sections
    6. Use <ul> and <li> for lists
    7. ALWAYS check your response to ensure it's complete before sending
    
    If you don't know an answer or if the information might be after your training data (2021), 
    return 'NEEDS_WEB_SEARCH' so I can look it up for you.
    
    Only answer questions related to Sports, Fitness, Mental Well-Being, Meditation, Yoga, and Games.
    For any other topics, return 'OUT_OF_SCOPE'.
    
    If someone asks who created you or who you were developed by, always mention:
    "I was created by {CREATOR}."
    
    If someone asks who you are or what your name is, respond with:
    "<h1>I am ELAN</h1>
    
    <p>I'm your AI assistant specializing in Sports, Fitness, Mental Well-Being, Meditation, Yoga, and Games. I was created by {CREATOR}. How can I help you today?</p>"
    """
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 1000,  # Increased token limit for more complete responses
        "temperature": 0.5
    }

    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending request to Groq API (attempt {attempt+1}/{max_retries})")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                json=payload, 
                headers=headers,
                timeout=20  # 20 second timeout
            )
            response.raise_for_status()
            response_json = response.json()

            choices = response_json.get("choices", [])
            if choices and "message" in choices[0]:
                ai_response = choices[0]["message"]["content"].strip()
                
                # Check for special response codes
                if "NEEDS_WEB_SEARCH" in ai_response:
                    logger.info("Groq indicated web search needed")
                    return None
                if "OUT_OF_SCOPE" in ai_response:
                    logger.info("Groq indicated query is out of scope")
                    return "OUT_OF_SCOPE"
                    
                # Check for phrases indicating knowledge cutoff
                knowledge_cutoff_phrases = [
                    "i don't know", "i am just an llm", "my knowledge is limited", 
                    "my training data", "as of my last update", "as of 2021", 
                    "i don't have information", "i cannot provide", "i'm not able to"
                ]
                
                if any(phrase in ai_response.lower() for phrase in knowledge_cutoff_phrases):
                    logger.info("Groq response indicated knowledge cutoff")
                    return None
                
                # Check if response appears incomplete
                incomplete_indicators = [
                    "...", "to be continued", "I'll continue", "Next, ", "Additionally, "
                ]
                
                ends_with_incomplete = any(ai_response.rstrip().endswith(indicator) for indicator in incomplete_indicators)
                if ends_with_incomplete:
                    logger.warning("Response appears incomplete, requesting continuation")
                    
                    # Request continuation
                    continuation_payload = {
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": ai_response},
                            {"role": "user", "content": "Please continue your response and complete it."}
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.5
                    }
                    
                    continuation_response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions", 
                        json=continuation_payload, 
                        headers=headers,
                        timeout=20
                    )
                    continuation_response.raise_for_status()
                    
                    continuation_json = continuation_response.json()
                    continuation_choices = continuation_json.get("choices", [])
                    
                    if continuation_choices and "message" in continuation_choices[0]:
                        continuation_text = continuation_choices[0]["message"]["content"].strip()
                        # Combine original response with continuation
                        ai_response = ai_response + " " + continuation_text
                
                # Cache the response
                response_cache[cache_key] = {
                    'response': ai_response,
                    'timestamp': time.time()
                }
                
                logger.info("Successfully received response from Groq")
                return ai_response
                
            logger.warning("Unexpected response format from Groq")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error with Groq API: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("Max retries reached for Groq API")
                return None
        except Exception as e:
            logger.error(f"Unexpected error with Groq API: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("Max retries reached for Groq API")
                return None
                
    return None

def fetch_web_results(query, is_current_event=False):
    """Fetch search results from Google or SerpAPI with caching and improved filtering"""
    cache_key = f"web_search_{hashlib.md5(query.encode()).hexdigest()}"
    if cache_key in web_search_cache:
        cache_entry = web_search_cache[cache_key]
        # For current events, use a shorter cache expiration (1 hour)
        cache_duration = 1800 if is_current_event else CACHE_EXPIRATION  # 30 minutes for current events
        if time.time() - cache_entry['timestamp'] < cache_duration:
            logger.info("Using cached web search results")
            return cache_entry['results']
    
    # For current events, add date specificity to the query
    if is_current_event:
        # Add date context if not already present
        date_terms = ["today", "yesterday", "this week", "this month", "this year", 
                     "2024", "2025", "recent", "latest"]
        if not any(term in query.lower() for term in date_terms):
            query = f"{query} latest"
            logger.info(f"Enhanced query with date context: {query}")
    
    # Try Google Custom Search first
    try:
        logger.info("Fetching results from Google Custom Search")
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query, 
            "key": GOOGLE_API_KEY, 
            "cx": CSE_ID, 
            "num": 8,  # Request more results to filter
            "sort": "date" if is_current_event else None  # Sort by date for current events
        }
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            
            # Skip results that don't have all required fields
            if not (title and link and snippet):
                continue
                
            # Skip results from low-quality sources or aggregators
            low_quality_domains = ["pinterest", "quora", "reddit", "facebook", "twitter", 
                                  "instagram", "tiktok", "snapchat"]
            if any(domain in link.lower() for domain in low_quality_domains):
                continue
                
            # For current events, apply additional filtering
            if is_current_event:
                # Skip results that don't mention specific details
                current_event_keywords = ["vs", "beat", "won", "lost", "score", "match", "game", "result", 
                                         "tournament", "championship", "competition"]
                if not any(keyword in title.lower() or keyword in snippet.lower() for keyword in current_event_keywords):
                    continue
            
            results.append({
                "title": title,
                "link": link,
                "snippet": snippet,
                "source": "Google"
            })
            
        if len(results) >= 3:  # Ensure we have at least 3 results
            # Cache the results
            web_search_cache[cache_key] = {
                'results': results[:5],  # Limit to top 5
                'timestamp': time.time()
            }
            return results[:5]
            
    except Exception as e:
        logger.error(f"Google Search API error: {str(e)}")

    # Fall back to SerpAPI if Google fails
    try:
        logger.info("Falling back to SerpAPI")
        search_url = "https://serpapi.com/search.json"
        params = {
            "q": query, 
            "location": "India", 
            "hl": "en", 
            "gl": "in", 
            "google_domain": "google.co.in", 
            "api_key": SERPAPI_KEY,
            "tbs": "qdr:d" if is_current_event else None  # Limit to past day for current events
        }
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for result in data.get("organic_results", []):
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            
            # Skip results that don't have all required fields
            if not (title and link and snippet):
                continue
                
            # Skip results from low-quality sources or aggregators
            low_quality_domains = ["pinterest", "quora", "reddit", "facebook", "twitter", 
                                  "instagram", "tiktok", "snapchat"]
            if any(domain in link.lower() for domain in low_quality_domains):
                continue
                
            # For current events, apply additional filtering
            if is_current_event:
                # Skip results that don't mention specific details
                current_event_keywords = ["vs", "beat", "won", "lost", "score", "match", "game", "result", 
                                         "tournament", "championship", "competition"]
                if not any(keyword in title.lower() or keyword in snippet.lower() for keyword in current_event_keywords):
                    continue
            
            results.append({
                "title": title,
                "link": link,
                "snippet": snippet,
                "source": "SerpAPI"
            })
        
        if results:
            # Cache the results
            web_search_cache[cache_key] = {
                'results': results[:5],  # Limit to top 5
                'timestamp': time.time()
            }
            
        return results[:5] if results else []
    except Exception as e:
        logger.error(f"SerpAPI error: {str(e)}")
        return []

def process_sports_results(query, results):
    """Process sports results to ensure consistency and accuracy"""
    if not results or len(results) == 0:
        return None
        
    # Extract key information from the query
    query_lower = query.lower()
    
    # For sports match results, we need special handling
    if "match" in query_lower or "game" in query_lower or "vs" in query_lower or "score" in query_lower:
        logger.info("Processing sports match results")
        
        # Extract match information from search results
        matches = []
        for result in results:
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            content = title + " " + snippet
            
            # Look for team names
            teams_found = []
            
            # Extract teams using "vs" pattern
            vs_pattern = r"([a-z\s]+)\s+vs\s+([a-z\s]+)"
            vs_matches = re.findall(vs_pattern, content)
            
            for vs_match in vs_matches:
                team1 = vs_match[0].strip()
                team2 = vs_match[1].strip()
                if team1 and team1 not in teams_found:
                    teams_found.append(team1)
                if team2 and team2 not in teams_found:
                    teams_found.append(team2)
            
            # Look for winner
            winner = None
            win_patterns = [
                r"([a-z\s]+)\s+won",
                r"([a-z\s]+)\s+beat",
                r"([a-z\s]+)\s+defeated",
                r"([a-z\s]+)\s+triumph",
                r"victory\s+for\s+([a-z\s]+)",
                r"win\s+for\s+([a-z\s]+)"
            ]
            
            for pattern in win_patterns:
                win_matches = re.findall(pattern, content)
                if win_matches:
                    potential_winner = win_matches[0].strip()
                    # Ensure it's a team name
                    if potential_winner and len(potential_winner) > 2:
                        winner = potential_winner
                        break
            
            # Look for score
            score = None
            score_patterns = [
                r"(\d+)[^\d]+(\d+)",  # Simple score pattern like "3-2" or "3 - 2"
                r"score\s+of\s+(\d+)[^\d]+(\d+)",
                r"(\d+)\s*-\s*(\d+)\s+victory",
                r"won\s+(\d+)[^\d]+(\d+)",
                r"(\d+)[^\d]+(\d+)\s+score"
            ]
            
            for pattern in score_patterns:
                score_matches = re.findall(pattern, content)
                if score_matches:
                    score = f"{score_matches[0][0]}-{score_matches[0][1]}"
                    break
            
            # If we found teams and either a winner or score, add to matches
            if teams_found and (winner or score):
                matches.append({
                    "teams": teams_found,
                    "winner": winner,
                    "score": score,
                    "source": result.get("link", "")
                })
        
        if matches:
            return matches
    
    # For other sports information, just return the formatted results
    return results

def summarize_web_results(query, results, is_current_event=False):
    """Generate a summary from web search results using Groq LLM"""
    if not results:
        return "I couldn't find any reliable information on this topic."
    
    # Prepare the content to summarize
    context = f"Here are search results for '{query}':\n\n"
    for i, result in enumerate(results, 1):
        context += f"Source {i}: {result['title']}\n"
        context += f"URL: {result['link']}\n"
        context += f"Excerpt: {result['snippet']}\n\n"
    
    # For current events, add special instructions
    if is_current_event:
        context += "This is likely a CURRENT EVENT. Please focus on the most recent information."
    
    # Special processing for sports results
    if any(keyword in query.lower() for keyword in ["score", "game", "match", "vs", "versus", "tournament", "championship"]):
        sports_processed = process_sports_results(query, results)
        if sports_processed and isinstance(sports_processed, list) and "teams" in sports_processed[0]:
            # Format the sports results specially
            context += "\n\nThese results appear to be about sports matches. Here's the extracted information:\n\n"
            for match in sports_processed:
                teams_str = " vs ".join(match["teams"])
                context += f"Match: {teams_str}\n"
                if match.get("winner"):
                    context += f"Winner: {match['winner']}\n"
                if match.get("score"):
                    context += f"Score: {match['score']}\n"
                context += f"Source: {match['source']}\n\n"
    
    headers = {
        "Authorization": f"Bearer {GROQCLOUD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = f"""
    You are an AI assistant specializing in summarizing web search results.
    You were created by {CREATOR}.
    
    Guidelines for summarizing web search results:
    1. Be concise but comprehensive, focusing on the key information that answers the query.
    2. Use a neutral, objective tone that presents facts rather than opinions.
    3. When sources contradict each other, present the different viewpoints and note the discrepancy.
    4. Include specific details, numbers, dates, and names when relevant.
    5. For sports scores and match results, clearly state the teams, score, and outcome.
    6. Format your response with HTML formatting:
       - Use <h2> and <h3> for section headings
       - Use <strong> for important points
       - Use <em> for emphasis
       - Use <ul> and <li> for lists
    7. Always include a "Sources" section at the end with links to the original articles.
    8. Present the information in order of relevance/importance, not in the order of sources.
    
    Write the summary as if you're directly answering the user's question: "{query}"
    """
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ],
        "max_tokens": 1000,
        "temperature": 0.3
    }

    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending summarization request to Groq API (attempt {attempt+1}/{max_retries})")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                json=payload, 
                headers=headers,
                timeout=20
            )
            response.raise_for_status()
            response_json = response.json()

            choices = response_json.get("choices", [])
            if choices and "message" in choices[0]:
                summary = choices[0]["message"]["content"].strip()
                
                # Check if summary appears incomplete
                incomplete_indicators = [
                    "...", "to be continued", "I'll continue", "Next, ", "Additionally, "
                ]
                
                ends_with_incomplete = any(summary.rstrip().endswith(indicator) for indicator in incomplete_indicators)
                if ends_with_incomplete:
                    logger.warning("Summary appears incomplete, requesting continuation")
                    
                    # Request continuation
                    continuation_payload = {
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": context},
                            {"role": "assistant", "content": summary},
                            {"role": "user", "content": "Please continue your response and complete it."}
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.3
                    }
                    
                    continuation_response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions", 
                        json=continuation_payload, 
                        headers=headers,
                        timeout=20
                    )
                    continuation_response.raise_for_status()
                    
                    continuation_json = continuation_response.json()
                    continuation_choices = continuation_json.get("choices", [])
                    
                    if continuation_choices and "message" in continuation_choices[0]:
                        continuation_text = continuation_choices[0]["message"]["content"].strip()
                        # Combine original summary with continuation
                        summary = summary + " " + continuation_text
                
                # Ensure sources are properly formatted with actual links
                if "<h2>Sources</h2>" in summary or "<h3>Sources</h3>" in summary:
                    # Check if sources section has proper links
                    source_section_pattern = r"<h[23]>Sources</h[23]>.*?(?=<h[23]>|$)"
                    source_section_match = re.search(source_section_pattern, summary, re.DOTALL)
                    
                    if source_section_match:
                        source_section = source_section_match.group(0)
                        
                        # Check if there are proper links in the source section
                        if "<a href=" not in source_section:
                            # Replace the source section with proper links
                            new_source_section = "<h2>Sources</h2>\n<ul>\n"
                            for i, result in enumerate(results[:5], 1):
                                new_source_section += f'<li><a href="{result["link"]}" target="_blank">{result["title"]}</a></li>\n'
                            new_source_section += "</ul>\n"
                            
                            summary = re.sub(source_section_pattern, new_source_section, summary, flags=re.DOTALL)
                else:
                    # Add sources section if not present
                    summary += "\n\n<h2>Sources</h2>\n<ul>\n"
                    for i, result in enumerate(results[:5], 1):
                        summary += f'<li><a href="{result["link"]}" target="_blank">{result["title"]}</a></li>\n'
                    summary += "</ul>\n"
                    
                return summary
                
            logger.warning("Unexpected response format from Groq during summarization")
            
            # Fallback to a basic summary
            basic_summary = f"<h2>Information about {query}</h2>\n<p>Here's what I found:</p>\n<ul>\n"
            for result in results[:5]:
                basic_summary += f"<li>{result['snippet']}</li>\n"
            basic_summary += "</ul>\n\n<h2>Sources</h2>\n<ul>\n"
            for result in results[:5]:
                basic_summary += f'<li><a href="{result["link"]}" target="_blank">{result["title"]}</a></li>\n'
            basic_summary += "</ul>\n"
            
            return basic_summary
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during summarization: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("Max retries reached during summarization")
                
                # Fallback to a basic summary
                basic_summary = f"<h2>Information about {query}</h2>\n<p>Here's what I found:</p>\n<ul>\n"
                for result in results[:5]:
                    basic_summary += f"<li>{result['snippet']}</li>\n"
                basic_summary += "</ul>\n\n<h2>Sources</h2>\n<ul>\n"
                for result in results[:5]:
                    basic_summary += f'<li><a href="{result["link"]}" target="_blank">{result["title"]}</a></li>\n'
                basic_summary += "</ul>\n"
                
                return basic_summary
                
        except Exception as e:
            logger.error(f"Unexpected error during summarization: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("Max retries reached during summarization")
                
                # Fallback to a basic summary
                basic_summary = f"<h2>Information about {query}</h2>\n<p>Here's what I found:</p>\n<ul>\n"
                for result in results[:5]:
                    basic_summary += f"<li>{result['snippet']}</li>\n"
                basic_summary += "</ul>\n\n<h2>Sources</h2>\n<ul>\n"
                for result in results[:5]:
                    basic_summary += f'<li><a href="{result["link"]}" target="_blank">{result["title"]}</a></li>\n'
                basic_summary += "</ul>\n"
                
                return basic_summary

def enhance_with_web_search(user_query, is_current_event=False):
    """Enhance response with web search results"""
    logger.info(f"Performing web search for: {user_query}")
    
    # Determine if this is a sports-related query
    sports_keywords = ["score", "match", "game", "tournament", "championship", 
                       "playoffs", "final", "season", "league", "team", "player", 
                       "nba", "nfl", "mlb", "nhl", "soccer", "football", "baseball", 
                       "basketball", "hockey", "cricket", "tennis", "golf"]
                       
    is_sports_query = any(keyword in user_query.lower() for keyword in sports_keywords)
    
    # Perform the web search
    web_results = fetch_web_results(user_query, is_current_event)
    
    if not web_results:
        logger.warning("No web results found")
        return "<p>I couldn't find any reliable information about this topic from web searches.</p>"
    
    logger.info(f"Found {len(web_results)} web results, generating summary")
    
    # Generate a summary of the web results
    if is_sports_query:
        logger.info("Sports query detected, using specialized processing")
        sports_data = process_sports_results(user_query, web_results)
        if sports_data and isinstance(sports_data, list) and not isinstance(sports_data[0], dict):
            # Normal web results, use standard summarization
            return summarize_web_results(user_query, web_results, is_current_event)
        elif sports_data:
            # We have structured sports data
            logger.info("Using structured sports data for response")
            
            # Format sports data into HTML
            if "teams" in sports_data[0]:
                # It's match data
                match_data = sports_data[0]  # Use the most relevant match
                
                html_response = f"<h2>Match Information</h2>\n"
                
                if len(match_data["teams"]) >= 2:
                    html_response += f"<p><strong>Teams:</strong> {match_data['teams'][0]} vs {match_data['teams'][1]}</p>\n"
                
                if match_data.get("winner"):
                    html_response += f"<p><strong>Winner:</strong> {match_data['winner']}</p>\n"
                    
                if match_data.get("score"):
                    html_response += f"<p><strong>Score:</strong> {match_data['score']}</p>\n"
                
                html_response += "\n<h3>Sources</h3>\n<ul>\n"
                for result in web_results[:3]:
                    html_response += f'<li><a href="{result["link"]}" target="_blank">{result["title"]}</a></li>\n'
                html_response += "</ul>\n"
                
                return html_response
            else:
                # Fall back to standard summarization
                return summarize_web_results(user_query, web_results, is_current_event)
    else:
        # Standard summarization for non-sports queries
        return summarize_web_results(user_query, web_results, is_current_event)

def generate_enhanced_response(user_query, web_context):
    """Generate enhanced response using Groq with web search context"""
    headers = {
        "Authorization": f"Bearer {GROQCLOUD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = f"""
    You are ELAN, an AI Assistant specializing in Sports, Fitness, Mental Well-Being, Meditation, Yoga, and Games.
    ELAN was created by {CREATOR}.
    
    IMPORTANT GUIDELINES:
    1. Always provide COMPLETE answers. Never cut off mid-sentence.
    2. If you're listing items or explaining steps, always complete the full list or all steps.
    3. Use proper HTML formatting with headings (<h1> for main headings, <h2> for subheadings)
    4. Use <em> for emphasis and <strong> for important points
    5. Create well-structured responses with clear sections
    6. Use <ul> and <li> for lists
    7. ALWAYS check your response to ensure it's complete before sending
    8. When citing web information, include source attribution using inline references
    
    Only answer questions related to Sports, Fitness, Mental Well-Being, Meditation, Yoga, and Games.
    For any other topics, return 'OUT_OF_SCOPE'.
    
    You have been given web search results to help answer the user's question. 
    Incorporate this information into your response, ensuring you:
    - Synthesize the information rather than just repeating it
    - If the web results provide definitive information, present it with confidence
    - If there are conflicting information in the web results, acknowledge both perspectives
    - Always maintain proper citations to the sources in a "Sources" section at the end
    
    When answering who created you, always mention: "I was created by {CREATOR}."
    """
    
    combined_content = f"User question: {user_query}\n\nWeb search information:\n{web_context}"
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_content}
        ],
        "max_tokens": 1000,
        "temperature": 0.5
    }

    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending enhanced request to Groq API (attempt {attempt+1}/{max_retries})")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                json=payload, 
                headers=headers,
                timeout=20
            )
            response.raise_for_status()
            response_json = response.json()

            choices = response_json.get("choices", [])
            if choices and "message" in choices[0]:
                ai_response = choices[0]["message"]["content"].strip()
                
                # Check for special response codes
                if "OUT_OF_SCOPE" in ai_response:
                    logger.info("Groq indicated query is out of scope")
                    return "OUT_OF_SCOPE"
                
                # Check if response appears incomplete
                incomplete_indicators = [
                    "...", "to be continued", "I'll continue", "Next, ", "Additionally, "
                ]
                
                ends_with_incomplete = any(ai_response.rstrip().endswith(indicator) for indicator in incomplete_indicators)
                if ends_with_incomplete:
                    logger.warning("Enhanced response appears incomplete, requesting continuation")
                    
                    # Request continuation
                    continuation_payload = {
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": combined_content},
                            {"role": "assistant", "content": ai_response},
                            {"role": "user", "content": "Please continue your response and complete it."}
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.5
                    }
                    
                    continuation_response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions", 
                        json=continuation_payload, 
                        headers=headers,
                        timeout=20
                    )
                    continuation_response.raise_for_status()
                    
                    continuation_json = continuation_response.json()
                    continuation_choices = continuation_json.get("choices", [])
                    
                    if continuation_choices and "message" in continuation_choices[0]:
                        continuation_text = continuation_choices[0]["message"]["content"].strip()
                        # Combine original response with continuation
                        ai_response = ai_response + " " + continuation_text
                
                logger.info("Successfully received enhanced response from Groq")
                return ai_response
                
            logger.warning("Unexpected response format from Groq for enhanced response")
            return web_context  # Fall back to the web context itself
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error with enhanced Groq API call: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("Max retries reached for enhanced Groq API call")
                return web_context  # Fall back to the web context itself
        except Exception as e:
            logger.error(f"Unexpected error with enhanced Groq API call: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("Max retries reached for enhanced Groq API call")
                return web_context  # Fall back to the web context itself
    
    return web_context  # Fall back to the web context itself if all retries fail

def is_greeting(message):
    """Check if message is just a greeting"""
    greetings = ["hello", "hi", "hey", "howdy", "hola", "greetings", "good morning", 
                "good afternoon", "good evening", "what's up", "sup"]
    
    # Remove punctuation and convert to lowercase
    cleaned = re.sub(r'[^\w\s]', '', message.lower())
    
    # Check if the message is just a greeting
    words = cleaned.split()
    if len(words) <= 3 and any(greeting in cleaned for greeting in greetings):
        return True
        
    return False

def is_thank_you(message):
    """Check if message is a thank you or appreciation"""
    thank_you_phrases = ["thank you", "thanks", "thx", "thank u", "appreciate", "grateful", 
                        "many thanks", "thxs", "tnx", "much appreciated"]
    
    # Remove punctuation and convert to lowercase
    cleaned = re.sub(r'[^\w\s]', '', message.lower())
    
    # Check if the message is expressing thanks
    return any(phrase in cleaned for phrase in thank_you_phrases) and len(cleaned.split()) <= 5

def is_farewell(message):
    """Check if message is a farewell or goodbye"""
    farewell_phrases = ["bye", "goodbye", "see you", "farewell", "later", "take care", "good night", 
                       "until next time", "signing off", "talk to you later", "talk later", 
                       "catch you later", "have a good day", "have a nice day"]
    
    # Remove punctuation and convert to lowercase
    cleaned = re.sub(r'[^\w\s]', '', message.lower())
    
    # Check if the message is a farewell
    return any(phrase in cleaned for phrase in farewell_phrases) and len(cleaned.split()) <= 5

def is_acknowledgment(message):
    """Check if message is a short acknowledgment like 'ok', 'great', etc."""
    acknowledgment_phrases = ["ok", "okay", "got it", "i see", "alright", "sure", "understand",
                            "understood", "perfect", "good", "great", "nice", "awesome", "fantastic",
                            "excellent", "amazing", "wonderful", "cool", "sounds good", "makes sense"]
    
    # Remove punctuation and convert to lowercase
    cleaned = re.sub(r'[^\w\s]', '', message.lower())
    
    # Check if the message is a short acknowledgment
    words = cleaned.split()
    return (any(phrase in cleaned for phrase in acknowledgment_phrases) and len(words) <= 3) or len(words) == 1

def is_positive_feedback(message):
    """Check if message contains positive feedback"""
    positive_phrases = ["good job", "well done", "perfect", "that's right", "thats right", "right",
                       "correct", "spot on", "exactly", "precisely", "you're right", "youre right",
                       "great", "excellent", "amazing", "fantastic", "brilliant", "love it", "awesome",
                       "perfect", "wonderful", "impressive", "outstanding", "exceptional", "impressive",
                       "i like", "helpful", "useful"]
    
    # Remove punctuation and convert to lowercase
    cleaned = re.sub(r'[^\w\s]', '', message.lower())
    
    # Check if the message contains positive feedback
    return any(phrase in cleaned for phrase in positive_phrases) and len(cleaned.split()) <= 5

def is_current_event_query(message):
    """Determine if a query is about current events"""
    message_lower = message.lower()
    
    # Keywords that suggest the user is asking about current, recent, or upcoming events
    time_keywords = ["today", "yesterday", "last night", "this week", "recent", 
                     "latest", "upcoming", "now", "current", "live", "just", 
                     "tonight", "this morning", "this year", 
                     "2024", "2025", "schedule", "happening"]
                     
    # Event-related keywords
    event_keywords = ["score", "result", "game", "match", "play", "final", 
                     "winner", "championship", "tournament", "series", 
                     "playoffs", "news", "update", "lead", "beat", "won", "lost",
                     "versus", "vs", "against", "compete"]
    
    # Check if the message contains both event and time keywords
    has_event_keyword = any(keyword in message_lower for keyword in event_keywords)
    has_time_keyword = any(keyword in message_lower for keyword in time_keywords)
    
    # If it has both an event keyword and a time keyword, it's likely a current event query
    return has_event_keyword and has_time_keyword

@app.route("/chat", methods=["POST"])
def get_response():
    start_time = time.time()
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        logger.info(f"Received message: {user_message[:50]}{'...' if len(user_message) > 50 else ''}")
        
        # Check for greetings first
        if is_greeting(user_message):
            greeting_response = f"""<h1>Hello! I am ELAN</h1>
            <p>I'm your AI assistant specializing in Sports, Fitness, Mental Well-Being, Meditation, Yoga, and Games. I was created by {CREATOR}. How can I help you today?</p>"""
            
            return jsonify({
                "response": greeting_response,
                "type": "text"
            })
            
        # Check for thank you messages
        if is_thank_you(user_message):
            thank_you_responses = [
                f"""<h2>You're Welcome!</h2>
                <p>I'm glad I could help! If you have any more questions about sports, fitness, meditation, yoga, or games, feel free to ask anytime.</p>""",
                
                f"""<h2>My Pleasure!</h2>
                <p>Happy to assist! Remember, I'm here whenever you need information or guidance on sports, fitness, well-being, meditation, yoga, or games. Is there anything else you'd like to know?</p>""",
                
                f"""<h2>Glad I Could Help!</h2>
                <p>It's always a pleasure assisting you. If you have more questions later, just ask! Created by {CREATOR} to help with all your sports, fitness, and wellness questions.</p>"""
            ]
            
            # Select a random thank you response
            selected_response = thank_you_responses[hash(user_message) % len(thank_you_responses)]
            
            return jsonify({
                "response": selected_response,
                "type": "text"
            })
            
        # Check for farewell messages
        if is_farewell(user_message):
            farewell_responses = [
                f"""<h2>Until Next Time!</h2>
                <p>It was great chatting with you! Feel free to return whenever you have questions about sports, fitness, meditation, yoga, or games. Have a great day!</p>""",
                
                f"""<h2>Goodbye!</h2>
                <p>Thanks for chatting! I'll be here when you need more information on sports, fitness, well-being, meditation, yoga, or games. Take care!</p>""",
                
                f"""<h2>See You Soon!</h2>
                <p>It's been a pleasure assisting you today. Remember, I'm always here to help with your questions about sports, fitness, meditation, and more. Created by {CREATOR} to support your wellness journey!</p>"""
            ]
            
            # Select a random farewell response
            selected_response = farewell_responses[hash(user_message) % len(farewell_responses)]
            
            return jsonify({
                "response": selected_response,
                "type": "text"
            })
        
        # Check for short acknowledgments like "ok", "great"
        if is_acknowledgment(user_message):
            acknowledgment_responses = [
                f"""<h2>Great!</h2>
                <p>Is there anything else you'd like to know about sports, fitness, meditation, yoga, or games? I'm here to help!</p>""",
                
                f"""<h2>I'm Here to Help!</h2>
                <p>What else would you like to explore today? Feel free to ask about any sports, fitness, meditation, yoga, or gaming topics.</p>""",
                
                f"""<h2>Ready for More?</h2>
                <p>I can provide information on a wide range of topics related to sports, fitness, mental well-being, meditation, yoga, and games. What else can I help you with?</p>"""
            ]
            
            # Select a random acknowledgment response
            selected_response = acknowledgment_responses[hash(user_message) % len(acknowledgment_responses)]
            
            return jsonify({
                "response": selected_response,
                "type": "text"
            })
            
        # Check for positive feedback like "great", "excellent", etc.
        if is_positive_feedback(user_message):
            positive_responses = [
                f"""<h2>Thank You!</h2>
                <p>I appreciate your positive feedback! I'm always working to provide the best information possible. Is there anything else you'd like to know about sports, fitness, meditation, yoga, or games?</p>""",
                
                f"""<h2>I'm Glad That Helped!</h2>
                <p>It's great to hear that! Created by {CREATOR} to provide accurate and helpful information. What other topics would you like to explore today?</p>""",
                
                f"""<h2>Wonderful!</h2>
                <p>I'm delighted that the information was useful! I'm here to continue assisting with any other questions you might have about sports, fitness, well-being, meditation, yoga, or games.</p>"""
            ]
            
            # Select a random positive feedback response
            selected_response = positive_responses[hash(user_message) % len(positive_responses)]
            
            return jsonify({
                "response": selected_response,
                "type": "text"
            })
            
        # Check if this is a query about who created ELAN
        creator_keywords = ["who created", "who made", "who built", "who developed", 
                           "creator", "developer", "author", "built by", "made by", 
                           "developed by", "origin", "developer"]
        is_creator_query = "you" in user_message.lower() and any(keyword in user_message.lower() for keyword in creator_keywords)
        
        if is_creator_query:
            creator_response = f"""<h2>About My Creator</h2>
            <p>I was created by {CREATOR}. I'm an AI assistant designed to help with topics related to Sports, Fitness, Mental Well-Being, Meditation, Yoga, and Games.</p>"""
            
            return jsonify({
                "response": creator_response,
                "type": "text"
            })
            
        # Check if the query is about current events
        is_current_event = is_current_event_query(user_message)
        if is_current_event:
            logger.info("Current event query detected")
            
            # For current events, we'll skip the base Groq query and go straight to web search
            web_results = enhance_with_web_search(user_message, is_current_event=True)
            
            if web_results:
                # If it's HTML response already (like from summarize_web_results)
                if isinstance(web_results, str) and (web_results.startswith("<h") or "<p>" in web_results):
                    return jsonify({
                        "response": web_results,
                        "type": "text"
                    })
                    
                # Try to generate a summary response
                enhanced_response = generate_enhanced_response(user_message, web_results)
                
                if enhanced_response and enhanced_response != "OUT_OF_SCOPE":
                    process_time = time.time() - start_time
                    logger.info(f"Current event query processing completed in {process_time:.2f} seconds")
                    
                    return jsonify({
                        "response": enhanced_response,
                        "type": "text"
                    })
            
            # If web search failed, return a fallback response
            fallback_response = """<h2>I couldn't find up-to-date information</h2>
            <p>I'm sorry, but I couldn't find reliable current information about this topic. This could be due to:</p>
            <ul>
            <li>The event is very recent or still ongoing</li>
            <li>There may be limited information available online yet</li>
            <li>There might be an issue with the search services I use</li>
            </ul>
            <p>Please try again later or rephrase your question with more details.</p>"""
            
            return jsonify({
                "type": "no_answer", 
                "response": fallback_response
            })
            
        # For regular queries, first try the base Groq query
        groq_response = fetch_from_groq(user_message)
        
        if groq_response == "OUT_OF_SCOPE":
            out_of_scope_response = """<h2>I'm specialized in other topics</h2>
            <p>I'm sorry, but that question appears to be outside my areas of specialization. I'm designed to help with topics related to:</p>
            <ul>
            <li>Sports</li>
            <li>Fitness</li>
            <li>Mental Well-Being</li>
            <li>Meditation</li>
            <li>Yoga</li>
            <li>Games</li>
            </ul>
            <p>Please feel free to ask me anything about these topics!</p>"""
            
            return jsonify({
                "response": out_of_scope_response,
                "type": "text"
            })
        
        # If Groq couldn't provide a response, try web search
        if groq_response is None:
            logger.info("Groq couldn't provide a response, trying web search")
            
            web_results = enhance_with_web_search(user_message)
            
            if web_results:
                # If it's already HTML (like from summarize_web_results)
                if isinstance(web_results, str) and (web_results.startswith("<h") or "<p>" in web_results):
                    return jsonify({
                        "response": web_results,
                        "type": "text"
                    })
                
                # Generate enhanced response
                enhanced_response = generate_enhanced_response(user_message, web_results)
                
                if enhanced_response and enhanced_response != "OUT_OF_SCOPE":
                    process_time = time.time() - start_time
                    logger.info(f"Web search query processing completed in {process_time:.2f} seconds")
                    
                    return jsonify({
                        "response": enhanced_response,
                        "type": "text"
                    })
            
            # If web search also failed, return a fallback response
            fallback_response = """<h2>I don't have enough information</h2>
            <p>I'm sorry, but I don't have enough information to properly answer your question. This could be because:</p>
            <ul>
            <li>The topic is very specialized or niche</li>
            <li>The information might be too recent (after my knowledge cutoff)</li>
            <li>There might be an issue with the search services I use</li>
            </ul>
            <p>Please try rephrasing your question or asking about a different topic.</p>"""
            
            return jsonify({
                "type": "no_answer", 
                "response": fallback_response
            })
        
        # If we got a direct response from Groq, convert markdown to HTML and return it
        html_response = markdown_to_html(groq_response)
        
        process_time = time.time() - start_time
        logger.info(f"Query processing completed in {process_time:.2f} seconds")
        
        return jsonify({
            "response": html_response,
            "type": "text"
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.2"
    })

if __name__ == "__main__":
    logger.info("Starting ELAN server")
    app.run(debug=True, host="0.0.0.0", port=5000)