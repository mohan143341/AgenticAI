# import os
# import json
# import requests
# import datetime
# from typing import Optional, Dict, Any
# from pydantic import BaseModel
# from dotenv import load_dotenv
# load_dotenv()

# # Constants
# API_URL = "https://api.typefully.com/v2"
# SOCIAL_SET_ID = 306938

# API_KEY = os.getenv("TYPEFULLY_API_KEY")
# HEADERS = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json"
# }


# if not API_KEY:
#     raise ValueError("TYPEFULLY_API_KEY is not set in environment variables.")

# def json_to_typefully_content(thread_json: Dict[str, Any]) -> str:
#     """Convert JSON thread format to Typefully's format with 4 newlines between tweets."""
#     tweets = thread_json['tweets']
#     formatted_tweets = []
#     for tweet in tweets:
#         tweet_text = tweet['content']
#         if 'media_urls' in tweet and tweet['media_urls']:
#             tweet_text += f"\n{tweet['media_urls'][0]}"
#         formatted_tweets.append(tweet_text)
    
#     return '\n\n\n\n'.join(formatted_tweets)

# def json_to_linkedin_content(thread_json: Dict[str, Any]) -> str:
#     """Convert JSON thread format to Typefully's format."""
#     content = thread_json['content']
#     if 'url' in thread_json and thread_json['url']:
#         content += f"\n{thread_json['url']}"
#     return content

# def schedule_thread(
#     content: str,
#     schedule_date: str = "next-free-slot",
#     threadify: bool = False,
#     share: bool = False,
#     auto_retweet_enabled: bool = False,
#     auto_plug_enabled: bool = False
# ) -> Optional[Dict[str, Any]]:
#     """Schedule a thread on Typefully."""
#     payload = {
#         "content": content,
#         "schedule-date": schedule_date,
#         "threadify": threadify,
#         "share": share,
#         "auto_retweet_enabled": auto_retweet_enabled,
#         "auto_plug_enabled": auto_plug_enabled
#     }
    
#     payload = {key: value for key, value in payload.items() if value is not None}

#     try:
#         response = requests.post(API_URL, json=payload, headers=HEADERS)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error: {e}")
#         return None

# def schedule(
#     thread_model: BaseModel,
#     hours_from_now: int = 1,
#     threadify: bool = False,
#     share: bool = True,
#     post_type: str = "twitter"
# ) -> Optional[Dict[str, Any]]:
#     """
#     Schedule a thread from a Pydantic model.
    
#     Args:
#         thread_model: Pydantic model containing thread data
#         hours_from_now: Hours from now to schedule the thread (default: 1)
#         threadify: Whether to let Typefully split the content (default: False)
#         share: Whether to get a share URL in response (default: True)
    
#     Returns:
#         API response dictionary or None if failed
#     """
#     try:
#         # Convert Pydantic model to dict
#         thread_json = thread_model.pydantic.model_dump()
#         print("######## Thread JSON: ", thread_json)
#         # Convert to Typefully format
#         if post_type == "twitter":
#             thread_content = json_to_typefully_content(thread_json)
#         elif post_type == "linkedin":
#             thread_content = json_to_linkedin_content(thread_json)
        
#         # Calculate schedule time
#         schedule_date = (datetime.datetime.utcnow() + 
#                         datetime.timedelta(hours=hours_from_now)).isoformat() + "Z"
        
#         # Schedule the thread
#         response = schedule_thread(
#             content=thread_content,
#             schedule_date=schedule_date,
#             threadify=threadify,
#             share=share
#         )
        
#         if response:
#             print("Thread scheduled successfully!")
#             return response
#         else:
#             print("Failed to schedule the thread.")
#             return None
            
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return None



# # Test with your actual LinkedIn output
# from pydantic import BaseModel
# from typing import Optional

# class LinkedInPost(BaseModel):
#     content: str
#     media_url: Optional[str] = None

# fake_linkedin = LinkedInPost(
#     content="🚀 **5 Chunking Strategies For RAG** 🚀\n\nIn today's fast-paced digital world...",  # your content here
#     media_url="https://example.com/blog-post"
# )

# # Test the conversion
# thread_json = fake_linkedin.model_dump()
# content = json_to_linkedin_content(thread_json)
# print(content)

# # Test API call — try without Bearer prefix
# API_KEY = os.getenv("TYPEFULLY_API_KEY")
# print(f"API Key loaded: {API_KEY[:10]}..." if API_KEY else "API Key is MISSING!")

# HEADERS = {"X-API-KEY": API_KEY}  # No "Bearer"

# response = schedule_thread(
#     content=content,
#     schedule_date="next-free-slot",
#     share=True
# )
# print("Response:", response)


import os
import json
import requests
import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# Constants — UPDATED to v2
API_KEY = os.getenv("TYPEFULLY_API_KEY")
BASE_URL = "https://api.typefully.com/v2"
SOCIAL_SET_ID = 306938
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

if not API_KEY:
    raise ValueError("TYPEFULLY_API_KEY is not set in environment variables.")


def schedule_twitter_thread(tweets, publish_at="next-free-slot"):
    """Schedule a Twitter thread using v2 API."""
    posts = [{"text": t["content"]} for t in tweets]
    payload = {
        "platforms": {
            "x": {"enabled": True, "posts": posts}
        },
        "publish_at": publish_at
    }
    try:
        r = requests.post(
            f"{BASE_URL}/social-sets/{SOCIAL_SET_ID}/drafts",
            json=payload, headers=HEADERS
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def schedule_linkedin_post(content, publish_at="next-free-slot"):
    """Schedule a LinkedIn post using v2 API."""
    payload = {
        "platforms": {
            "linkedin": {"enabled": True, "posts": [{"text": content}]}
        },
        "publish_at": publish_at
    }
    try:
        r = requests.post(
            f"{BASE_URL}/social-sets/{SOCIAL_SET_ID}/drafts",
            json=payload, headers=HEADERS
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def schedule(
    thread_model: BaseModel,
    hours_from_now: int = 1,
    post_type: str = "twitter"
) -> Optional[Dict[str, Any]]:
    """Schedule from a CrewAI result."""
    try:
        thread_json = thread_model.pydantic.model_dump()
        print("######## Thread JSON: ", thread_json)

        publish_at = (
            datetime.datetime.utcnow() +
            datetime.timedelta(hours=hours_from_now)
        ).isoformat() + "Z"

        if post_type == "twitter":
            response = schedule_twitter_thread(thread_json["tweets"], publish_at)
        elif post_type == "linkedin":
            response = schedule_linkedin_post(thread_json["content"], publish_at)
        else:
            print(f"Unknown post type: {post_type}")
            return None

        if response:
            print("Thread scheduled successfully!")
            return response
        else:
            print("Failed to schedule the thread.")
            return None

    except Exception as e:
        print(f"Error: {str(e)}")
        return None