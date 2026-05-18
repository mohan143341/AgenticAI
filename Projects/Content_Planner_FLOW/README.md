
# 📝 Content Writing Agentic Workflow

This project leverages CrewAI Flows to scrape a blog post, generate platform-specific social media content, and publish it automatically via Typefully.

---

## How It Works

```
Blog URL → 

Scrape (FireCrawl) → 

Route (Twitter/LinkedIn)→ 

AI Crew (Generate Post) → 

Save + Publish (Typefully v2)
```

1. **Scrape** — FireCrawl extracts the blog post content as markdown
2. **Route** — Flow router selects the target platform (Twitter thread or LinkedIn post)
3. **Generate** — AI agents create optimized content for the chosen platform
4. **Save** — Generated content is saved as JSON locally
5. **Publish** — Post is scheduled via Typefully API v2

---

## Installation and Setup

### Prerequisites

- Python 3.11 or later
- A Typefully account (paid plan required for API access)
- A FireCrawl account

### Install Dependencies

```bash
pip install crewai crewai-tools firecrawl-py python-dotenv pydantic requests
```

### Get API Keys

- [FireCrawl](https://docs.firecrawl.dev/introduction) — for web scraping
- [Typefully](https://support.typefully.com/en/articles/8718287-typefully-api) — for scheduling posts

### Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_OPENAI_API_KEY
FIRECRAWL_API_KEY=your_firecrawl_api_key
TYPEFULLY_API_KEY=your_typefully_api_key
```

## Usage

### Run the Full Flow

```python
from flow import CreateContentPlanningFlow

flow = CreateContentPlanningFlow()
result = flow.kickoff()
```

### Configure the Flow

Edit the state to change the blog URL or target platform:

```python
post_type = "linkedin"    # "twitter" or "linkedin"
flow = CreateContentPlanningFlow()
flow.state.post_type = post_type
flow.state  

```

## Demo

### LinkedIn Post Published via Workflow

Here's a LinkedIn post generated and scheduled automatically by this workflow:

🔗 [View LinkedIn Post](https://www.linkedin.com/feed/update/urn:li:activity:7461413409851625472/)