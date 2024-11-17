"""
Tone

Informal
• OpenSubtitles: The OpenSubtitles dataset contains conversational language across many languages and is highly diverse in tone.
• Common Crawl: Filtered segments of Common Crawl from social media or forums can provide informal language samples across languages.
• YouTube Captions and Comments Datasets: These datasets often include informal and conversational language that may vary significantly in tone and register
• Webscrape YT comments/reddit

Todo
Webscrape Reddit, Get informal movie dialogue, YT Caption/Comment dataset



Neutral
• CNN/DailyMail Dataset (translated or adapted for other languages): Known for providing a neutral, journalistic tone.
• Global Voices: An international journalism dataset in multiple languages with a fairly neutral tone.
• Europarl: Speeches from the European Parliament, though slightly formal, tend to be neutral and are available in a number of languages.
• Wikipedia: Wikipedia articles are generally written in a formal, neutral tone and cover many languages. Extract articles by category to focus on academic or technical topics.

TODO
Webscrape Wikipedia, Europarl, DailyMail Dataset



Formal
• JRC-Acquis: This dataset contains legislative texts in multiple languages, including English, Spanish, French, and others, with a highly formal tone.
• Academic Papers
• OpenSubtitles (Filtered): OpenSubtitles includes scripts from various movies and shows that can be filtered by context to find more formal speech.

Todo
Compile a list of legislative texts from JRC-Acquis, webscrape Google Scholar, Webscrape OpenSubtitles/Speech archives for formal addresses/speaches


Strangeness

• Low Strangeness:
	• Literature Collections (Project Gutenberg): In classics, there are many slightly unusual phrases or metaphors, which may help with low-level strangeness while still being understandable.
	• Short Story Corpora: Collections like CHILDES or certain children’s book corpora often contain elements of unusual but sensible ideas.
• Moderate Strangeness:
	• Internet Scraped Text (Common Crawl filtered by topics): For moderately strange language that remains comprehensible, scraping from blog posts or speculative fiction sources can yield good results.
	• Surrealist Literature (translated): Finding open collections of translated surrealist works or poetry in different languages can offer manageable yet strange examples.
• High Strangeness:
	• Corpus of Experimental Literature: Specific datasets focused on experimental literature can be challenging to locate but may exist in academic settings or as part of creative commons projects.
	• Creative Writing Datasets: Writing prompt responses from platforms like Reddit's r/WritingPrompts, filtered by language, could introduce highly unusual content that may fulfill high strangeness requirements.
"""

"""
Reddit webscraping for informal language
"""
import praw
from langdetect import detect
import os

# Reddit API credentials
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    user_agent='YOUR_USER_AGENT'
)

# Subreddits by language
subreddits = {
    'english': ['AskReddit', 'casualconversation', 'writingprompts'],
    'french': ['france', 'french', 'traduction'],
    'spanish': ['espanol', 'spanish', 'argentina'],
    'russian': ['ru', 'russia', 'learnrussian'],
    'chinese': ['chinese', 'mandarin', 'sinosphere']
}

# Output files for each language
output_files = {
    'english': 'english_corpus.txt',
    'french': 'french_corpus.txt',
    'spanish': 'spanish_corpus.txt',
    'russian': 'russian_corpus.txt',
    'chinese': 'chinese_corpus.txt'
}

# Create output directory if it doesn't exist
output_dir = "reddit_corpora"
os.makedirs(output_dir, exist_ok=True)

def save_to_file(language, comments):
    """Save comments to the corresponding language file."""
    file_path = os.path.join(output_dir, output_files[language])
    with open(file_path, 'a', encoding='utf-8') as f:
        for comment in comments:
            f.write(comment + '\n')

def scrape_comments(subreddits, language, limit=500):
    """Scrape comments from specified subreddits."""
    comments = []
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        try:
            for submission in subreddit.hot(limit=limit):
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    if comment.body:
                        detected_lang = detect(comment.body)
                        if detected_lang == language[:2]:  # Match language code
                            comments.append(comment.body.strip())
        except Exception as e:
            print(f"Error scraping subreddit {subreddit_name}: {e}")
    return comments

# Scrape and save comments for each language
language_codes = {'english': 'en', 'french': 'fr', 'spanish': 'es', 'russian': 'ru', 'chinese': 'zh'}

for lang, subs in subreddits.items():
    print(f"Scraping comments for {lang}...")
    lang_comments = scrape_comments(subs, language_codes[lang])
    save_to_file(lang, lang_comments)
    print(f"Saved {len(lang_comments)} comments to {output_files[lang]}.")

print("Scraping completed.")


"""
Youtube comments webscraping for informal language
Need Youtube API Key
"""

from googleapiclient.discovery import build
from langdetect import detect
import os

# YouTube API Key
API_KEY = 'YOUR_YOUTUBE_API_KEY'

# YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# YouTube channels or video IDs by language
youtube_sources = {
    'english': ['UCq-Fj5jknLsUf-MWSy4_brA',  # Example: Kurzgesagt
                'UCBJycsmduvYEL83R_U4JriQ'],  # Example: Veritasium
    'french': ['UCIpR2Oheev0tndRkjXmwfCw',  # Example: Cyprien
               'UCpko_-a4wgz2u_DgDgd9fqA'],  # Example: Nota Bene
    'spanish': ['UCYOjDNf2QS1nR7eRDPFQbDw',  # Example: Luisito Comunica
                'UCAm8T03EOFBsNdR0thrFHdQ'],  # Example: El Rubius
    'russian': ['UCvtT19MZW8dq5Wwfu6B0oxw',  # Example: Wylsacom
                'UCq7oK1cQG-_N7eNtjzH4_QA'],  # Example: This is Хорошо
    'chinese': ['UC4eYXhJI4-7wSWc8UNRwD4A',  # Example: CCTV中国中央电视台
                'UC7BOHMhzMbBkae2qccxZ62w']   # Example: 李子柒
}

# Output files for each language
output_files = {
    'english': 'youtube_english_corpus.txt',
    'french': 'youtube_french_corpus.txt',
    'spanish': 'youtube_spanish_corpus.txt',
    'russian': 'youtube_russian_corpus.txt',
    'chinese': 'youtube_chinese_corpus.txt'
}

# Create output directory
output_dir = "youtube_corpora"
os.makedirs(output_dir, exist_ok=True)

def save_to_file(language, comments):
    """Save comments to the corresponding language file."""
    file_path = os.path.join(output_dir, output_files[language])
    with open(file_path, 'a', encoding='utf-8') as f:
        for comment in comments:
            f.write(comment + '\n')

def get_video_ids(channel_id):
    """Fetch video IDs from a YouTube channel."""
    video_ids = []
    request = youtube.search().list(
        part='id',
        channelId=channel_id,
        maxResults=50,
        type='video'
    )
    response = request.execute()

    for item in response['items']:
        video_ids.append(item['id']['videoId'])
    return video_ids

def get_comments(video_id, language_code):
    """Fetch comments from a video and filter by language."""
    comments = []
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()

    for item in response.get('items', []):
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        try:
            if detect(comment) == language_code:
                comments.append(comment.strip())
        except:
            continue
    return comments

def scrape_comments(sources, language, language_code):
    """Scrape comments for a given language from YouTube."""
    all_comments = []
    for source in sources:
        try:
            video_ids = get_video_ids(source)
            for video_id in video_ids:
                comments = get_comments(video_id, language_code)
                all_comments.extend(comments)
        except Exception as e:
            print(f"Error processing source {source}: {e}")
    return all_comments

# Scrape and save comments for each language
language_codes = {'english': 'en', 'french': 'fr', 'spanish': 'es', 'russian': 'ru', 'chinese': 'zh'}

for lang, sources in youtube_sources.items():
    print(f"Scraping comments for {lang}...")
    lang_comments = scrape_comments(sources, lang, language_codes[lang])
    save_to_file(lang, lang_comments)
    print(f"Saved {len(lang_comments)} comments to {output_files[lang]}.")

print("Scraping completed.")


"""
Wikipedia for middle register 
"""
import requests
import os
from langdetect import detect

# Base Wikipedia API URLs for each language
wikipedia_apis = {
    'english': 'https://en.wikipedia.org/w/api.php',
    'french': 'https://fr.wikipedia.org/w/api.php',
    'spanish': 'https://es.wikipedia.org/w/api.php',
    'russian': 'https://ru.wikipedia.org/w/api.php',
    'chinese': 'https://zh.wikipedia.org/w/api.php'
}

# Output files for each language
output_files = {
    'english': 'wikipedia_english_corpus.txt',
    'french': 'wikipedia_french_corpus.txt',
    'spanish': 'wikipedia_spanish_corpus.txt',
    'russian': 'wikipedia_russian_corpus.txt',
    'chinese': 'wikipedia_chinese_corpus.txt'
}

# Create output directory
output_dir = "wikipedia_corpora"
os.makedirs(output_dir, exist_ok=True)

def save_to_file(language, articles):
    """Save articles to the corresponding language file."""
    file_path = os.path.join(output_dir, output_files[language])
    with open(file_path, 'a', encoding='utf-8') as f:
        for article in articles:
            f.write(article + '\n')

def fetch_articles(api_url, language_code, num_articles=50):
    """Fetch articles from Wikipedia using the MediaWiki API."""
    articles = []
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,  # Only main namespace (articles)
            'rnlimit': num_articles
        }
        response = requests.get(api_url, params=params).json()

        for page in response['query']['random']:
            page_id = page['id']
            page_title = page['title']

            # Get page content
            content_params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts',
                'explaintext': True,
                'pageids': page_id
            }
            content_response = requests.get(api_url, params=content_params).json()
            page_content = content_response['query']['pages'][str(page_id)].get('extract', '')

            # Language detection for accuracy
            if page_content and detect(page_content) == language_code:
                articles.append(page_content.strip())
    except Exception as e:
        print(f"Error fetching articles from {api_url}: {e}")
    return articles

# Scrape and save articles for each language
language_codes = {'english': 'en', 'french': 'fr', 'spanish': 'es', 'russian': 'ru', 'chinese': 'zh'}

for lang, api_url in wikipedia_apis.items():
    print(f"Scraping articles for {lang}...")
    lang_articles = fetch_articles(api_url, language_codes[lang])
    save_to_file(lang, lang_articles)
    print(f"Saved {len(lang_articles)} articles to {output_files[lang]}.")

print("Scraping completed.")


"""
News Sources
"""

import requests
from bs4 import BeautifulSoup
from langdetect import detect
import os

# News sources by language
news_sources = {
    'english': [
        'https://www.bbc.com/news',  # BBC News
        'https://edition.cnn.com/world'  # CNN
    ],
    'french': [
        'https://www.lemonde.fr/',  # Le Monde
        'https://www.lefigaro.fr/'  # Le Figaro
    ],
    'spanish': [
        'https://elpais.com/',  # El País
        'https://www.lanacion.com.ar/'  # La Nación
    ],
    'russian': [
        'https://ria.ru/',  # RIA Novosti
        'https://tass.ru/'  # TASS
    ],
    'chinese': [
        'https://www.chinadaily.com.cn/',  # China Daily
        'https://news.sina.com.cn/'  # Sina News
    ]
}

# Output files for each language
output_files = {
    'english': 'news_english_corpus.txt',
    'french': 'news_french_corpus.txt',
    'spanish': 'news_spanish_corpus.txt',
    'russian': 'news_russian_corpus.txt',
    'chinese': 'news_chinese_corpus.txt'
}

# Create output directory
output_dir = "news_corpora"
os.makedirs(output_dir, exist_ok=True)

def save_to_file(language, articles):
    """Save articles to the corresponding language file."""
    file_path = os.path.join(output_dir, output_files[language])
    with open(file_path, 'a', encoding='utf-8') as f:
        for article in articles:
            f.write(article + '\n\n')

def scrape_news(source_url, language_code):
    """Scrape articles from a news source."""
    articles = []
    try:
        response = requests.get(source_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all paragraphs (generic for most news sites)
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])

        # Filter content by language
        if content and detect(content) == language_code:
            articles.append(content.strip())
    except Exception as e:
        print(f"Error scraping {source_url}: {e}")
    return articles

# Scrape and save articles for each language
language_codes = {'english': 'en', 'french': 'fr', 'spanish': 'es', 'russian': 'ru', 'chinese': 'zh'}

for lang, sources in news_sources.items():
    print(f"Scraping news articles for {lang}...")
    all_articles = []
    for source in sources:
        articles = scrape_news(source, language_codes[lang])
        all_articles.extend(articles)
    save_to_file(lang, all_articles)
    print(f"Saved {len(all_articles)} articles to {output_files[lang]}.")

print("Scraping completed.")

"""
Fetching from academic sources
Focus on CompLing
"""
import requests
import os
from langdetect import detect

# Base arXiv API URL
arxiv_api_url = "http://export.arxiv.org/api/query"

# Search queries by language
search_queries = {
    'english': 'cat:cs.CL+AND+language:en',  # Computational Linguistics, English
    'french': 'cat:cs.CL+AND+language:fr',   # Computational Linguistics, French
    'spanish': 'cat:cs.CL+AND+language:es',  # Computational Linguistics, Spanish
    'russian': 'cat:cs.CL+AND+language:ru',  # Computational Linguistics, Russian
    'chinese': 'cat:cs.CL+AND+language:zh'   # Computational Linguistics, Chinese
}

# Output files for each language
output_files = {
    'english': 'academic_english_corpus.txt',
    'french': 'academic_french_corpus.txt',
    'spanish': 'academic_spanish_corpus.txt',
    'russian': 'academic_russian_corpus.txt',
    'chinese': 'academic_chinese_corpus.txt'
}

# Create output directory
output_dir = "academic_corpora"
os.makedirs(output_dir, exist_ok=True)

def save_to_file(language, abstracts):
    """Save abstracts to the corresponding language file."""
    file_path = os.path.join(output_dir, output_files[language])
    with open(file_path, 'a', encoding='utf-8') as f:
        for abstract in abstracts:
            f.write(abstract + '\n\n')

def fetch_abstracts(query, max_results=50):
    """Fetch abstracts from arXiv using a query."""
    abstracts = []
    try:
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        response = requests.get(arxiv_api_url, params=params)
        response.raise_for_status()
        content = response.text

        # Parse abstracts from XML response
        entries = content.split('<entry>')
        for entry in entries[1:]:  # Skip the first non-entry section
            if '<summary>' in entry:
                abstract = entry.split('<summary>')[1].split('</summary>')[0].strip()
                abstracts.append(abstract)
    except Exception as e:
        print(f"Error fetching abstracts: {e}")
    return abstracts

# Scrape and save abstracts for each language
for lang, query in search_queries.items():
    print(f"Fetching academic abstracts for {lang}...")
    lang_abstracts = fetch_abstracts(query)
    save_to_file(lang, lang_abstracts)
    print(f"Saved {len(lang_abstracts)} abstracts to {output_files[lang]}.")

print("Fetching completed.")

"""
Fetching from Legal sources
Check for API Keys
"""

import requests
from bs4 import BeautifulSoup
from langdetect import detect
import os

# Base URLs or API endpoints for legal data
legal_sources = {
    'english': [
        'https://www.courtlistener.com/api/rest/v3/opinions/'  # CourtListener API
    ],
    'french': [
        'https://www.canlii.org/fr/'  # CanLII
    ],
    'spanish': [
        'https://www.boe.es/'  # BOE (Boletín Oficial del Estado)
    ],
    'russian': [
        'https://www.consultant.ru/'  # КонсультантПлюс
    ],
    'chinese': [
        'http://wenshu.court.gov.cn/'  # China Judgments Online
    ]
}

# Output files for each language
output_files = {
    'english': 'legal_english_corpus.txt',
    'french': 'legal_french_corpus.txt',
    'spanish': 'legal_spanish_corpus.txt',
    'russian': 'legal_russian_corpus.txt',
    'chinese': 'legal_chinese_corpus.txt'
}

# Create output directory
output_dir = "legal_corpora"
os.makedirs(output_dir, exist_ok=True)

def save_to_file(language, documents):
    """Save legal documents to the corresponding language file."""
    file_path = os.path.join(output_dir, output_files[language])
    with open(file_path, 'a', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc + '\n\n')

def fetch_legal_documents(source_url, language_code, headers=None):
    """Fetch legal documents from a given source."""
    documents = []
    try:
        response = requests.get(source_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract paragraphs (generic for most legal sites)
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])

        # Filter content by language
        if content and detect(content) == language_code:
            documents.append(content.strip())
    except Exception as e:
        print(f"Error fetching legal documents from {source_url}: {e}")
    return documents

# Scrape and save legal documents for each language
language_codes = {'english': 'en', 'french': 'fr', 'spanish': 'es', 'russian': 'ru', 'chinese': 'zh'}

for lang, sources in legal_sources.items():
    print(f"Fetching legal reports for {lang}...")
    all_documents = []
    for source in sources:
        headers = {}
        if "courtlistener" in source:  # Example for APIs requiring authorization
            headers = {'Authorization': 'Token YOUR_API_KEY'}  # Replace with actual API key
        documents = fetch_legal_documents(source, language_codes[lang], headers=headers)
        all_documents.extend(documents)
    save_to_file(lang, all_documents)
    print(f"Saved {len(all_documents)} documents to {output_files[lang]}.")

print("Fetching completed.")

"""
V1 Scraping subreddits for strangeness
"""
import praw
import os
from langdetect import detect

# Reddit API credentials
REDDIT_CLIENT_ID = 'your_client_id'
REDDIT_CLIENT_SECRET = 'your_client_secret'
REDDIT_USER_AGENT = 'your_user_agent'

# Subreddits by language
subreddits = {
    'english': ['SurrealMemes', 'Dreams', 'Glitch_in_the_Matrix'],
    'french': ['Reves', 'Inexplique'],
    'spanish': ['Suenos', 'Inexplicable'],
    'russian': ['Сны', 'Странности'],
    'chinese': ['做梦', '奇妙']
}

# Output files for each language
output_files = {
    'english': 'high_strangeness_english_corpus.txt',
    'french': 'high_strangeness_french_corpus.txt',
    'spanish': 'high_strangeness_spanish_corpus.txt',
    'russian': 'high_strangeness_russian_corpus.txt',
    'chinese': 'high_strangeness_chinese_corpus.txt'
}

# Create output directory
output_dir = "high_strangeness_corpora"
os.makedirs(output_dir, exist_ok=True)

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def save_to_file(language, posts):
    """Save posts to the corresponding language file."""
    file_path = os.path.join(output_dir, output_files[language])
    with open(file_path, 'a', encoding='utf-8') as f:
        for post in posts:
            f.write(post + '\n\n')

def fetch_posts(subreddit_name, language_code, limit=50):
    """Fetch posts from a subreddit."""
    posts = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.hot(limit=limit):
            content = submission.title + "\n\n" + submission.selftext
            if content and detect(content) == language_code:
                posts.append(content.strip())
    except Exception as e:
        print(f"Error fetching posts from {subreddit_name}: {e}")
    return posts

# Scrape and save posts for each language
language_codes = {'english': 'en', 'french': 'fr', 'spanish': 'es', 'russian': 'ru', 'chinese': 'zh'}

for lang, subs in subreddits.items():
    print(f"Fetching posts for {lang}...")
    all_posts = []
    for subreddit_name in subs:
        posts = fetch_posts(subreddit_name, language_codes[lang])
        all_posts.extend(posts)
    save_to_file(lang, all_posts)
    print(f"Saved {len(all_posts)} posts to {output_files[lang]}.")

print("Fetching completed.")


