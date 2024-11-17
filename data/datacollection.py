import requests
import praw
import json
import pandas as pd
from bs4 import BeautifulSoup
import concurrent.futures
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect
import logging
import os
from datetime import datetime
import time
from tqdm import tqdm
import random
from urllib.parse import urljoin
import re
from ratelimit import limits, sleep_and_retry
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from collections import defaultdict
import spacy
import arxiv
from scholarly import scholarly
import feedparser
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
import wikipediaapi

class EnhancedDataCollector:
    def __init__(self):
        self.languages = ['en', 'es', 'fr', 'ru', 'zh']
        self.setup_logging()
        self.setup_reddit()
        self.setup_nlp_tools()
        self.setup_rate_limits()
        
    def setup_nlp_tools(self):
        """Initialize NLP tools and models"""
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        
        # Load spaCy models for different languages
        self.nlp_models = {
            'en': spacy.load('en_core_web_sm'),
            'es': spacy.load('es_core_news_sm'),
            'fr': spacy.load('fr_core_news_sm')
            # Add more language models as needed
        }
        
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
    def setup_rate_limits(self):
        """Set up rate limiting for various APIs"""
        self.CALLS_PER_SECOND = {
            'reddit': 30,
            'wikipedia': 20,
            'scholar': 10,
            'arxiv': 15,
            'gutenberg': 5
        }
        
    @sleep_and_retry
    @limits(calls=30, period=60)
    def reddit_api_call(self, *args, **kwargs):
        """Rate-limited Reddit API call"""
        return self.reddit.subreddit(*args, **kwargs)
    
    @sleep_and_retry
    @limits(calls=20, period=60)
    def wikipedia_api_call(self, *args, **kwargs):
        """Rate-limited Wikipedia API call"""
        return requests.get(*args, **kwargs)
    
    def collect_literature_data(self, limit=1000):
        """Collect data from Project Gutenberg and other literature sources"""
        data = []
        
        # Project Gutenberg collection
        gutenberg_books = {
            'en': [1342, 11, 1661],  # Pride and Prejudice, Alice in Wonderland, Sherlock Holmes
            'fr': [17989, 13736],    # Les Misérables, Notre-Dame de Paris
            'es': [2000, 2009],      # Don Quixote, Novelas Ejemplares
        }
        
        for lang, books in gutenberg_books.items():
            for book_id in books:
                try:
                    text = strip_headers(load_etext(book_id)).strip()
                    sentences = sent_tokenize(text)
                    
                    # Sample sentences from the book
                    selected_sentences = random.sample(sentences, min(50, len(sentences)))
                    
                    for sentence in selected_sentences:
                        data.append({
                            'text': self.clean_text(sentence),
                            'tone': 'neutral',
                            'language': lang,
                            'source': 'gutenberg',
                            'book_id': book_id
                        })
                except Exception as e:
                    logging.error(f"Error collecting Gutenberg data for book {book_id}: {str(e)}")
                    
        return data
    
    def collect_arxiv_data(self, limit=1000):
        """Collect data from arXiv papers"""
        data = []
        
        # Define search queries for different languages
        queries = {
            'en': 'language OR linguistics',
            'fr': 'langue OR linguistique',
            'es': 'lengua OR linguistica'
        }
        
        for lang, query in queries.items():
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=limit//len(queries),
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                
                for paper in search.results():
                    abstract = self.clean_text(paper.summary)
                    sentences = sent_tokenize(abstract)
                    
                    for sentence in sentences:
                        data.append({
                            'text': sentence,
                            'tone': 'formal',
                            'language': lang,
                            'source': 'arxiv'
                        })
            except Exception as e:
                logging.error(f"Error collecting arXiv data for {lang}: {str(e)}")
                
        return data

    def analyze_strangeness_sophisticated(self, text, language='en'):
        """
        Enhanced strangeness detection using multiple factors
        Returns a score between 0 and 1, and detailed metrics
        """
        metrics = defaultdict(float)
        
        try:
            # Use appropriate spaCy model if available
            nlp = self.nlp_models.get(language, self.nlp_models['en'])
            doc = nlp(text)
            
            # 1. Lexical Complexity
            words = [token.text.lower() for token in doc if token.is_alpha]
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            metrics['lexical_complexity'] = min(avg_word_length / 12, 1.0)
            
            # 2. Syntactic Complexity
            if language == 'en':
                sentence_length = len(doc)
                clause_count = len([token for token in doc if token.dep_ == 'ROOT'])
                metrics['syntactic_complexity'] = min(sentence_length / (clause_count * 15) if clause_count else 0, 1.0)
            
            # 3. Semantic Unusualness (for English)
            if language == 'en':
                unusual_word_count = 0
                for word in words:
                    synsets = wordnet.synsets(word)
                    if len(synsets) < 2:  # Rare words have fewer synsets
                        unusual_word_count += 1
                metrics['semantic_unusualness'] = unusual_word_count / len(words) if words else 0
            
            # 4. Named Entity Density
            named_entities = len(list(doc.ents))
            metrics['named_entity_density'] = min(named_entities / len(words) if words else 0, 1.0)
            
            # 5. Sentiment Extremity
            if language == 'en':
                sentiment = self.sentiment_analyzer(text)[0]
                metrics['sentiment_extremity'] = abs(sentiment['score'] - 0.5) * 2
            
            # 6. Structural Patterns
            metrics['structural_oddity'] = self._analyze_structural_patterns(text)
            
            # Calculate final strangeness score
            weights = {
                'lexical_complexity': 0.2,
                'syntactic_complexity': 0.2,
                'semantic_unusualness': 0.25,
                'named_entity_density': 0.1,
                'sentiment_extremity': 0.1,
                'structural_oddity': 0.15
            }
            
            final_score = sum(metrics[key] * weights.get(key, 0) for key in metrics)
            
            # Classify strangeness level
            if final_score < 0.3:
                strangeness_level = 'low'
            elif final_score < 0.6:
                strangeness_level = 'medium'
            else:
                strangeness_level = 'high'
                
            return {
                'level': strangeness_level,
                'score': final_score,
                'metrics': dict(metrics)
            }
            
        except Exception as e:
            logging.error(f"Error in strangeness analysis: {str(e)}")
            return {'level': 'low', 'score': 0, 'metrics': {}}
    
    def _analyze_structural_patterns(self, text):
        """Analyze structural patterns in text"""
        patterns = {
            'repetition': r'(\b\w+\b)(?:\s+\w+){0,3}\s+\1',
            'unusual_punctuation': r'[!?]{2,}|\.{3,}|[;:]{2,}',
            'unusual_capitalization': r'[A-Z]{2,}',
            'nested_parentheses': r'\([^()]*\([^()]*\)[^()]*\)'
        }
        
        score = 0
        for pattern in patterns.values():
            matches = len(re.findall(pattern, text))
            score += min(matches * 0.1, 0.25)
        
        return min(score, 1.0)

    def collect_creative_writing_data(self, limit=1000):
        """Collect data from creative writing sources"""
        data = []
        
        # Reddit creative writing subreddits
        creative_subreddits = [
            'WritingPrompts',
            'shortstories',
            'creativewriting',
            'poetry'
        ]
        
        for subreddit in creative_subreddits:
            try:
                for post in self.reddit_api_call(subreddit).hot(limit=limit//len(creative_subreddits)):
                    text = self.clean_text(post.selftext)
                    if text and len(text.split()) > 5:
                        strangeness_analysis = self.analyze_strangeness_sophisticated(text)
                        
                        data.append({
                            'text': text,
                            'tone': 'informal',
                            'language': 'en',
                            'source': f'reddit_{subreddit}',
                            'strangeness': strangeness_analysis['level'],
                            'strangeness_metrics': strangeness_analysis['metrics']
                        })
            except Exception as e:
                logging.error(f"Error collecting creative writing data from {subreddit}: {str(e)}")
        
        return data

    def collect_and_save_data(self, output_file='enhanced_training_data.jsonl', samples_per_category=1000):
        """Enhanced main method to collect and save all data"""
        all_data = []
        
        # Collect data from various sources
        data_sources = {
            'informal': self.collect_informal_data,
            'neutral': self.collect_neutral_data,
            'formal': self.collect_formal_data,
            'literature': self.collect_literature_data,
            'arxiv': self.collect_arxiv_data,
            'creative': self.collect_creative_writing_data
        }
        
        for source_name, collector_func in tqdm(data_sources.items()):
            try:
                source_data = collector_func(limit=samples_per_category)
                all_data.extend(source_data)
                logging.info(f"Collected {len(source_data)} samples from {source_name}")
            except Exception as e:
                logging.error(f"Error collecting data from {source_name}: {str(e)}")
        
        # Analyze and save data
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(all_data):
                if 'strangeness' not in item:  # If not already analyzed
                    strangeness_analysis = self.analyze_strangeness_sophisticated(
                        item['text'],
                        item['language']
                    )
                    item['strangeness'] = strangeness_analysis['level']
                    item['strangeness_metrics'] = strangeness_analysis['metrics']
                
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logging.info(f"Saved {len(all_data)} samples to {output_file}")
        self.print_collection_statistics(all_data)

    def print_collection_statistics(self, data):
        """Enhanced statistics about collected data"""
        df = pd.DataFrame(data)
        
        print("\nEnhanced Collection Statistics:")
        print("-" * 50)
        
        print("\nDistribution by Source:")
        print(df['source'].value_counts())
        
        print("\nDistribution by Language:")
        print(df['language'].value_counts())
        
        print("\nDistribution by Tone:")
        print(df['tone'].value_counts())
        
        print("\nDistribution by Strangeness:")
        print(df['strangeness'].value_counts())
        
        print("\nCross-tabulation of Tone and Strangeness:")
        print(pd.crosstab(df['tone'], df['strangeness']))
        
        # Calculate average text length by category
        df['text_length'] = df['text'].str.len()
        print("\nAverage Text Length by Category:")
        print(df.groupby(['tone', 'strangeness'])['text_length'].mean().round(2))

def main():
    collector = EnhancedDataCollector()
    collector.collect_and_save_data(samples_per_category=1000)

if __name__ == "__main__":
    main()