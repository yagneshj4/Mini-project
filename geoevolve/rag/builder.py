import os
import sys
import json
from typing import List
import chromadb
import arxiv
import requests
from urllib.parse import urljoin

class KnowledgeBuilder:
    """
    Fetches and indexes geospatial knowledge from arxiv and other sources.
    One-time run to populate the knowledge base.
    """
    
    def __init__(self, kb_dir: str = "data/knowledge"):
        """
        Initialize knowledge builder.
        
        Args:
            kb_dir: Directory to store knowledge files
        """
        self.kb_dir = kb_dir
        os.makedirs(kb_dir, exist_ok=True)
    
    def fetch_arxiv_papers(self, query: str, max_papers: int = 20) -> List[dict]:
        """
        Fetch papers from arXiv.
        
        Args:
            query: Search query
            max_papers: Max papers to fetch
            
        Returns:
            List of dicts with 'title', 'summary', 'authors', 'source'
        """
        papers = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in search.results():
                papers.append({
                    'title': result.title,
                    'summary': result.summary.replace('\n', ' '),
                    'authors': ', '.join([author.name for author in result.authors[:3]]),
                    'source': f'arXiv:{result.entry_id}',
                    'categories': result.categories
                })
            
            print(f"Fetched {len(papers)} papers from arXiv")
            return papers
            
        except Exception as e:
            print(f"Error fetching arXiv papers: {e}")
            return []
    
    def fetch_wikipedia_articles(self, titles: List[str]) -> List[dict]:
        """
        Fetch Wikipedia articles.
        
        Args:
            titles: List of article titles
            
        Returns:
            List of dicts with 'title', 'content', 'source'
        """
        articles = []
        
        try:
            import wikipedia
            
            for title in titles:
                try:
                    page = wikipedia.page(title, auto_suggest=True)
                    articles.append({
                        'title': page.title,
                        'content': page.content[:2000],  # Truncate to first 2000 chars
                        'source': f'Wikipedia:{title}',
                        'url': page.url
                    })
                    print(f"Fetched Wikipedia article: {page.title}")
                except Exception as e:
                    print(f"Error fetching {title}: {e}")
        
        except ImportError:
            print("Wikipedia module not available")
        
        return articles
    
    def save_documents(self, documents: List[dict], filename: str):
        """
        Save documents as JSON lines.
        
        Args:
            documents: List of document dicts
            filename: Output file name
        """
        filepath = os.path.join(self.kb_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')
        
        print(f"Saved {len(documents)} documents to {filepath}")
    
    def build_knowledge_base(self):
        """
        Fetch and save all knowledge base documents.
        """
        print("Building knowledge base...")
        
        # Fetch arXiv papers
        kriging_papers = self.fetch_arxiv_papers(
            "kriging spatial interpolation variogram",
            max_papers=15
        )
        
        geo_papers = self.fetch_arxiv_papers(
            "geostatistics spatial analysis machine learning",
            max_papers=10
        )
        
        # Save papers
        all_papers = kriging_papers + geo_papers
        self.save_documents(all_papers, 'arxiv_papers.jsonl')
        
        # Fetch Wikipedia articles
        wiki_titles = [
            "Kriging",
            "Variogram",
            "Spatial autocorrelation",
            "Interpolation",
            "Geostatistics"
        ]
        
        wiki_articles = self.fetch_wikipedia_articles(wiki_titles)
        self.save_documents(wiki_articles, 'wikipedia_articles.jsonl')
        
        print(f"\nKnowledge base built with:")
        print(f"  - {len(all_papers)} arXiv papers")
        print(f"  - {len(wiki_articles)} Wikipedia articles")


if __name__ == "__main__":
    builder = KnowledgeBuilder()
    builder.build_knowledge_base()
