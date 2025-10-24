#!/usr/bin/env python3
"""
IHS Deep Donor Research + Partner Profile Generator (Enhanced v5.0 - HYBRID)
------------------------------------------------------------------------------
Produces comprehensive donor profiles with AI-powered connection discovery to inner circle.
Uses HYBRID approach: Google CSE finds evidence + OpenAI o1 analyzes connections.

SETUP REQUIRED:
Create a file named 'spider.env' in the same directory with your API keys:

    OPENAI_API_KEY=sk-your-openai-key-here
    OPENAI_MODEL=gpt-4o
    OPENAI_RESEARCH_MODEL=o1
    GOOGLE_API_KEY=your-google-key-here
    GOOGLE_CSE_ID=your-cse-id-here
    FEC_API_KEY=DEMO_KEY
    CACHE_DIR=./cache
    CACHE_TTL=86400
    INNER_CIRCLE_CSV=Inner_Circle.csv

HYBRID DEEP RESEARCH:
1. Google CSE searches for actual evidence (articles, publications, events)
2. OpenAI o1 analyzes search results to extract and structure connections
3. Best results require BOTH API keys, but works with either

DO NOT hardcode API keys in this script!
"""

import os, re, sys, json, time, math, difflib, csv, zipfile, tempfile, hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from functools import wraps
import requests
import pandas as pd
from dotenv import load_dotenv
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import openai

# -----------------------------
# Utilities
# -----------------------------

def log(msg: str):
    """Safe logging that handles encoding issues"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version
        safe_msg = msg.encode('ascii', 'replace').decode('ascii')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {safe_msg}", flush=True)

def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

# -----------------------------
# Caching Layer
# -----------------------------

class ResearchCache:
    def __init__(self, cache_dir: str = "./cache", ttl: int = 86400):
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, prefix: str, params: Dict) -> str:
        param_str = json.dumps(params, sort_keys=True)
        hash_key = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"{prefix}_{hash_key}"
    
    def get_or_fetch(self, key: str, fetcher, ttl: Optional[int] = None):
        ttl = ttl or self.ttl
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if time.time() - mtime < ttl:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except:
                    pass
        
        data = fetcher()
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except:
            pass
        
        return data

# -----------------------------
# Retry Logic
# -----------------------------

def retry_with_backoff(max_retries: int = 3, initial_wait: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        log(f"  ! Final retry failed for {func.__name__}: {e}")
                        raise
                    wait = initial_wait * (2 ** attempt)
                    log(f"  Retry {attempt + 1}/{max_retries} after {wait}s...")
                    time.sleep(wait)
            return None
        return wrapper
    return decorator

# -----------------------------
# API Clients
# -----------------------------

class GoogleCSE:
    def __init__(self, api_key: str, cse_id: str, cache: Optional[ResearchCache] = None):
        self.api_key = api_key
        self.cse_id = cse_id
        self.cache = cache or ResearchCache()
        self.endpoint = "https://www.googleapis.com/customsearch/v1"

    @retry_with_backoff(max_retries=3)
    def query(self, q: str, num=10):
        if not self.api_key or not self.cse_id:
            return []
        
        cache_key = self.cache._get_cache_key("google", {"q": q, "num": num})
        
        def fetcher():
            r = requests.get(self.endpoint, params={"key": self.api_key, "cx": self.cse_id, "q": q, "num": num}, timeout=15)
            r.raise_for_status()
            items = r.json().get("items", [])
            out = []
            for it in items:
                out.append({
                    "title": it.get("title",""),
                    "link": it.get("link",""),
                    "snippet": it.get("snippet",""),
                    "displayLink": it.get("displayLink",""),
                    "source": "google_cse",
                    "q": q
                })
            return out
        
        return self.cache.get_or_fetch(cache_key, fetcher)

class FECClient:
    def __init__(self, api_key: str, cache: Optional[ResearchCache] = None):
        self.api_key = api_key or "DEMO_KEY"
        self.cache = cache or ResearchCache()
        self.endpoint = "https://api.open.fec.gov/v1/schedules/schedule_a/"

    @retry_with_backoff(max_retries=3)
    def contributions(self, name: str, state: Optional[str] = None, per_page=50):
        cache_key = self.cache._get_cache_key("fec", {"name": name, "state": state})
        
        def fetcher():
            params = {"api_key": self.api_key, "contributor_name": name, "per_page": per_page}
            if state:
                params["contributor_state"] = state
            
            r = requests.get(self.endpoint, params=params, timeout=20)
            r.raise_for_status()
            rows = r.json().get("results", [])
            out = []
            for rec in rows:
                out.append({
                    "source": "fec",
                    "committee": rec.get("committee", {}).get("name"),
                    "amount": rec.get("contribution_receipt_amount"),
                    "date": rec.get("contribution_receipt_date"),
                    "employer": rec.get("contributor_employer"),
                    "occupation": rec.get("contributor_occupation"),
                    "recipient": rec.get("committee", {}).get("name"),
                    "party": self._identify_party(rec.get("committee", {}).get("name", "")),
                    "snippet": f"${rec.get('contribution_receipt_amount')} to {rec.get('committee',{}).get('name')}",
                    "link": "https://www.fec.gov/data/"
                })
            return out
        
        return self.cache.get_or_fetch(cache_key, fetcher)
    
    def _identify_party(self, recipient: str) -> str:
        recipient_lower = recipient.lower()
        if any(term in recipient_lower for term in ['democrat', 'dnc', 'dccc', 'dscc', 'actblue']):
            return 'Democrat'
        elif any(term in recipient_lower for term in ['republican', 'rnc', 'nrcc', 'nrsc', 'winred']):
            return 'Republican'
        else:
            return 'Other'
    
    def analyze_political_giving(self, contributions: List[Dict]) -> Dict:
        analysis = {
            'total_given': sum(c.get('amount', 0) for c in contributions),
            'num_contributions': len(contributions),
            'party_breakdown': defaultdict(float),
            'top_recipients': [],
            'giving_years': set(),
            'pattern': 'Unknown',
            'avg_contribution': 0
        }
        
        if not contributions:
            return analysis
        
        for contrib in contributions:
            party = contrib.get('party', 'Other')
            amount = contrib.get('amount', 0)
            analysis['party_breakdown'][party] += amount
            
            date_str = contrib.get('date', '')
            if date_str:
                try:
                    year = date_str[:4]
                    analysis['giving_years'].add(year)
                except:
                    pass
        
        analysis['avg_contribution'] = analysis['total_given'] / len(contributions) if contributions else 0
        
        party_totals = dict(analysis['party_breakdown'])
        if party_totals:
            total = sum(party_totals.values())
            dem_pct = party_totals.get('Democrat', 0) / total if total > 0 else 0
            rep_pct = party_totals.get('Republican', 0) / total if total > 0 else 0
            
            if dem_pct > 0.9:
                analysis['pattern'] = 'Consistent Democrat donor'
            elif rep_pct > 0.9:
                analysis['pattern'] = 'Consistent Republican donor'
            elif dem_pct > 0 and rep_pct > 0:
                analysis['pattern'] = 'Bipartisan donor'
            else:
                analysis['pattern'] = 'Independent/Third-party donor'
        
        recipient_totals = defaultdict(float)
        for contrib in contributions:
            recipient_totals[contrib.get('recipient', 'Unknown')] += contrib.get('amount', 0)
        
        analysis['top_recipients'] = sorted(
            recipient_totals.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return analysis

# -----------------------------
# ENHANCED INNER CIRCLE CONNECTION FINDER
# -----------------------------

class DeepConnectionFinder:
    """
    HYBRID connection finder that discovers donor-to-inner-circle relationships:
    
    STEP 1: Google CSE searches gather actual evidence (articles, publications, events)
    STEP 2: OpenAI o1 analyzes search results to extract structured connections
    
    This approach ensures we find REAL connections (via Google) and intelligently
    extract them (via o1), avoiding the hallucination problem of pure AI reasoning.
    """
    
    def __init__(self, csv_path: str, google_cse: GoogleCSE, cache: ResearchCache, ai_client=None, model: str = "o1"):
        self.path = csv_path
        self.google = google_cse
        self.cache = cache
        self.ai_client = ai_client
        self.model = model
        self.df = None
        self.load()
    
    def load(self):
        if not os.path.exists(self.path):
            log(f"  ! Inner circle CSV not found: {self.path}")
            return
        
        for enc in ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]:
            try:
                self.df = pd.read_csv(self.path, encoding=enc, on_bad_lines="skip")
                log(f"  Loaded Inner Circle CSV with {enc} encoding ({len(self.df)} members)")
                break
            except Exception:
                continue
        
        if self.df is None:
            log(f"  ! Could not read inner circle CSV")
            return
        
        # Don't pre-construct Full_Name - let _construct_member_name() handle it properly
        # (Old code only used First + Last, missing Middle names/initials)
    
    def _construct_member_name(self, member: Dict) -> str:
        """Construct full member name including middle name/initial"""
        # Check for pre-constructed Full_Name
        full_name = str(member.get("Full_Name", "")).strip()
        if full_name:
            return full_name
        
        # Build from First, Middle, Last
        first = str(member.get("First", "")).strip() if pd.notna(member.get("First")) else ""
        middle = str(member.get("Middle", "")).strip() if pd.notna(member.get("Middle")) else ""
        last = str(member.get("Last", "")).strip() if pd.notna(member.get("Last")) else ""
        
        # Construct name with available parts
        parts = []
        if first:
            parts.append(first)
        if middle:
            parts.append(middle)
        if last:
            parts.append(last)
        
        return " ".join(parts) if parts else ""
    
    def extract_donor_context(self, findings: List[Dict]) -> Dict:
        """Extract contextual information about the donor to help validate search results"""
        context = {
            'companies': set(),
            'positions': set(),
            'locations': set(),
            'keywords': set()
        }
        
        for finding in findings[:50]:  # Check top findings
            snippet = finding.get('snippet', '').lower()
            title = finding.get('title', '').lower()
            text = snippet + " " + title
            
            # Extract company names (CEO of X, founder of Y, etc.)
            company_patterns = [
                r'ceo of ([a-z\s]+)',
                r'founder of ([a-z\s]+)',
                r'president of ([a-z\s]+)',
                r'at ([a-z\s]+) ',
            ]
            
            for pattern in company_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    company = match.strip()
                    if len(company) > 2 and len(company) < 50:
                        context['companies'].add(company)
            
            # Extract positions
            positions = ['ceo', 'founder', 'president', 'chairman', 'director', 'executive', 'chief']
            for pos in positions:
                if pos in text:
                    context['positions'].add(pos)
        
        return context
    
    def validate_person_identity(self, search_result: Dict, donor: Dict, donor_context: Dict) -> Tuple[bool, float]:
        """
        RELAXED validation that a search result is about the correct donor.
        Returns (is_valid, confidence_score)
        """
        snippet = search_result.get('snippet', '').lower()
        title = search_result.get('title', '').lower()
        text = snippet + " " + title
        
        confidence = 0.0
        validation_points = []
        
        # Check location match
        donor_city = donor.get('city', '')
        donor_state = donor.get('state', '')
        if donor_city and len(donor_city) > 2 and donor_city.lower() in text:
            confidence += 0.25
            validation_points.append(f"city:{donor_city}")
        if donor_state and len(donor_state) > 1 and donor_state.lower() in text:
            confidence += 0.15
            validation_points.append(f"state:{donor_state}")
        
        # Check company/position context
        for company in donor_context.get('companies', set()):
            if company and len(company) > 3 and company in text:
                confidence += 0.3
                validation_points.append(f"company:{company}")
                break
        
        for position in donor_context.get('positions', set()):
            if position in text:
                confidence += 0.15
                validation_points.append(f"position:{position}")
                break
        
        # RELAXED: Accept if we have ANY validation OR if it's a direct name match search
        # This reduces false negatives
        is_valid = (
            confidence >= 0.2 or  # Lower threshold
            len(validation_points) >= 1 or  # At least one validation point
            '"' in search_result.get('q', '')  # Quoted name search (more specific)
        )
        
        if is_valid:
            log(f"      [OK] Validated (confidence: {confidence:.1f}, points: {', '.join(validation_points) if validation_points else 'direct search'})")
        else:
            log(f"      [SKIP] Low confidence (confidence: {confidence:.1f})")
        
        return is_valid, max(confidence, 0.3)  # Minimum confidence of 0.3 for valid results
    
    def deep_research_connection(self, donor: Dict, member: Dict, donor_context: Dict) -> List[Dict]:
        """
        Perform DEEP research using HYBRID approach:
        1. Google CSE gathers actual evidence about both individuals
        2. OpenAI o1 analyzes the evidence to extract connections
        """
        donor_name = f"{donor['first']} {donor.get('last', '')}"
        
        # Construct member name (including middle name/initial)
        member_name = self._construct_member_name(member)
        
        if not member_name:
            return []
        
        log(f"    Hybrid deep research: {donor_name} <-> {member_name}")
        
        # STEP 1: Gather actual evidence using Google CSE (if available)
        search_results = []
        
        if self.google and self.google.api_key:
            log(f"      [1/2] Gathering evidence via Google...")
            
            donor_last = donor.get('last', '')
            member_last = str(member.get("Last", "")).strip() if pd.notna(member.get("Last")) else ""
            
            search_queries = [
                f'"{donor_name}" "{member_name}"',
                f'"{donor_name}" AND "{member_name}"',
                f'"{donor_last}" "{member_last}"',
                f'"{donor_name}" "{member_name}" author',
                f'"{donor_name}" "{member_name}" article',
                f'"{donor_name}" "{member_name}" conference',
                f'"{donor_name}" "{member_name}" board',
                f'"{donor_name}" "{member_name}" foundation',
            ]
            
            for query in search_queries[:6]:  # Top 6 most relevant queries
                try:
                    results = self.google.query(query, num=5)
                    for res in results[:3]:  # Top 3 results per query
                        search_results.append({
                            'title': res.get('title', ''),
                            'snippet': res.get('snippet', ''),
                            'link': res.get('link', ''),
                            'query': query
                        })
                    time.sleep(0.3)
                except Exception as e:
                    log(f"        ! Search failed: {e}")
                    continue
            
            log(f"        Collected {len(search_results)} search results")
        else:
            log(f"      [1/2] Google CSE not available, using AI reasoning only")
        
        # STEP 2: Use OpenAI o1 to analyze the evidence
        if not self.ai_client:
            log(f"      ! OpenAI client not available")
            # Fallback to basic extraction if no AI
            return self._basic_extraction(search_results)
        
        log(f"      [2/2] Analyzing evidence with OpenAI {self.model}...")
        
        # Prepare context
        donor_info = f"""Donor: {donor_name}
Location: {donor.get('city', 'Unknown')}, {donor.get('state', 'Unknown')}"""
        
        member_info = f"""Inner Circle Member: {member_name}
Location: {member.get('City', 'Unknown')}, {member.get('State', 'Unknown')}
Notes: {str(member.get('Notes', ''))[:500] if pd.notna(member.get('Notes')) else 'None'}"""
        
        # Format search results for AI
        if search_results:
            evidence_text = "\n\nSearch Results Found:\n"
            for i, result in enumerate(search_results[:20], 1):  # Send top 20 results to AI
                evidence_text += f"\n{i}. Title: {result['title']}\n"
                evidence_text += f"   Snippet: {result['snippet']}\n"
                evidence_text += f"   URL: {result['link']}\n"
        else:
            evidence_text = "\n\nNo search results available. Use your knowledge to identify potential connections."
        
        prompt = f"""Analyze the connection between these two individuals:

{donor_info}

{member_info}
{evidence_text}

Task: Extract ALL connections between {donor_name} and {member_name} from the search results above.

For EACH connection you find, provide:
- type: Connection category (e.g., "co_authorship", "shared_board", "shared_event", "co_occurrence")
- detail: Specific description of the connection from the search results
- source: The URL or publication name from the search results
- confidence: Score from 0.0 to 1.0 based on evidence strength

Important:
- Only include connections that are explicitly mentioned in the search results
- Use exact quotes and citations from the results
- Be specific about what the connection is (e.g., "Co-authored article 'X' in Journal Y")
- If search results mention them together, include that as a connection
- Higher confidence for clear, direct connections (co-authorship, shared board)
- Lower confidence for indirect mentions or weak associations

Return ONLY a JSON array. If no connections found in results, return empty array [].

Example format:
[
  {{
    "type": "co_authorship",
    "detail": "Co-authored article 'Economic Freedom and Innovation' published in Journal of Economics 2020",
    "source": "https://example.com/article",
    "confidence": 0.95
  }}
]

JSON only:"""
        
        try:
            response = self.ai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "[" in content and "]" in content:
                json_start = content.index("[")
                json_end = content.rindex("]") + 1
                json_str = content[json_start:json_end]
                evidence = json.loads(json_str)
                
                log(f"        AI extracted {len(evidence)} connections")
                
                # Add query field for consistency
                for ev in evidence:
                    if 'query' not in ev:
                        ev['query'] = f'AI analysis: {donor_name} <-> {member_name}'
                
                return evidence
            else:
                log(f"        No valid JSON response from AI")
                return self._basic_extraction(search_results)
                
        except json.JSONDecodeError as e:
            log(f"      ! Failed to parse AI response: {e}")
            return self._basic_extraction(search_results)
        except Exception as e:
            log(f"      ! AI analysis failed: {e}")
            return self._basic_extraction(search_results)
    
    def _basic_extraction(self, search_results: List[Dict]) -> List[Dict]:
        """Fallback: Basic extraction if AI fails"""
        evidence = []
        for result in search_results[:10]:
            snippet = result.get('snippet', '')
            title = result.get('title', '')
            if snippet or title:
                connection_type = self.identify_connection_type(snippet + " " + title)
                evidence.append({
                    'type': connection_type,
                    'detail': f"{title}: {snippet[:250]}",
                    'source': result.get('link', ''),
                    'confidence': 0.5,
                    'query': result.get('query', '')
                })
        return evidence
    
    def identify_connection_type(self, text: str) -> str:
        """Identify the type of connection based on text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['interview', 'interviewed', 'conversation', 'podcast', 'spoke with']):
            return 'interview_conversation'
        elif any(word in text_lower for word in ['co-author', 'co-wrote', 'together wrote', 'article by']):
            return 'co_authorship'
        elif any(word in text_lower for word in ['panel', 'conference', 'symposium', 'summit']):
            return 'shared_event'
        elif any(word in text_lower for word in ['board', 'director', 'trustee']):
            return 'shared_board'
        elif any(word in text_lower for word in ['dinner', 'meeting', 'lunch']):
            return 'personal_meeting'
        elif any(word in text_lower for word in ['foundation', 'organization', 'institute']):
            return 'shared_organization'
        else:
            return 'co_occurrence'
    
    def find_all_connections(self, donor: Dict, findings: List[Dict]) -> List[Dict]:
        """
        Find connections between donor and ALL inner circle members.
        Returns list sorted by connection strength.
        """
        if self.df is None or len(self.df) == 0:
            log("  ! No inner circle members loaded")
            return []
        
        log(f"\n{'='*60}")
        log(f"DEEP CONNECTION RESEARCH: {donor['first']} {donor.get('last', '')}")
        log(f"Analyzing connections to {len(self.df)} inner circle members...")
        log(f"{'='*60}\n")
        
        # Extract donor context for validation
        donor_context = self.extract_donor_context(findings)
        log(f"  Donor context: {len(donor_context.get('companies', set()))} companies, "
            f"{len(donor_context.get('positions', set()))} positions identified")
        
        connections = []
        
        for idx, member in self.df.iterrows():
            # Construct member name (including middle name/initial)
            member_name = self._construct_member_name(member)
            
            if not member_name:
                log(f"\n[{idx+1}/{len(self.df)}] Skipping member with no name")
                continue
            
            log(f"\n[{idx+1}/{len(self.df)}] Researching: {member_name}")
            
            # Perform deep research
            evidence = self.deep_research_connection(donor, member, donor_context)
            
            # Calculate connection strength based on evidence quality
            connection_strength = self.calculate_connection_strength(evidence)
            
            connections.append({
                'member_name': member_name,
                'member_id': member.get('Client ID', ''),
                'member_city': member.get('City', ''),
                'member_state': member.get('State', ''),
                'member_notes': str(member.get('Notes', ''))[:300] if pd.notna(member.get('Notes')) else '',
                'evidence': evidence,
                'connection_strength': connection_strength,
                'evidence_count': len(evidence)
            })
            
            # Progress update
            if (idx + 1) % 5 == 0:
                strong = sum(1 for c in connections if c['connection_strength'] >= 50)
                log(f"\n  Progress: {idx+1}/{len(self.df)} members analyzed ({strong} strong connections so far)")
            
            # Small delay between members
            time.sleep(0.3)
        
        # Sort by connection strength
        connections.sort(key=lambda x: x['connection_strength'], reverse=True)
        
        # Summary
        strong_connections = [c for c in connections if c['connection_strength'] >= 50]
        medium_connections = [c for c in connections if 30 <= c['connection_strength'] < 50]
        
        log(f"\n{'='*60}")
        log(f"CONNECTION RESEARCH COMPLETE")
        log(f"  Strong connections (>=50): {len(strong_connections)}")
        log(f"  Medium connections (30-49): {len(medium_connections)}")
        log(f"  Total evidence pieces: {sum(c['evidence_count'] for c in connections)}")
        log(f"{'='*60}\n")
        
        return connections
    
    def calculate_connection_strength(self, evidence: List[Dict]) -> int:
        """Calculate overall connection strength from evidence"""
        if not evidence:
            return 0
        
        # Weight by confidence and type
        type_weights = {
            'co_authorship': 50,
            'interview_conversation': 45,
            'shared_board': 45,
            'shared_event': 40,
            'personal_meeting': 40,
            'organization_connection': 35,
            'shared_organization': 30,
            'co_occurrence': 20
        }
        
        total_score = 0
        seen_types = set()
        
        for ev in evidence:
            ev_type = ev.get('type', 'co_occurrence')
            confidence = ev.get('confidence', 0.5)
            
            base_weight = type_weights.get(ev_type, 20)
            score = base_weight * confidence
            
            # Diminishing returns for same type
            if ev_type in seen_types:
                score *= 0.5
            else:
                seen_types.add(ev_type)
            
            total_score += score
        
        return min(100, int(total_score))

# -----------------------------
# Enhanced DOCX Output
# -----------------------------

def render_enhanced_partner_profile(profile: Dict, out_dir="outputs") -> str:
    """
    Render a comprehensive partner profile matching the IHS template
    with enhanced connection section and proper citations.
    """
    os.makedirs(out_dir, exist_ok=True)
    p = profile["person"]
    name = f'{p["first"]} {p.get("last","")}'.strip()
    structured = profile.get("structured_data", {})
    political = profile.get("political_analysis", {})
    connections = profile.get("inner_circle_connections", [])

    doc = Document()
    
    # Title
    title = doc.add_heading("Partner Profile", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Main profile table
    table_data = [
        ("Name", name),
        ("Contact Info (Address)", f"{p.get('city','')}, {p.get('state','')}"),
        ("Salesforce Link", "https://theihs.lightning.force.com/lightning/r/Contact/[ID]/view"),
        ("Date of Birth", structured.get("date_of_birth", "TBD")),
        ("Family", structured.get("family", "TBD")),
        ("Education", structured.get("education", "TBD")),
        ("Current Position or\nOccupation", structured.get("current_position", "TBD")),
        ("Previous Positions", structured.get("previous_positions", "TBD")),
        ("Foundation and Nonprofit\nBoard Memberships", structured.get("board_memberships", "TBD")),
        ("Other Affiliations", structured.get("other_affiliations", "TBD")),
        ("Political Affiliation", structured.get("political_affiliation", "TBD")),
        ("Political Giving", structured.get("political_giving", "TBD")),
        ("Net Worth/Giving\nCapacity", structured.get("estimated_net_worth", "TBD")),
        ("Real Estate & Properties", structured.get("real_estate", "TBD")),
        ("Personal Interests", structured.get("personal_interests", "TBD")),
        ("Geographic Ties", structured.get("geographic_ties", "TBD")),
        ("Philanthropic Interests", structured.get("philanthropic_interests", "TBD")),
        ("Relevant Gifts", "TBD - Requires manual research"),
        ("Other Giving History", format_political_summary(political))
    ]

    table = doc.add_table(rows=len(table_data), cols=2)
    table.style = 'Light Grid Accent 1'
    
    for i, (label, value) in enumerate(table_data):
        row = table.rows[i]
        row.cells[0].text = label
        row.cells[0].paragraphs[0].runs[0].bold = True
        
        # Ensure value is a string (handle dict, list, None, etc.)
        if isinstance(value, dict):
            # If it's a dict, convert to JSON string
            value = json.dumps(value, indent=2)
        elif isinstance(value, list):
            # If it's a list, join items
            value = ", ".join(str(item) for item in value)
        elif value is None:
            value = "TBD"
        else:
            value = str(value)
        
        row.cells[1].text = value

    # Biography section
    doc.add_heading("Biography", level=2)
    biography = profile.get("biography", "Biography pending.")
    # Ensure biography is a string
    if not isinstance(biography, str):
        biography = str(biography) if biography else "Biography pending."
    bio_para = doc.add_paragraph(biography)
    
    # Political Giving Analysis (if significant)
    if political.get('total_given', 0) > 0:
        doc.add_heading("Political Contribution Analysis", level=2)
        para = doc.add_paragraph()
        para.add_run(f"Total Given: ${political.get('total_given', 0):,.0f}\n")
        para.add_run(f"Number of Contributions: {political.get('num_contributions', 0)}\n")
        para.add_run(f"Average Contribution: ${political.get('avg_contribution', 0):,.0f}\n")
        para.add_run(f"Giving Pattern: {political.get('pattern', 'Unknown')}\n")
        
        if political.get('top_recipients'):
            para.add_run("\nTop Recipients:\n")
            for recipient, amount in political['top_recipients'][:5]:
                para.add_run(f"  * {recipient}: ${amount:,.0f}\n")

    # ENHANCED CONNECTIONS SECTION
    doc.add_heading("Connections - Inner Circle Network", level=2)
    
    if connections:
        strong_connections = [c for c in connections if c['connection_strength'] >= 50]
        medium_connections = [c for c in connections if 30 <= c['connection_strength'] < 50]
        
        if strong_connections:
            doc.add_heading("Strong Connections", level=3)
            
            for conn in strong_connections:
                # Member name header
                para = doc.add_paragraph()
                run = para.add_run(f"* {conn['member_name']}")
                run.bold = True
                
                # Location and strength
                para.add_run(f" ({conn.get('member_city', '')}, {conn.get('member_state', '')})")
                para.add_run(f" - Connection Strength: {conn['connection_strength']}/100")
                
                # Evidence with citations
                if conn.get('evidence'):
                    evidence_para = doc.add_paragraph()
                    evidence_para.paragraph_format.left_indent = Inches(0.5)
                    
                    for ev in conn['evidence'][:5]:  # Top 5 pieces of evidence
                        ev_type = ev['type'].replace('_', ' ').title()
                        detail = ev['detail'][:300]
                        source = ev.get('source', '')
                        
                        # Handle source being a list or string
                        if isinstance(source, list):
                            source = ', '.join(str(s) for s in source if s)
                        elif not isinstance(source, str):
                            source = str(source) if source else ''
                        
                        # Format as citation
                        ev_run = evidence_para.add_run(f"  - {ev_type}: {detail}")
                        if source:
                            evidence_para.add_run(f"\n    Source: {source}\n")
                        else:
                            evidence_para.add_run("\n")
                
                # Add spacing
                doc.add_paragraph()
        
        if medium_connections:
            doc.add_heading("Medium Connections", level=3)
            
            for conn in medium_connections[:5]:  # Top 5 medium connections
                para = doc.add_paragraph()
                run = para.add_run(f"* {conn['member_name']}")
                run.bold = True
                para.add_run(f" ({conn.get('member_city', '')}, {conn.get('member_state', '')})")
                para.add_run(f" - Connection Strength: {conn['connection_strength']}/100")
                
                if conn.get('evidence'):
                    evidence_para = doc.add_paragraph()
                    evidence_para.paragraph_format.left_indent = Inches(0.5)
                    
                    for ev in conn['evidence'][:3]:  # Top 3 pieces
                        ev_type = ev['type'].replace('_', ' ').title()
                        detail = ev['detail'][:200]
                        evidence_para.add_run(f"  - {ev_type}: {detail}\n")
        
        # Other connections summary
        other_connections = [c for c in connections if c['connection_strength'] < 30 and c['evidence_count'] > 0]
        if other_connections:
            doc.add_heading("Other Potential Connections", level=3)
            para = doc.add_paragraph()
            para.add_run(f"Additional {len(other_connections)} inner circle members have minor or indirect connections. ")
            para.add_run("See detailed CSV export for complete analysis.")
    
    else:
        doc.add_paragraph("No connections identified through automated research. Manual review recommended.")

    # Research Metadata
    doc.add_heading("Research Metadata", level=2)
    metadata_para = doc.add_paragraph()
    metadata_para.add_run(f"Research Date: {profile['timestamp']}\n")
    metadata_para.add_run(f"Total Sources Analyzed: {len(profile.get('findings', []))}\n")
    metadata_para.add_run(f"Research Time: {profile.get('research_time', 0):.1f} seconds\n")
    metadata_para.add_run(f"Inner Circle Members Analyzed: {len(connections)}\n")
    strong_count = sum(1 for c in connections if c['connection_strength'] >= 50)
    metadata_para.add_run(f"Strong Connections Found: {strong_count}\n")

    # CITATIONS PAGE - List all sources used
    doc.add_page_break()
    doc.add_heading("Citations and Sources", level=1)
    
    # Collect all unique URLs from findings
    all_sources = set()
    
    # From main research findings
    for finding in profile.get('findings', []):
        url = finding.get('link', '')
        if isinstance(url, str):
            url = url.strip()
            if url and url.startswith('http'):
                all_sources.add(url)
        elif isinstance(url, list):
            # Handle list of URLs
            for u in url:
                if isinstance(u, str) and u.strip().startswith('http'):
                    all_sources.add(u.strip())
    
    # From connection evidence
    for conn in connections:
        for ev in conn.get('evidence', []):
            source = ev.get('source', '')
            
            # Handle different types of source data
            if isinstance(source, str):
                source = source.strip()
                if source and source.startswith('http'):
                    all_sources.add(source)
            elif isinstance(source, list):
                # Source is a list of URLs
                for s in source:
                    if isinstance(s, str):
                        s = s.strip()
                        if s and s.startswith('http'):
                            all_sources.add(s)
    
    # Sort and display
    sorted_sources = sorted(list(all_sources))
    
    if sorted_sources:
        doc.add_paragraph(f"Total unique sources consulted: {len(sorted_sources)}")
        doc.add_paragraph()  # Spacing
        
        # Group sources by domain for better organization
        from urllib.parse import urlparse
        sources_by_domain = {}
        for url in sorted_sources:
            try:
                domain = urlparse(url).netloc
                if domain not in sources_by_domain:
                    sources_by_domain[domain] = []
                sources_by_domain[domain].append(url)
            except:
                if 'other' not in sources_by_domain:
                    sources_by_domain['other'] = []
                sources_by_domain['other'].append(url)
        
        # Display by domain
        for domain in sorted(sources_by_domain.keys()):
            if domain != 'other':
                doc.add_heading(domain, level=3)
            else:
                doc.add_heading("Other Sources", level=3)
            
            for url in sources_by_domain[domain]:
                para = doc.add_paragraph(style='List Bullet')
                para.add_run(url)
    else:
        doc.add_paragraph("No external sources were cited in this research.")

    # Save document
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    out_path = os.path.join(out_dir, f"{safe_name} - Partner Profile.docx")
    doc.save(out_path)
    return out_path

def format_political_summary(political: Dict) -> str:
    """Format political giving for table display"""
    if not political or political.get('total_given', 0) == 0:
        return "No political contributions found in FEC database"
    
    total = political.get('total_given', 0)
    pattern = political.get('pattern', 'Unknown')
    
    summary = f"${total:,.0f} total donated; {pattern}"
    
    if political.get('top_recipients'):
        top_recipient, top_amount = political['top_recipients'][0]
        summary += f"; Top recipient: {top_recipient} (${top_amount:,.0f})"
    
    return summary

# -----------------------------
# Enhanced Research Orchestrator
# -----------------------------

class EnhancedResearcher:
    def __init__(self):
        # Load environment from spider.env
        env_path = "spider.env"
        if not os.path.exists(env_path):
            log(f"WARNING: {env_path} not found. Create this file with your API keys.")
            log("Example spider.env contents:")
            log("  OPENAI_API_KEY=your_key_here")
            log("  GOOGLE_API_KEY=your_key_here")
            log("  GOOGLE_CSE_ID=your_cse_id_here")
            
            # Create a sample spider.env file
            sample_env = """# IHS Donor Research API Configuration
# Fill in your actual API keys below (remove the 'your_' placeholders)

# OpenAI API key (required for AI synthesis and connection analysis)
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o
OPENAI_RESEARCH_MODEL=o1

# Google Custom Search Engine (required for gathering connection evidence)
# Best results require BOTH Google and OpenAI
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here

# FEC API key (optional, DEMO_KEY has rate limits)
FEC_API_KEY=DEMO_KEY

# Cache settings
CACHE_DIR=./cache
CACHE_TTL=86400

# Inner circle CSV file path
INNER_CIRCLE_CSV=Inner_Circle.csv
"""
            try:
                with open(env_path, 'w') as f:
                    f.write(sample_env)
                log(f"Created sample {env_path} file. Please edit it with your actual API keys.")
            except:
                pass
        else:
            load_dotenv(env_path)
            log(f"Loaded environment from {env_path}")
        
        # API Keys - Load from spider.env (NO hardcoded keys!)
        self.keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "OPENAI_RESEARCH_MODEL": os.getenv("OPENAI_RESEARCH_MODEL", "o1"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
            "GOOGLE_CSE_ID": os.getenv("GOOGLE_CSE_ID", ""),
            "FEC_API_KEY": os.getenv("FEC_API_KEY", "DEMO_KEY"),
        }
        
        # Check for placeholder values
        if "your_" in self.keys.get("OPENAI_API_KEY", "").lower():
            log("WARNING: OPENAI_API_KEY appears to be a placeholder. Update spider.env with your actual key.")
            self.keys["OPENAI_API_KEY"] = ""
        if "your_" in self.keys.get("GOOGLE_API_KEY", "").lower():
            log("WARNING: GOOGLE_API_KEY appears to be a placeholder. Update spider.env with your actual key.")
            self.keys["GOOGLE_API_KEY"] = ""
        
        # Initialize cache
        cache_dir = os.getenv("CACHE_DIR", "./cache")
        cache_ttl = int(os.getenv("CACHE_TTL", "86400"))
        self.cache = ResearchCache(cache_dir, cache_ttl)
        
        # Initialize clients
        self.google = GoogleCSE(self.keys["GOOGLE_API_KEY"], self.keys["GOOGLE_CSE_ID"], self.cache)
        self.fec = FECClient(self.keys["FEC_API_KEY"], self.cache)
        
        # Initialize AI synthesizer
        from openai import OpenAI
        self.ai_client = OpenAI(api_key=self.keys["OPENAI_API_KEY"]) if self.keys["OPENAI_API_KEY"] else None
        
        # Initialize connection finder
        inner_circle_paths = [
            os.getenv("INNER_CIRCLE_CSV", ""),
            "Inner_Circle.csv",
            "inner_circle.csv",
        ]
        
        self.inner_circle_path = None
        for path in inner_circle_paths:
            if path and os.path.exists(path):
                self.inner_circle_path = path
                break
        
        if self.inner_circle_path:
            # Deep research works best with both Google (for evidence) and OpenAI (for analysis)
            # But can work with either alone
            self.connection_finder = DeepConnectionFinder(
                self.inner_circle_path,
                self.google,
                self.cache,
                ai_client=self.ai_client,
                model=self.keys.get("OPENAI_RESEARCH_MODEL", "o1")
            )
        else:
            self.connection_finder = None

    def research(self, person: Dict) -> Dict:
        first, last = person["first"], person.get("last","")
        city, state = person.get("city"), person.get("state")

        log(f"\n{'='*70}")
        log(f"COMPREHENSIVE DONOR RESEARCH: {first} {last}")
        log(f"Location: {city or 'Unknown'}, {state or 'Unknown'}")
        log(f"{'='*70}")
        
        start_time = time.time()
        findings = []

        # 1) Google searches
        queries = self._generate_queries(first, last, city, state)
        log(f"\n[1/5] Running {len(queries)} search queries...")
        
        for i, q in enumerate(queries):
            if i > 0 and i % 10 == 0:
                log(f"      Progress: {i}/{len(queries)} searches completed")
            try:
                findings.extend(self.google.query(q, num=10))
            except Exception as e:
                log(f"      Search failed: {e}")
            time.sleep(0.25)
        
        log(f"      Collected {len(findings)} search results")

        # 2) FEC contributions
        log(f"\n[2/5] Searching FEC political contributions...")
        try:
            fec_results = self.fec.contributions(f"{first} {last}", state=state)
            findings.extend(fec_results)
            log(f"      Found {len(fec_results)} political contributions")
        except Exception as e:
            log(f"      FEC search failed: {e}")
            fec_results = []
        
        political_analysis = self.fec.analyze_political_giving(fec_results)

        # 3) Deduplicate
        log(f"\n[3/5] Deduplicating findings...")
        seen = set()
        unique = []
        for f in findings:
            key = (f.get("link",""), f.get("title",""))
            if key not in seen:
                seen.add(key)
                unique.append(f)
        findings = unique
        log(f"      {len(findings)} unique findings retained")

        # 4) AI synthesis
        log(f"\n[4/5] Synthesizing profile with AI...")
        if self.ai_client:
            try:
                biography = self.synthesize_biography(person, findings)
                giving_capacity = self.synthesize_giving_capacity(person, findings, political_analysis)
                structured = self.extract_structured_data(person, findings)
                log(f"      AI synthesis complete")
            except Exception as e:
                log(f"      AI synthesis error: {e}")
                biography = "Biography synthesis failed."
                giving_capacity = "Giving capacity analysis failed."
                structured = self._default_structured_data()
        else:
            log(f"      AI synthesis disabled (no OpenAI key)")
            biography = "AI synthesis requires OpenAI API key."
            giving_capacity = "AI synthesis requires OpenAI API key."
            structured = self._default_structured_data()

        # 5) Deep connection research
        log(f"\n[5/5] DEEP CONNECTION RESEARCH...")
        inner_circle_connections = []
        if self.connection_finder:
            try:
                inner_circle_connections = self.connection_finder.find_all_connections(person, findings)
                log(f"      Connection research complete")
            except Exception as e:
                log(f"      Connection research failed: {e}")
        else:
            if not self.inner_circle_path:
                log(f"      ! Inner circle CSV not found")
            else:
                log(f"      ! Connection finder not initialized")

        elapsed = time.time() - start_time
        
        log(f"\n{'='*70}")
        log(f"RESEARCH COMPLETE")
        log(f"  Total time: {elapsed:.1f} seconds")
        log(f"  Findings: {len(findings)}")
        log(f"  Connections analyzed: {len(inner_circle_connections)}")
        log(f"{'='*70}\n")

        return {
            "person": person,
            "findings": findings,
            "biography": biography,
            "giving_capacity": giving_capacity,
            "structured_data": structured,
            "inner_circle_connections": inner_circle_connections,
            "political_analysis": political_analysis,
            "timestamp": datetime.utcnow().isoformat(),
            "research_time": elapsed
        }
    
    def _generate_queries(self, first, last, city=None, state=None):
        """Generate comprehensive search queries matching PDF profile depth"""
        base = f'"{first} {last}"'
        geo = ""
        if city and state:
            geo = f' "{city}" "{state}"'
        elif state:
            geo = f' "{state}"'
        
        queries = [
            # Core biographical and professional
            f'{base} site:linkedin.com/in',
            f'{base} biography career history',
            f'{base} CEO OR founder OR president OR executive',
            f'{base} current position occupation',
            f'{base} previous positions employment history',
            
            # Education
            f'{base} education university college degree',
            f'{base} Harvard OR Yale OR Stanford OR MIT',
            f'{base} MBA OR masters OR bachelor degree',
            
            # Board positions and affiliations
            f'{base} "board of directors"',
            f'{base} board member OR director OR trustee',
            f'{base} chairman OR chair',
            f'{base} advisory board',
            
            # Nonprofit and philanthropy
            f'{base} nonprofit OR foundation',
            f'{base} philanthropist OR donation OR charitable',
            f'{base} giving capacity OR donor',
            f'{base} family foundation',
            
            # Publications and media
            f'{base} author OR book OR article OR published',
            f'{base} wrote OR co-authored',
            f'{base} interview OR podcast OR speaking',
            f'{base} quoted OR featured',
            
            # Wealth and business
            f'{base} "net worth" OR wealth OR millionaire',
            f'{base} investor OR investment',
            f'{base} entrepreneur OR startup OR venture',
            f'{base} site:sec.gov',
            f'{base} stock OR equity OR shares',
            
            # Professional connections
            f'{base} worked with OR collaborated',
            f'{base} mentor OR mentee',
            f'{base} colleague OR partner',
            f'{base} knows OR friends with',
            
            # Personal and family
            f'{base} spouse OR wife OR husband OR family',
            f'{base} married OR children',
            
            # Social media and online presence
            f'{base} twitter OR social media',
            f'{base} site:crunchbase.com',
            f'{base} site:forbes.com OR site:bloomberg.com',
        ]
        
        return [q + geo for q in queries]
    
    def synthesize_biography(self, person: Dict, findings: List[Dict]) -> str:
        """Generate comprehensive biography with detailed sections matching PDF profile"""
        if not self.ai_client:
            return "Biography synthesis requires OpenAI API key."
        
        person_name = f"{person['first']} {person.get('last', '')}"
        
        # Prepare all findings as context
        findings_text = "\n\n".join([
            f"Source: {f.get('title', 'Unknown')}\n{f.get('snippet', '')}\nURL: {f.get('link', '')}"
            for f in findings[:100]  # Use more findings for comprehensive profile
        ])
        
        prompt = f"""You are creating a comprehensive partner profile for {person_name}. Based on the research findings below, write detailed sections covering:

1. **Background and Education** - Educational history, degrees, institutions, formative experiences
2. **Professional Leadership and Roles** - Current and previous positions, companies led, major accomplishments
3. **Board Positions and Affiliations** - Corporate boards, nonprofit boards, advisory roles
4. **Key Professional Relationships and Connections** - Mentors, colleagues, collaborators, notable associations
5. **Interview History and Media Participation** - Notable interviews, podcasts, articles, speaking engagements
6. **Personal Life and Philanthropy** - Family information (if public), philanthropic activities, civic engagement

Write in a narrative style similar to an executive profile. Be comprehensive and detailed, using specific examples, dates, company names, and quantifiable achievements where available. Each section should be 2-4 paragraphs.

Research Findings:
{findings_text}

Write the comprehensive profile now, using clear section headers:"""
        
        try:
            response = self.ai_client.chat.completions.create(
                model=self.keys["OPENAI_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,  # Much longer for detailed profile
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log(f"  ! AI error: {e}")
            return "Biography synthesis failed."
    
    def synthesize_giving_capacity(self, person: Dict, findings: List[Dict], political: Dict) -> str:
        """Generate comprehensive giving capacity analysis matching PDF depth"""
        if not self.ai_client:
            return "Giving capacity analysis requires OpenAI API key."
        
        person_name = f"{person['first']} {person.get('last', '')}"
        total_given = political.get('total_given', 0)
        
        # Prepare detailed findings
        findings_text = "\n\n".join([
            f"Source: {f.get('title', '')}\nContent: {f.get('snippet', '')}"
            for f in findings[:60]
        ])
        
        prompt = f"""Analyze the giving capacity and estimated net worth of {person_name} in comprehensive detail, similar to a wealth profile.

Include these sections:

**Estimated Wealth:**
- Analyze their career trajectory and positions held to estimate net worth
- Identify executive compensation indicators (CEO roles, board positions, equity holdings)
- Note any venture funding, company valuations, or public company stock
- Mention specific dollar figures or ranges if available in sources
- Overall estimate their net worth category (millions, tens of millions, etc.)

**Lifestyle & Assets:**
- Foundation assets if they have a family foundation
- Real estate or other visible assets
- Investment patterns and capital raising ability
- Professional networks that indicate wealth level

**Giving Capacity Analysis:**
- What level of donations could they feasibly make (specific dollar ranges)
- Pattern of their giving based on political contributions: ${total_given:,.0f} total
- Comparison to typical donors at their wealth level
- Growth potential if current ventures succeed
- Strategic vs. splashy giving patterns

Be specific with numbers, dates, company names, and funding amounts where available. Write 3-5 detailed paragraphs.

Research Findings:
{findings_text}

Write the comprehensive wealth and giving capacity analysis now:"""
        
        try:
            response = self.ai_client.chat.completions.create(
                model=self.keys["OPENAI_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,  # Much longer for detailed analysis
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log(f"  ! AI error: {e}")
            return "Giving capacity analysis failed."
    
    def extract_structured_data(self, person: Dict, findings: List[Dict]) -> Dict:
        """Extract detailed structured data using AI"""
        if not self.ai_client:
            return self._default_structured_data()
        
        person_name = f"{person['first']} {person.get('last', '')}"
        
        # Prepare findings with more context
        findings_text = "\n\n".join([
            f"Title: {f.get('title', 'Unknown')}\nContent: {f.get('snippet', '')}\nURL: {f.get('link', '')}"
            for f in findings[:50]  # Use more findings
        ])
        
        prompt = f"""Extract detailed structured information about {person_name} from the research findings below.

Return ONLY valid JSON with these fields. Be specific and detailed:
{{
    "date_of_birth": "Full date, year, or age (e.g., 'January 15, 1950' or 'Age 74' or '1950'). Use 'TBD' if not found.",
    "education": "List all degrees with institutions (e.g., 'Bachelor's Degree: Harvard University, History; MBA: Stanford')",
    "current_position": "Current role(s) with company names (e.g., 'CEO of Company X, Board Chair of Organization Y')",
    "previous_positions": "Detailed list of previous major roles with companies (e.g., 'CEO of Company A (2010-2020); VP at Company B (2005-2010)')",
    "board_memberships": "All nonprofit and foundation board positions, current and past",
    "other_affiliations": "Corporate boards, advisory roles, professional associations, civic organizations",
    "family": "Spouse name and background/occupation, children and their info, parents if relevant (e.g., 'Married to Barbara Smith (former state legislator); two adult children')",
    "philanthropic_interests": "Specific causes, charitable organizations supported, donor-advised funds, foundation involvement",
    "political_affiliation": "Political party (Republican/Democrat/Independent/Libertarian) or political leaning. Use 'TBD' if unclear.",
    "political_giving": "Major political donations, PAC contributions, political organizations supported, political network connections",
    "real_estate": "Primary residence location, other properties, land holdings, investment properties (e.g., 'Primary residence: Sioux Falls, SD; Vacation home: Lake Okoboji, IA; Land development: Colorado')",
    "personal_interests": "Hobbies, recreational activities, lifestyle interests (e.g., 'Sailing, skiing, outdoor sports')",
    "estimated_net_worth": "Net worth range if mentioned, business valuation indicators, wealth indicators. If not explicitly stated, use phrases like 'High net worth indicated by...' or 'TBD'",
    "geographic_ties": "Cities, states, or regions where they live, work, or have strong connections (e.g., 'Based in Phoenix, AZ; Operations in Nevada; Alumni network in Boston')"
}}

Research Findings:
{findings_text}

Extract as much detail as possible. Use "TBD" only if truly not found. Return JSON only:"""
        
        try:
            response = self.ai_client.chat.completions.create(
                model=self.keys["OPENAI_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,  # Increased for comprehensive extraction with new fields
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            if "{" in content and "}" in content:
                json_str = content[content.index("{"):content.rindex("}")+1]
                data = json.loads(json_str)
                
                # Ensure all values are strings (not nested dicts or lists)
                for key, value in data.items():
                    if isinstance(value, dict):
                        # Convert nested dict to readable string
                        data[key] = json.dumps(value, indent=2)
                    elif isinstance(value, list):
                        # Convert list to comma-separated string
                        data[key] = ", ".join(str(item) for item in value)
                    elif value is None:
                        data[key] = "TBD"
                    else:
                        data[key] = str(value)
                
                return data
        except Exception as e:
            log(f"      ! Structured extraction failed: {e}")
        
        return self._default_structured_data()
    
    def _default_structured_data(self):
        return {
            "date_of_birth": "TBD",
            "education": "TBD",
            "current_position": "TBD",
            "previous_positions": "TBD",
            "board_memberships": "TBD",
            "other_affiliations": "TBD",
            "family": "TBD",
            "philanthropic_interests": "TBD",
            "political_affiliation": "TBD",
            "political_giving": "TBD",
            "real_estate": "TBD",
            "personal_interests": "TBD",
            "estimated_net_worth": "TBD",
            "geographic_ties": "TBD"
        }

# -----------------------------
# Main CLI
# -----------------------------

def load_people_from_csv(path: str) -> List[Dict]:
    """Load prospects from CSV"""
    df = None
    for enc in ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            log(f"  Loaded CSV with {enc} encoding")
            break
        except:
            continue
    
    if df is None:
        raise RuntimeError(f"Could not read CSV: {path}")

    people = []
    for _, row in df.iterrows():
        person = {"first":"", "last":"", "city":"", "state":""}
        for col in df.columns:
            val = str(row[col]).strip() if pd.notna(row[col]) else ""
            cl = col.lower()
            if "full" in cl and "name" in cl:
                parts = val.split()
                if len(parts)>=2:
                    person["first"] = parts[0]
                    person["last"] = " ".join(parts[1:])
            elif cl == "first" or "first_name" in cl: 
                person["first"] = person["first"] or val
            elif cl == "last" or "last_name" in cl: 
                person["last"] = person["last"] or val
            elif cl == "city": 
                person["city"] = val
            elif cl == "state": 
                person["state"] = val
        if person["first"]:
            people.append(person)
    return people

def main():
    log("="*70)
    log("IHS ENHANCED DONOR RESEARCH SYSTEM v5.0 (HYBRID)")
    log("Deep Connection Analysis: Google Evidence + AI Reasoning")
    log("="*70)
    
    r = EnhancedResearcher()

    # System checks
    log("\nSystem Configuration:")
    
    # API key status
    log("\nAPI Keys Status:")
    log(f"  OpenAI: {'[OK] Loaded' if r.keys['OPENAI_API_KEY'] else '[!] Missing - set in spider.env'}")
    log(f"  Google CSE: {'[OK] Loaded' if r.keys['GOOGLE_API_KEY'] else '[!] Missing - set in spider.env'}")
    log(f"  FEC: {'[OK] Using ' + r.keys['FEC_API_KEY'][:8] + '...' if r.keys['FEC_API_KEY'] else '[!] Missing'}")
    
    log("\nComponents Status:")
    if r.connection_finder:
        log(f"  [OK] Inner Circle: {len(r.connection_finder.df)} members loaded")
        
        # Show deep research capability
        has_google = bool(r.google.api_key)
        has_openai = bool(r.ai_client)
        research_model = r.keys.get('OPENAI_RESEARCH_MODEL', 'o1')
        
        if has_google and has_openai:
            log(f"  [OK] Deep Research: HYBRID MODE (Google + OpenAI {research_model})")
            log(f"       → Google CSE gathers evidence")
            log(f"       → OpenAI {research_model} analyzes connections")
        elif has_openai:
            log(f"  [PARTIAL] Deep Research: OpenAI {research_model} only (no Google CSE)")
            log(f"            Add Google API keys for better results")
        elif has_google:
            log(f"  [PARTIAL] Deep Research: Google CSE only (no AI analysis)")
            log(f"            Add OpenAI API key for intelligent extraction")
        else:
            log(f"  [!] Deep Research: Limited (no Google or OpenAI)")
    else:
        log(f"  [!] Inner Circle: Not found (looking for Inner_Circle.csv)")
    
    if r.ai_client:
        log(f"  [OK] OpenAI Synthesis: Enabled ({r.keys['OPENAI_MODEL']})")
    else:
        log(f"  [!] OpenAI: Disabled - add OPENAI_API_KEY to spider.env")
    
    if r.google.api_key:
        log(f"  [OK] Google CSE: Enabled")
    else:
        log(f"  [!] Google CSE: Disabled - add GOOGLE_API_KEY & GOOGLE_CSE_ID to spider.env")
    
    log(f"  [OK] Cache: {r.cache.cache_dir} (TTL: {r.cache.ttl}s)")

    # Check for data.csv
    default_csv = "data.csv"
    args = sys.argv[1:]
    batch = []
    
    if not args:
        if os.path.exists(default_csv):
            log(f"\n[OK] Found {default_csv}")
            batch = load_people_from_csv(default_csv)
            log(f"[OK] Loaded {len(batch)} prospects")
        else:
            log(f"\n[!] No {default_csv} found")
            log("\nUsage:")
            log(f"  1. Create '{default_csv}' with columns: First, Last, City, State")
            log(f"  2. Run: python3 test_spider.py")
            sys.exit(1)
    elif len(args) == 1 and args[0].lower().endswith(".csv"):
        batch = load_people_from_csv(args[0])
    else:
        first = args[0]
        last = args[1] if len(args)>1 else ""
        city = args[2] if len(args)>2 else ""
        state = args[3] if len(args)>3 else ""
        batch = [{"first":first,"last":last,"city":city,"state":state}]

    if not batch:
        log("\n[!] No prospects to process")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)

    # Process each person
    for i, person in enumerate(batch):
        log(f"\n{'='*70}")
        log(f"PROSPECT {i+1}/{len(batch)}: {person['first']} {person.get('last','')}")
        log(f"{'='*70}")
        
        try:
            prof = r.research(person)
            docx_path = render_enhanced_partner_profile(prof, out_dir="outputs")
            
            log(f"\n[OK] COMPLETED: {person['first']} {person.get('last','')}")
            log(f"  Document: {docx_path}")
            
            # Summary
            strong_connections = sum(1 for c in prof.get('inner_circle_connections', []) 
                                   if c['connection_strength'] >= 50)
            log(f"  Summary:")
            log(f"     * Strong connections: {strong_connections}")
            log(f"     * Total evidence: {sum(c['evidence_count'] for c in prof.get('inner_circle_connections', []))}")
            log(f"     * Political giving: ${prof.get('political_analysis', {}).get('total_given', 0):,.0f}")
            
        except Exception as e:
            log(f"\n[ERROR]: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if i < len(batch) - 1:
            log("\n  Waiting 5 seconds before next prospect...")
            time.sleep(5)
    
    log(f"\n{'='*70}")
    log(f"[OK] ALL RESEARCH COMPLETE")
    log(f"  Processed: {len(batch)} prospects")
    log(f"  Output: ./outputs/")
    log(f"{'='*70}")

if __name__ == "__main__":
    main()