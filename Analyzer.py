#!/usr/bin/env python3
"""
IHS Scholar Citation Finder - STRICT VERSION (Enhanced v3)
Only finds actual academic citations of scholars' research.

USAGE:
  python3 analyzer.py                    # All years (no filter)
  python3 analyzer.py --years 2024       # Single year
  python3 analyzer.py --years 2020-2026  # Year range
  python3 analyzer.py --years 2022,2023,2024  # Specific years
  python3 analyzer.py --limit 100        # Limit to first 100 files (for testing)

STRICT RULES:
- Only matches when scholar's research is CITED (not just named)
- Requires citation signals: supra, see, law review pattern, (year), etc.
- ONE match per scholar per document maximum
- Skips Table of Authorities, cover pages, amici lists
- Does NOT count: amicus signers, brief authors, attorneys, parties

ENHANCEMENTS (v3):
- Added amicus_brief_name column (who filed the brief)
- Fixed engagement column: Total Engagements Last 5 FYs
- Performance optimization for large scholar lists (52K+)
- Better progress reporting during processing
- Brief Color (Petitioner/Respondent/Neither party alignment)
- Column Definitions tab explaining all fields
- Confidence rating for each citation match
"""

import os
import sys
import pandas as pd
import re
import urllib.parse
import time
from collections import defaultdict
import argparse
import fitz  # PyMuPDF

# ============================================================================
# EXCEL SANITIZATION
# ============================================================================
# Regex to match illegal Excel characters (control chars except tab, newline, carriage return)
ILLEGAL_EXCEL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

def sanitize_for_excel(value):
    """Remove illegal characters that cause openpyxl to fail"""
    if isinstance(value, str):
        return ILLEGAL_EXCEL_CHARS.sub('', value)
    return value

def sanitize_dataframe(df):
    """Sanitize all string columns in a DataFrame for Excel export"""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(sanitize_for_excel)
    return df

# ============================================================================
# CONFIGURATION
# ============================================================================
CONTACTS_FILE = '/Users/jronyak/Desktop/amicus/contacts.csv'
AMICUS_FOLDER = '/Users/jronyak/Desktop/amicus'

# Minimum citation score to count as a real citation
MIN_CITATION_SCORE = 3

# Context windows
CONTEXT_CHARS = 400
EXTENDED_CONTEXT_CHARS = 800


def parse_years(year_str):
    """Parse year argument into a list of years"""
    if not year_str:
        return None
    
    years = []
    for part in year_str.split(','):
        part = part.strip()
        if '-' in part:
            # Range like "2020-2026"
            start, end = part.split('-')
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return years


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='IHS Scholar Citation Finder')
    parser.add_argument('--years', type=str, default='2020-2026',
                        help='Years to filter: "2024", "2020-2026", or "2022,2023,2024" (default: 2020-2026)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to process (for testing)')
    parser.add_argument('--all-years', action='store_true',
                        help='Process all years (no filter)')
    return parser.parse_args()

# ============================================================================
# EXCLUDED SCHOLARS (common names that cause false positives)
# ============================================================================
EXCLUDED_SCHOLARS = {
    'John Smith', 'James Smith', 'Michael Smith', 'David Smith',
    'Robert Smith', 'William Smith', 'Richard Smith', 'Thomas Smith',
    'John Johnson', 'James Johnson', 'Michael Johnson',
    'John Williams', 'James Williams', 'Michael Williams',
    'John Brown', 'James Brown', 'Michael Brown',
    'John Jones', 'James Jones', 'Michael Jones',
    'John Davis', 'James Davis', 'Michael Davis',
}

# ============================================================================
# LOAD SCHOLARS (with new Salesforce columns)
# ============================================================================
def load_scholars(contacts_file):
    """Load scholars from contacts CSV with engagement data"""
    print(f"Loading scholars from contacts file...")
    
    df = pd.read_csv(contacts_file, low_memory=False)
    print(f"Columns in CSV: {list(df.columns)}")
    
    scholars = []
    for _, row in df.iterrows():
        full_name = str(row.get('Full Name', '')).strip()
        if not full_name or full_name == 'nan' or len(full_name) < 5:
            continue
        
        if full_name in EXCLUDED_SCHOLARS:
            continue
        
        # Helper function to safely get numeric values
        def safe_int(val):
            try:
                if pd.isna(val) or str(val).strip() in ['', 'nan', 'None']:
                    return None
                return int(float(val))
            except (ValueError, TypeError):
                return None
        
        scholars.append({
            'full_name': full_name,
            'contact_id': str(row.get('CaseSafeID', '')),
            'organization': str(row.get('Primary Organization', '')),
            'discipline': str(row.get('Primary Discipline', '')),
            # New Salesforce columns - will be None if not in CSV
            'total_engaged': safe_int(row.get('Total_Engaged', row.get('Total Engaged', None))),
            'total_engagements_last_5_fy': safe_int(row.get('Total_Engagements_Last_5_FY', 
                                                           row.get('Total Engagements Last 5 FY',
                                                           row.get('Total Engagements Last 5 FYs', None)))),
            'primary_affiliation_record_type': str(row.get('Primary_Affiliation_Record_Type', 
                                                           row.get('Primary Affiliation Record Type', ''))).strip()
        })
    
    # Report on new columns
    has_engaged = sum(1 for s in scholars if s['total_engaged'] is not None)
    has_5fy = sum(1 for s in scholars if s['total_engagements_last_5_fy'] is not None)
    has_affil = sum(1 for s in scholars if s['primary_affiliation_record_type'] and s['primary_affiliation_record_type'] != 'nan')
    
    print(f"Loaded {len(scholars)} scholars")
    print(f"  - With Total_Engaged: {has_engaged}")
    print(f"  - With Total_Engagements_Last_5_FY: {has_5fy}")
    print(f"  - With Primary_Affiliation_Record_Type: {has_affil}")
    
    return scholars

# ============================================================================
# YEAR/TERM EXTRACTION (FIXED for October Term format)
# ============================================================================
def extract_year_from_content(text):
    """
    Extract year from document.
    Returns the October Term START year (e.g., 2025 for October Term 2025-2026)
    """
    if not text:
        return None
    
    search_text = text[:4000]
    
    # Pattern 1: Explicit "October Term 2025" or similar
    match = re.search(r'(?:OCTOBER|JANUARY)\s+TERM[,]?\s+(20\d{2})', search_text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: Docket with letter "No. 25A326" -> October Term 2025
    match = re.search(r'Nos?\.\s*(\d{2})[A-Z]-?\d+', search_text)
    if match:
        year_short = int(match.group(1))
        if 0 <= year_short <= 30:
            return str(2000 + year_short)
    
    # Pattern 3: Docket with hyphen "No. 24-1287" -> October Term 2024
    match = re.search(r'Nos?\.\s*(\d{2})-\d+', search_text)
    if match:
        year_short = int(match.group(1))
        if 0 <= year_short <= 30:
            return str(2000 + year_short)
    
    return None


def extract_term_from_content(text):
    """
    Extract Supreme Court term in SCOTUSblog format.
    Returns format like "October Term 2025" (which spans Oct 2025 - Jun 2026)
    """
    if not text:
        return None
    
    search_text = text[:3000]
    
    # Pattern 1: Explicit "OCTOBER TERM, 2024" or "OCTOBER TERM 2024"
    match = re.search(r'((?:OCTOBER|JANUARY)\s+TERM[,]?\s+\d{4})', search_text, re.IGNORECASE)
    if match:
        # Normalize format
        term_text = match.group(1).strip()
        # Extract year and format consistently
        year_match = re.search(r'(\d{4})', term_text)
        if year_match:
            return f"October Term {year_match.group(1)}"
    
    # Pattern 2: From docket number with letter (emergency applications)
    match = re.search(r'Nos?\.\s*(\d{2})[A-Z]-?\d+', search_text)
    if match:
        year_short = int(match.group(1))
        if 0 <= year_short <= 30:
            return f"October Term {2000 + year_short}"
    
    # Pattern 3: From standard docket number
    match = re.search(r'Nos?\.\s*(\d{2})-\d+', search_text)
    if match:
        year_short = int(match.group(1))
        if 0 <= year_short <= 30:
            return f"October Term {2000 + year_short}"
    
    return None


def extract_case_name(text, filename):
    """Extract case name from document"""
    if not text:
        return filename
    
    search_text = text[:2000]
    
    # Pattern: "PARTY v. PARTY"
    match = re.search(r'([A-Z][A-Z\s,\.&]+?)\s+v\.\s+([A-Z][A-Z\s,\.&]+?)(?:\s+No\.|\s+ON|\s*\n)', search_text)
    if match:
        p1 = match.group(1).strip()[:50]
        p2 = match.group(2).strip()[:50]
        return f"{p1} v. {p2}"
    
    # Use filename - clean it up
    base = os.path.splitext(filename)[0]
    base = re.sub(r'^brief_\d+_', '', base)
    # Extract meaningful part
    if 'Brief' in base:
        match = re.search(r'Brief.*?of\s+(.+?)(?:\s+filed|\s+VIDED|\.pdf|$)', base, re.IGNORECASE)
        if match:
            return f"BRIEF OF {match.group(1).strip()[:60]}"
    return base[:80]


def extract_actual_case_name(text):
    """
    Extract the actual Supreme Court case name (e.g., "LOPER BRIGHT v. RAIMONDO")
    Returns ONLY the party v. party format, or None if not found.
    This is separate from the brief title.
    """
    if not text:
        return None
    
    search_text = text[:4000]
    
    # Normalize whitespace for matching
    normalized = re.sub(r'\s+', ' ', search_text)
    
    # Remove common prefixes that shouldn't be in case name
    normalized = re.sub(r'IN THE Supreme Court of the United States\s*', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'Supreme Court of the United States\s*', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^IN THE\s+', '', normalized, flags=re.IGNORECASE)  # Remove leading "IN THE"
    
    # Fix double "v v." issue
    normalized = re.sub(r'\bv\s+v\.', 'v.', normalized)
    normalized = re.sub(r'\., v\.', ' v.', normalized)  # Fix "., v."
    
    # Pattern 1: "PARTY v. PARTY" with "ON WRIT" or "ON PETITION" terminator
    # Handles "et al.", mixed case in titles like "Secretary of Commerce"
    match = re.search(r'([A-Z][A-Za-z\s,\.&\']+?(?:et al\.)?)\s+v\.\s+([A-Z][A-Za-z\s,\.&\']+?)(?:\s+ON\s+WRIT|\s+ON\s+PETITION)', normalized, re.IGNORECASE)
    if match:
        p1 = match.group(1).strip()[:60]
        p2 = match.group(2).strip()[:60]
        # Clean up - remove Petitioner/Respondent/Appellant/Appellee
        p1 = re.sub(r',?\s*(?:Petitioner|Appellant|Respondent|Appellee)s?', '', p1, flags=re.IGNORECASE)
        p2 = re.sub(r',?\s*(?:Petitioner|Appellant|Respondent|Appellee)s?', '', p2, flags=re.IGNORECASE)
        p1 = re.sub(r'\s+', ' ', p1).strip(' ,.')
        p2 = re.sub(r'\s+', ' ', p2).strip(' ,.')
        if len(p1) > 2 and len(p2) > 2:
            return f"{p1} v. {p2}"
    
    # Pattern 2: With Petitioner/Respondent markers
    match = re.search(r'([A-Z][A-Za-z\s,\.&\']+?),?\s*(?:Petitioner|Appellant)s?,?\s*v\.\s+([A-Z][A-Za-z\s,\.&\']+?)(?:,?\s*(?:Respondent|Appellee))', normalized, re.IGNORECASE)
    if match:
        p1 = match.group(1).strip()[:60]
        p2 = match.group(2).strip()[:60]
        p1 = re.sub(r'\s+', ' ', p1).strip(' ,.')
        p2 = re.sub(r'\s+', ' ', p2).strip(' ,.')
        if len(p1) > 2 and len(p2) > 2:
            return f"{p1} v. {p2}"
    
    # Pattern 3: Simple "X v. Y" followed by No. or common terminators
    match = re.search(r'([A-Z][A-Z\s,\.\'&]+?)\s+v\.\s+([A-Z][A-Z\s,\.\'&]+?)(?:\s+No\.|\s+BRIEF|\s+ON\s)', normalized)
    if match:
        p1 = match.group(1).strip()[:60]
        p2 = match.group(2).strip()[:60]
        p1 = re.sub(r'\s+', ' ', p1).strip(' ,.')
        p2 = re.sub(r'\s+', ' ', p2).strip(' ,.')
        # Make sure we got something meaningful
        if len(p1) > 3 and len(p2) > 3:
            return f"{p1} v. {p2}"
    
    return None


def extract_case_number(text):
    """Extract Supreme Court docket number from document"""
    if not text:
        return None
    
    search_text = text[:3000]
    
    # Pattern 1: "No. 24-1287" or "Nos. 24-1287, 24-1288"
    match = re.search(r'Nos?\.\s*(\d{2}-\d+(?:\s*,\s*\d{2}-\d+)*)', search_text)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: Emergency application style "No. 25A326"
    match = re.search(r'Nos?\.\s*(\d{2}[A-Z]-?\d+)', search_text)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: Older style with full year "No. 2024-1287"
    match = re.search(r'Nos?\.\s*(20\d{2}-\d+)', search_text)
    if match:
        return match.group(1).strip()
    
    return None


def extract_amicus_brief_name(text, filename):
    """
    Extract the amicus brief name/title (e.g., "BRIEF OF Equal Protection Project")
    This is who filed the brief.
    """
    if not text:
        return filename
    
    search_text = text[:3000]
    
    # Pattern 1: "BRIEF OF [organization] AS AMICI CURIAE"
    match = re.search(r'BRIEF\s+(?:OF|FOR)\s+(.+?)\s+(?:AS\s+)?(?:AMICI?\s+CURIAE|IN\s+SUPPORT)', search_text, re.IGNORECASE)
    if match:
        name = match.group(1).strip()[:100]
        name = re.sub(r'\s+', ' ', name)
        return f"BRIEF OF {name}"
    
    # Pattern 2: "AMICUS CURIAE BRIEF OF [organization]"
    match = re.search(r'AMICI?\s+CURIAE\s+BRIEF\s+(?:OF|FOR)\s+(.+?)(?:\s+IN\s+SUPPORT|\s+URGING|\n)', search_text, re.IGNORECASE)
    if match:
        name = match.group(1).strip()[:100]
        name = re.sub(r'\s+', ' ', name)
        return f"BRIEF OF {name}"
    
    # Fallback to filename
    base = os.path.splitext(filename)[0]
    base = re.sub(r'^brief_\d+_', '', base)
    if 'Brief' in base:
        match = re.search(r'Brief.*?of\s+(.+?)(?:\s+filed|\s+VIDED|\.pdf|$)', base, re.IGNORECASE)
        if match:
            return f"BRIEF OF {match.group(1).strip()[:60]}"
    return base[:80]


# ============================================================================
# BRIEF COLOR / PARTY ALIGNMENT DETECTION
# ============================================================================
def extract_brief_color(text):
    """
    Determine which party the amicus brief supports.
    Returns: 'Petitioner' (blue), 'Respondent' (red), 'Neither' (green), or 'Unknown'
    
    In Supreme Court practice:
    - Blue brief = Supporting Petitioner
    - Red brief = Supporting Respondent  
    - Green brief = Supporting Neither Party / Court-appointed
    """
    if not text:
        return 'Unknown'
    
    search_text = text[:5000].upper()
    
    # Check for explicit party support statements
    petitioner_signals = [
        'IN SUPPORT OF PETITIONER',
        'SUPPORTING PETITIONER',
        'IN FAVOR OF PETITIONER',
        'FOR PETITIONER',
        'URGING REVERSAL',
        'IN SUPPORT OF APPELLANT',
        'SUPPORTING APPELLANT',
        'IN SUPPORT OF PLAINTIFF',
    ]
    
    respondent_signals = [
        'IN SUPPORT OF RESPONDENT',
        'SUPPORTING RESPONDENT', 
        'IN FAVOR OF RESPONDENT',
        'FOR RESPONDENT',
        'URGING AFFIRMANCE',
        'IN SUPPORT OF APPELLEE',
        'SUPPORTING APPELLEE',
        'IN SUPPORT OF DEFENDANT',
    ]
    
    neither_signals = [
        'IN SUPPORT OF NEITHER PARTY',
        'SUPPORTING NEITHER PARTY',
        'IN SUPPORT OF NEITHER',
        'NOT IN SUPPORT OF EITHER PARTY',
        'COURT-APPOINTED',
        'INVITED BY THE COURT',
    ]
    
    # Check neither first (most specific)
    for signal in neither_signals:
        if signal in search_text:
            return 'Neither'
    
    # Check petitioner
    for signal in petitioner_signals:
        if signal in search_text:
            return 'Petitioner'
    
    # Check respondent
    for signal in respondent_signals:
        if signal in search_text:
            return 'Respondent'
    
    return 'Unknown'


def get_brief_color_display(party):
    """Convert party alignment to color display"""
    color_map = {
        'Petitioner': '🔵 Petitioner (Blue)',
        'Respondent': '🔴 Respondent (Red)',
        'Neither': '🟢 Neither Party (Green)',
        'Unknown': '⚪ Unknown'
    }
    return color_map.get(party, '⚪ Unknown')


def get_influence_level(score):
    """Convert score to influence level string"""
    if score >= 20:
        return "🔥 VERY HIGH - Opinion explicitly adopted this scholar's reasoning/framework (quoted directly)"
    elif score >= 15:
        return "🔥 VERY HIGH - Opinion explicitly adopted this scholar's reasoning/framework"
    elif score >= 10:
        return "⭐ HIGH - Substantial engagement with scholar's work"
    elif score >= 7:
        return "MODERATE - Meaningful citation with analysis"
    elif score >= 5:
        return "MODERATE - Citation with context"
    elif score >= 3:
        return "LOW - Brief citation"
    else:
        return "MINIMAL - Simple mention"


def get_influence_description(score):
    """Get influence description"""
    if score >= 15:
        return "Court relied on analysis; Court adopted analytical framework"
    elif score >= 10:
        return "Court relied on analysis; Court found argument persuasive"
    elif score >= 7:
        return "Court cited with approval"
    else:
        return "Standard citation"


# ============================================================================
# CONFIDENCE RATING SYSTEM
# ============================================================================
def calculate_confidence(score, signals, context, scholar_name):
    """
    Calculate confidence rating for the citation match.
    Returns: tuple (confidence_level, confidence_score, confidence_notes)
    
    Confidence levels:
    - VERY HIGH: Near certain this is a valid academic citation
    - HIGH: Strong indicators of academic citation
    - MEDIUM: Likely valid but some ambiguity
    - LOW: Possible match but needs manual review
    """
    confidence_score = 0
    notes = []
    
    # Strong positive indicators
    if 'law review' in signals or 'L. Rev' in signals or 'L.J.' in signals:
        confidence_score += 30
        notes.append("Law review citation pattern")
    
    if 'supra' in signals.lower():
        confidence_score += 25
        notes.append("Supra reference (clear legal citation)")
    
    if 'See' in signals or 'see also' in signals.lower():
        confidence_score += 20
        notes.append("Citation signal present")
    
    if 'citing' in signals.lower():
        confidence_score += 20
        notes.append("Explicit citing reference")
    
    if 'page cite' in signals.lower():
        confidence_score += 15
        notes.append("Page number citation")
    
    # Year in citation adds credibility
    year_match = re.search(r'\((?:19|20)\d{2}\)', context)
    if year_match:
        confidence_score += 10
        notes.append("Year in citation")
    
    # Work title pattern
    if 'work title' in signals.lower():
        confidence_score += 15
        notes.append("Work title identified")
    
    # Check for academic context words near name
    academic_words = ['professor', 'dr.', 'argues', 'analysis', 'theory', 'research', 
                      'study', 'article', 'paper', 'journal', 'publication']
    context_lower = context.lower()
    academic_count = sum(1 for word in academic_words if word in context_lower)
    if academic_count >= 2:
        confidence_score += 15
        notes.append(f"Academic context ({academic_count} indicators)")
    elif academic_count == 1:
        confidence_score += 5
        notes.append("Some academic context")
    
    # Penalty for ambiguous contexts
    ambiguous_words = ['attorney', 'counsel', 'filed', 'signed', 'represented']
    ambiguous_count = sum(1 for word in ambiguous_words if word in context_lower)
    if ambiguous_count > 0:
        confidence_score -= 10 * ambiguous_count
        notes.append(f"⚠️ Ambiguous context ({ambiguous_count} role words)")
    
    # Penalty for very common last names (even if not excluded)
    last_name = scholar_name.split()[-1].lower()
    common_names = ['smith', 'johnson', 'williams', 'brown', 'jones', 'davis', 
                    'miller', 'wilson', 'moore', 'taylor', 'anderson', 'thomas',
                    'jackson', 'white', 'harris', 'martin', 'thompson', 'garcia']
    if last_name in common_names:
        confidence_score -= 10
        notes.append("⚠️ Common name - verify manually")
    
    # Boost for unique/distinctive names
    if len(scholar_name.split()) >= 3:
        confidence_score += 5
        notes.append("Distinctive name")
    
    # Base score contribution
    confidence_score += min(score * 2, 20)  # Cap at 20 from base score
    
    # Determine confidence level
    if confidence_score >= 70:
        level = "VERY HIGH"
    elif confidence_score >= 50:
        level = "HIGH"
    elif confidence_score >= 30:
        level = "MEDIUM"
    else:
        level = "LOW"
    
    return level, max(0, min(100, confidence_score)), "; ".join(notes) if notes else "Standard match"


# ============================================================================
# STRICT CITATION MATCHING
# ============================================================================
def find_all_citations(text, scholar_name, text_lower=None):
    """
    Find ALL citations of scholar's research in document.
    Returns: list of citation dictionaries with score, context, how_cited, etc.
    
    Note: Caller should pre-check if scholar_name appears in document for performance.
    """
    if not text or len(text) < 3500:
        return []
    
    # Skip first 3000 chars (cover, TOC, amici list)
    search_text = text[3000:]
    
    # Use pre-computed lowercase if provided
    if text_lower is not None:
        search_lower = text_lower[3000:]
    else:
        search_lower = search_text.lower()
    
    pattern = r'\b' + re.escape(scholar_name) + r'\b'
    
    all_citations = []
    seen_contexts = set()  # Track unique context snippets to avoid duplicates
    
    for match in re.finditer(pattern, search_text, re.IGNORECASE):
        pos = match.start()
        
        # Get context
        start = max(0, pos - 300)
        end = min(len(search_text), pos + 400)
        context = search_text[start:end]
        context_lower = context.lower()
        
        # Extended context
        ext_start = max(0, pos - EXTENDED_CONTEXT_CHARS)
        ext_end = min(len(search_text), pos + EXTENDED_CONTEXT_CHARS)
        extended_context = search_text[ext_start:ext_end]
        
        # SKIP: Table of Authorities
        preceding = search_text[max(0, pos-500):pos].lower()
        if 'table of' in preceding and ('authorities' in preceding or 'contents' in preceding):
            continue
        
        # SKIP: Attorney/counsel listing
        if any(x in context_lower for x in ['attorney', 'counsel', 'on the brief', 'esq.', 'law firm']):
            continue
        
        # SKIP: Party name pattern
        if re.search(r'\bv\.\s*' + re.escape(scholar_name.split()[-1].lower()), context_lower):
            continue
        
        # SKIP: Amici signer list (multiple professors/PhDs without citation signals)
        prof_count = len(re.findall(r'professor|ph\.d\.|university', context_lower))
        has_citation_signal = any(x in context_lower for x in ['see ', 'citing', 'supra', ' at ', 'l. rev', 'l.j.'])
        if prof_count >= 3 and not has_citation_signal:
            continue
        
        # SCORE: Citation signals
        score = 0
        signals = []
        
        # Strong signals
        if re.search(r'\bsee\s+' + re.escape(scholar_name.split()[0].lower()), context_lower):
            score += 5
            signals.append("See")
        if re.search(r'\bsee also\b', context_lower):
            score += 4
            signals.append("See also")
        if re.search(r'\bciting\b', context_lower):
            score += 5
            signals.append("citing")
        if re.search(r',\s*supra', context_lower):
            score += 6
            signals.append("supra reference")
        elif re.search(r'\bsupra\b', context_lower):
            score += 5
            signals.append("supra")
        
        # Medium signals
        if re.search(r'\bat\s+\d+', context_lower):
            score += 3
            signals.append("page cite")
        if re.search(r'\baccord\b', context_lower):
            score += 3
            signals.append("accord")
        
        # Law review / journal pattern
        if re.search(r'\d+\s+\w+\.?\s*L\.\s*(?:Rev|J)', context):
            score += 6
            signals.append("law review")
        
        # Year in parens near name
        year_match = re.search(r'\((\d{4})\)', context)
        if year_match:
            score += 2
            signals.append(f"({year_match.group(1)})")
        
        # Book/article title pattern
        if re.search(re.escape(scholar_name) + r',\s+[A-Z][^,]{10,60}[,\s]+\d+', context, re.IGNORECASE):
            score += 3
            signals.append("work title")
        
        # Must meet minimum score
        if score >= MIN_CITATION_SCORE:
            # Create a normalized context key for duplicate detection
            # Use a shortened, normalized version of the context
            context_key = re.sub(r'\s+', ' ', context_lower[:200].strip())
            
            if context_key not in seen_contexts:
                seen_contexts.add(context_key)
                all_citations.append({
                    'score': score,
                    'context': context.strip(),
                    'extended_context': extended_context.strip(),
                    'how_cited': ', '.join(signals) if signals else 'direct citation',
                    'position': pos + 3000
                })
    
    return all_citations


def find_citation(text, scholar_name, text_lower=None):
    """
    Find if scholar's RESEARCH is actually cited.
    Returns: dict with score, context, how_cited, extended_context, citation_count, duplicate_note OR None
    """
    all_citations = find_all_citations(text, scholar_name, text_lower)
    
    if not all_citations:
        return None
    
    # Get the best citation (highest score)
    best_citation = max(all_citations, key=lambda x: x['score'])
    
    # Add duplicate information
    citation_count = len(all_citations)
    if citation_count > 1:
        # Consolidate unique citation types
        all_how_cited = set()
        for c in all_citations:
            for signal in c['how_cited'].split(', '):
                all_how_cited.add(signal.strip())
        
        best_citation['citation_count'] = citation_count
        best_citation['duplicate_note'] = f"[{citation_count} citations in this document]"
        best_citation['all_citation_types'] = ', '.join(sorted(all_how_cited))
    else:
        best_citation['citation_count'] = 1
        best_citation['duplicate_note'] = ""
        best_citation['all_citation_types'] = best_citation['how_cited']
    
    return best_citation


def get_doc_type(text):
    """Determine if document is Opinion or Brief"""
    first_3000 = text[:3000].upper()
    
    # Brief indicators (check first - more common)
    brief_patterns = ['AMICI CURIAE', 'AMICUS CURIAE', 'BRIEF OF', 'BRIEF FOR', 
                      'IN SUPPORT OF PETITIONER', 'IN SUPPORT OF RESPONDENT']
    for p in brief_patterns:
        if p in first_3000:
            return 'Amicus Brief'
    
    # Opinion indicators
    opinion_patterns = ['DELIVERED THE OPINION', 'PER CURIAM', 'IT IS SO ORDERED',
                        'OPINION OF THE COURT', 'SYLLABUS']
    for p in opinion_patterns:
        if p in first_3000:
            return 'Opinion'
    
    return 'Amicus Brief'


# ============================================================================
# PROCESS FILES
# ============================================================================
def process_file_single(file_path, scholars):
    """Process a single PDF - return at most ONE citation per scholar"""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        if not text.strip() or len(text) < 1000:
            return []
        
        filename = os.path.basename(file_path)
        
        # Pre-compute lowercase text for fast scholar lookups (BIG performance boost)
        text_lower = text.lower()
        search_text_lower = text_lower[3000:] if len(text_lower) > 3000 else text_lower
        
        # Extract metadata
        year = extract_year_from_content(text)
        if not year:
            year = "Unknown"
        
        term = extract_term_from_content(text)
        if not term and year != "Unknown":
            term = f"October Term {year}"
        if not term:
            term = ""
        
        # Extract actual case name (party v. party format) - separate from brief name
        case_name = extract_actual_case_name(text)
        if not case_name:
            case_name = "Unknown Case"
        
        case_number = extract_case_number(text)
        amicus_brief_name = extract_amicus_brief_name(text, filename)
        doc_type = get_doc_type(text)
        
        # Extract brief color (party alignment)
        brief_party = extract_brief_color(text)
        brief_color = get_brief_color_display(brief_party)
        
        # Find citations - ONE per scholar max
        # Use pre-computed lowercase search text for fast lookups
        matches = []
        for scholar in scholars:
            # Quick check: skip if scholar name not in document at all
            scholar_lower = scholar['full_name'].lower()
            if scholar_lower not in search_text_lower:
                continue
            
            citation = find_citation(text, scholar['full_name'], text_lower)
            
            if citation:
                score = citation['score']
                # Double score for opinions
                if doc_type == 'Opinion':
                    score = score * 2
                
                # Calculate confidence
                confidence_level, confidence_score, confidence_notes = calculate_confidence(
                    score, citation['how_cited'], citation['context'], scholar['full_name']
                )
                
                # Build clean how_mentioned without duplicates
                how_mentioned_parts = []
                
                # Add the core context (cleaned up)
                clean_context = citation['context'][:400].strip()
                # Remove repeated whitespace
                clean_context = re.sub(r'\s+', ' ', clean_context)
                how_mentioned_parts.append(clean_context)
                
                # Add duplicate note if applicable
                if citation['duplicate_note']:
                    how_mentioned_parts.append(citation['duplicate_note'])
                
                how_mentioned = ' '.join(how_mentioned_parts)
                
                matches.append({
                    'scholar_name': scholar['full_name'],
                    'contact_id': scholar['contact_id'],
                    'amicus_brief_name': amicus_brief_name,
                    'case_name': case_name,
                    'case_number': case_number,
                    'year': year,
                    'term': term,
                    'doc_type': doc_type,
                    'brief_color': brief_color if doc_type != 'Opinion' else 'N/A (Opinion)',
                    'influence_level': get_influence_level(score),
                    'influence_score': score,
                    'influence_description': get_influence_description(score),
                    'confidence_level': confidence_level,
                    'confidence_score': confidence_score,
                    'confidence_notes': confidence_notes,
                    'how_mentioned': how_mentioned,
                    'citation_types': citation['all_citation_types'],
                    'citation_count_in_doc': citation['citation_count'],
                    'cross_doc_note': '',  # Filled in post-processing
                    'validation_indicators': citation['how_cited'],
                    'organization': scholar['organization'],
                    'discipline': scholar['discipline'],
                    # Salesforce columns
                    'total_engaged': scholar['total_engaged'],
                    'total_engagements_last_5_fy': scholar['total_engagements_last_5_fy'],
                    'primary_affiliation_record_type': scholar['primary_affiliation_record_type'],
                    # Extended data
                    'extended_context': citation['extended_context'],
                    'file': filename,
                })
        
        return matches
        
    except Exception as e:
        return []


def process_all_files(folder_path, scholars, filter_years=None, limit_files=None):
    """Process all PDFs with year filtering"""
    print(f"\nScanning for PDF files in {folder_path}...")
    
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, f))
    
    print(f"Found {len(pdf_files):,} total PDF files")
    
    # Filter by year if specified
    if filter_years:
        filter_set = set(filter_years)
        filtered = []
        for pdf_file in pdf_files:
            # Check for filing_year_YYYY folder pattern (most reliable)
            filing_year_match = re.search(r'filing_year_(\d{4})', pdf_file)
            if filing_year_match:
                year = int(filing_year_match.group(1))
                if year in filter_set:
                    filtered.append(pdf_file)
                continue
            
            # Fallback: Check for single year in path (but not in range pattern)
            path_without_range = re.sub(r'\d{4}[-–]\d{4}', '', pdf_file)
            year_matches = re.findall(r'\b(20\d{2})\b', path_without_range)
            for y in year_matches:
                if int(y) in filter_set:
                    filtered.append(pdf_file)
                    break
        
        pdf_files = filtered
        year_list = sorted(filter_years)
        if len(year_list) > 3:
            print(f"Filtered to {len(pdf_files):,} files for years {min(year_list)}-{max(year_list)}")
        else:
            print(f"Filtered to {len(pdf_files):,} files for years {year_list}")
    
    # Apply limit if set
    if limit_files:
        pdf_files = pdf_files[:limit_files]
        print(f"LIMITED to first {limit_files} files for testing")
    
    total = len(pdf_files)
    if total == 0:
        print("No files to process!")
        return []
    
    print(f"Processing {total:,} files...")
    print(f"  (Searching {len(scholars):,} scholars per file - this may take a while)", flush=True)
    
    all_matches = []
    start_time = time.time()
    first_file_time = None
    
    for i, pdf_file in enumerate(pdf_files, 1):
        file_start = time.time()
        
        # Show progress for EVERY file until we know it's working
        if i <= 10:
            print(f"  [{i}/{total}] Opening: {os.path.basename(pdf_file)[:50]}...", end='', flush=True)
        
        matches = process_file_single(pdf_file, scholars)
        all_matches.extend(matches)
        
        file_time = time.time() - file_start
        
        # Show result for first 10 files
        if i <= 10:
            print(f" done ({file_time:.1f}s, {len(matches)} citations)", flush=True)
        
        # After first file, estimate total time
        if i == 1:
            estimated_total = file_time * total / 60
            print(f"  Estimated total time: {estimated_total:.1f} minutes", flush=True)
            if estimated_total > 60:
                print(f"  ⚠️  This will take over an hour! Consider using --limit 50 to test first.", flush=True)
        
        # Progress every 25 files after the first 10
        if i > 10 and (i % 25 == 0 or i == total):
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (total - i) / rate / 60 if rate > 0 else 0
            print(f"  Progress: {i:,}/{total:,} ({100*i//total}%) - {len(all_matches):,} citations - ~{remaining:.1f} min remaining", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Processing complete in {elapsed/60:.1f} minutes: {len(all_matches)} scholar citations found")
    
    return all_matches


# ============================================================================
# POST-PROCESS: IDENTIFY CROSS-DOCUMENT DUPLICATES
# ============================================================================
def identify_scholar_duplicates(matches):
    """
    Identify and annotate when a scholar appears multiple times across documents.
    Also identify when same scholar is cited multiple times in same case (different briefs).
    """
    # Group by scholar
    scholar_docs = defaultdict(list)
    for m in matches:
        scholar_docs[m['scholar_name']].append(m)
    
    # Group by scholar + case
    scholar_case = defaultdict(list)
    for m in matches:
        key = (m['scholar_name'], m['case_name'])
        scholar_case[key].append(m)
    
    # Annotate matches
    for m in matches:
        scholar = m['scholar_name']
        case = m['case_name']
        
        # Total documents citing this scholar
        total_docs = len(scholar_docs[scholar])
        
        # Documents in same case citing this scholar
        same_case_docs = len(scholar_case[(scholar, case)])
        
        # Build duplicate note
        duplicate_notes = []
        
        if same_case_docs > 1:
            duplicate_notes.append(f"⚠️ Cited in {same_case_docs} docs for same case")
        
        if total_docs > 1:
            duplicate_notes.append(f"📚 Scholar cited in {total_docs} total documents")
        
        m['cross_doc_note'] = '; '.join(duplicate_notes) if duplicate_notes else ''
        m['scholar_total_citations'] = total_docs
        m['scholar_same_case_citations'] = same_case_docs
    
    return matches


# ============================================================================
# COLUMN DEFINITIONS (for new tab)
# ============================================================================
def get_column_definitions():
    """Return column definitions for the Column Definitions tab"""
    definitions = [
        {
            'Column': '--- DATA SOURCES ---',
            'Display Name': 'PDF Collection Sources',
            'Description': 'PDFs are collected by amicus.py scraper from: CourtListener API, SCOTUSblog, Justia, Supreme Court website, Library of Congress. Cases are organized into filing_year folders based on docket number (+1 offset: docket 24-xxxx → filing_year_2025).',
            'Source': 'amicus.py scraper',
            'Calculation': 'See scotusblog.com/case-files/terms/ for term-based case listings'
        },
        {
            'Column': 'scholar_name',
            'Display Name': 'Scholar Name',
            'Description': 'Full name of the IHS scholar whose work was cited',
            'Source': 'Salesforce contacts.csv',
            'Calculation': 'Direct match from contacts file'
        },
        {
            'Column': 'contact_id',
            'Display Name': 'Contact ID',
            'Description': 'Salesforce CaseSafeID for the scholar',
            'Source': 'Salesforce contacts.csv',
            'Calculation': 'Direct match from contacts file'
        },
        {
            'Column': 'amicus_brief_name',
            'Display Name': 'Amicus Brief Name',
            'Description': 'Title of the amicus brief - who filed it (e.g., "BRIEF OF Equal Protection Project")',
            'Source': 'Extracted from PDF',
            'Calculation': 'Pattern matching for "BRIEF OF [Organization]" format in first 3000 characters, falls back to filename'
        },
        {
            'Column': 'case_name',
            'Display Name': 'Case Name',
            'Description': 'Supreme Court case name in party v. party format (e.g., "LOPER BRIGHT v. RAIMONDO"). Shows "Unknown Case" if not found.',
            'Source': 'Extracted from PDF',
            'Calculation': 'Regex pattern matching for "PARTY v. PARTY" format in first 4000 characters'
        },
        {
            'Column': 'case_number',
            'Display Name': 'Case/Docket Number',
            'Description': 'Supreme Court docket number (e.g., "24-1287" or "25A326")',
            'Source': 'Extracted from PDF',
            'Calculation': 'Regex pattern matching for "No. XX-XXXX" format in first 3000 characters'
        },
        {
            'Column': 'year',
            'Display Name': 'Year',
            'Description': 'October Term year derived from docket number. Docket "24-xxxx" = October Term 2024. Note: The scraper organizes files into filing_year folders with +1 offset (filing_year_2025 contains OT2024/docket 24-xxxx cases).',
            'Source': 'Extracted from PDF docket number',
            'Calculation': 'First two digits of docket number (e.g., "24-1287" → 2024). This matches SCOTUSblog October Term numbering.'
        },
        {
            'Column': 'term',
            'Display Name': 'Term',
            'Description': 'Supreme Court term in SCOTUSblog format. October Term 2024 runs Oct 2024 - Jun 2025. Folder mapping: filing_year_2025 = OT2024, filing_year_2026 = OT2025.',
            'Source': 'Extracted from PDF',
            'Calculation': 'From explicit "October Term YYYY" in document or derived from docket number prefix. Matches SCOTUSblog term pages (e.g., scotusblog.com/case-files/terms/ot2024/).'
        },
        {
            'Column': 'doc_type',
            'Display Name': 'Document Type',
            'Description': 'Whether the document is an Amicus Brief or Court Opinion',
            'Source': 'Extracted from PDF',
            'Calculation': 'Pattern matching for brief indicators (AMICI CURIAE, BRIEF OF) vs opinion indicators (PER CURIAM, IT IS SO ORDERED)'
        },
        {
            'Column': 'brief_color',
            'Display Name': 'Brief Color',
            'Description': 'Party alignment: 🔵 Petitioner (Blue), 🔴 Respondent (Red), 🟢 Neither (Green)',
            'Source': 'Extracted from PDF',
            'Calculation': 'Pattern matching for "IN SUPPORT OF PETITIONER/RESPONDENT/NEITHER PARTY" in first 5000 characters'
        },
        {
            'Column': 'influence_level',
            'Display Name': 'Influence Level',
            'Description': 'Categorical rating of how substantially the scholar\'s work was engaged with',
            'Source': 'Algorithm-generated',
            'Calculation': 'Based on influence_score thresholds: VERY HIGH (15+), HIGH (10-14), MODERATE (5-9), LOW (3-4)'
        },
        {
            'Column': 'influence_score',
            'Display Name': 'Influence Score',
            'Description': 'Numeric score (0-20+) measuring depth of engagement with scholar\'s work',
            'Source': 'Algorithm-generated',
            'Calculation': '''Additive scoring based on citation signals found:
• "See [Scholar]": +5 points
• "See also": +4 points  
• "citing": +5 points
• "supra" reference: +5-6 points
• Page citation ("at 123"): +3 points
• "accord": +3 points
• Law review pattern: +6 points
• Year in parentheses: +2 points
• Work title pattern: +3 points
• DOUBLED if found in Court Opinion (vs brief)'''
        },
        {
            'Column': 'influence_description',
            'Display Name': 'Influence Description',
            'Description': 'Human-readable summary of what the influence score indicates',
            'Source': 'Algorithm-generated',
            'Calculation': '''Based on score:
• 15+: "Court relied on analysis; Court adopted analytical framework"
• 10-14: "Court relied on analysis; Court found argument persuasive"
• 7-9: "Court cited with approval"
• <7: "Standard citation"'''
        },
        {
            'Column': 'confidence_level',
            'Display Name': 'Confidence Level',
            'Description': 'Categorical rating of certainty that this is a valid academic citation (vs false positive)',
            'Source': 'Algorithm-generated',
            'Calculation': 'Based on confidence_score thresholds: VERY HIGH (70+), HIGH (50-69), MEDIUM (30-49), LOW (<30)'
        },
        {
            'Column': 'confidence_score',
            'Display Name': 'Confidence Score',
            'Description': 'Numeric score (0-100) measuring certainty of the citation match',
            'Source': 'Algorithm-generated',
            'Calculation': '''Additive/subtractive scoring:
POSITIVE indicators:
• Law review citation pattern: +30
• "Supra" reference: +25
• Citation signal (See, citing): +20
• Page number citation: +15
• Work title identified: +15
• Year in citation: +10
• Academic context words: +5-15
• Distinctive name (3+ parts): +5
• Base influence score contribution: up to +20

NEGATIVE indicators:
• Role words (attorney, counsel): -10 each
• Common last name: -10'''
        },
        {
            'Column': 'confidence_notes',
            'Display Name': 'Confidence Notes',
            'Description': 'Explanation of why the confidence score is what it is',
            'Source': 'Algorithm-generated',
            'Calculation': 'Lists the specific positive and negative indicators found for this citation'
        },
        {
            'Column': 'how_mentioned',
            'Display Name': 'How Mentioned',
            'Description': 'Context snippet showing how the scholar was cited in the document',
            'Source': 'Extracted from PDF',
            'Calculation': '~400 characters of text surrounding the scholar name mention, plus duplicate count if multiple citations'
        },
        {
            'Column': 'citation_types',
            'Display Name': 'Citation Types',
            'Description': 'All citation signals found across ALL instances of this scholar in this document',
            'Source': 'Algorithm-generated',
            'Calculation': 'Aggregated list of signals (See, supra, law review, etc.) from all citation instances'
        },
        {
            'Column': 'citation_count_in_doc',
            'Display Name': 'Citations in Document',
            'Description': 'Number of times this scholar was cited in this specific document',
            'Source': 'Algorithm-generated',
            'Calculation': 'Count of unique citation instances meeting minimum score threshold'
        },
        {
            'Column': 'cross_doc_note',
            'Display Name': 'Cross-Document Notes',
            'Description': 'Flags when scholar appears in multiple documents or multiple briefs for same case',
            'Source': 'Algorithm-generated',
            'Calculation': 'Post-processing analysis comparing scholar appearances across all processed documents'
        },
        {
            'Column': 'validation_indicators',
            'Display Name': 'Validation Indicators',
            'Description': 'Citation signals found for the PRIMARY (highest-scored) citation instance',
            'Source': 'Algorithm-generated',
            'Calculation': 'List of signals from the single best citation match for this scholar in this document'
        },
        {
            'Column': 'organization',
            'Display Name': 'Organization',
            'Description': 'Scholar\'s primary organization/affiliation',
            'Source': 'Salesforce contacts.csv',
            'Calculation': 'Direct match from "Primary Organization" column'
        },
        {
            'Column': 'discipline',
            'Display Name': 'Discipline',
            'Description': 'Scholar\'s primary academic discipline',
            'Source': 'Salesforce contacts.csv',
            'Calculation': 'Direct match from "Primary Discipline" column'
        },
        {
            'Column': 'total_engaged',
            'Display Name': 'Total Engaged',
            'Description': 'Total number of engagements with IHS (all-time)',
            'Source': 'Salesforce contacts.csv',
            'Calculation': 'Direct match from "Total_Engaged" column (requires column in CSV export)'
        },
        {
            'Column': 'total_engagements_last_5_fy',
            'Display Name': 'Engagements Last 5 FY',
            'Description': 'Total engagements in the last 5 fiscal years',
            'Source': 'Salesforce contacts.csv',
            'Calculation': 'Direct match from "Total_Engagements_Last_5_FY" column (requires column in CSV export)'
        },
        {
            'Column': 'primary_affiliation_record_type',
            'Display Name': 'Primary Affiliation Record Type',
            'Description': 'Type of primary affiliation record in Salesforce',
            'Source': 'Salesforce contacts.csv',
            'Calculation': 'Direct match from "Primary_Affiliation_Record_Type" column (requires column in CSV export)'
        },
    ]
    return definitions


# ============================================================================
# CREATE REPORT - EXACT FORMAT MATCH
# ============================================================================
def create_report(matches):
    """Create Excel report matching exact format with new confidence column"""
    output_file = '/Users/jronyak/Desktop/amicus/ihs_scholar_impact_analysis.xlsx'
    
    print(f"\nCreating Excel report...")
    
    # Post-process to identify cross-document duplicates
    matches = identify_scholar_duplicates(matches)
    
    # Main columns (with new confidence, engagement, and brief color columns)
    main_columns = [
        'scholar_name', 'contact_id', 'amicus_brief_name', 'case_name', 'case_number', 'year', 'term', 'doc_type',
        'brief_color',
        'influence_level', 'influence_score', 'influence_description',
        'confidence_level', 'confidence_score', 'confidence_notes',
        'how_mentioned', 'citation_types', 'citation_count_in_doc',
        'cross_doc_note', 'validation_indicators',
        'organization', 'discipline',
        'total_engaged', 'total_engagements_last_5_fy', 'primary_affiliation_record_type'
    ]
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Sheet 1: Column Definitions (NEW - put first for reference)
        col_defs = get_column_definitions()
        df_defs = pd.DataFrame(col_defs)
        df_defs.to_excel(writer, sheet_name='📖 Column Definitions', index=False)
        print(f"  📖 Column Definitions: {len(col_defs)} columns documented")
        
        # Sheet 2: By Scholar
        if matches:
            df = pd.DataFrame(matches)
            df = df.sort_values(['scholar_name', 'influence_score', 'year'], ascending=[True, False, False])
            sanitize_dataframe(df)
            df[main_columns].to_excel(writer, sheet_name='By Scholar', index=False)
            print(f"  📋 By Scholar: {len(df)} rows")
        
        # Sheet 3: High Influence Opinions (score >= 10)
        high_influence = [m for m in matches if m['influence_score'] >= 10]
        if high_influence:
            df_high = pd.DataFrame(high_influence)
            df_high = df_high.sort_values('influence_score', ascending=False)
            sanitize_dataframe(df_high)
            df_high[main_columns].to_excel(writer, sheet_name='High Influence Opinions', index=False)
            print(f"  ⭐ High Influence Opinions: {len(high_influence)} rows")
        
        # Sheet 4: High Confidence (confidence_level = VERY HIGH or HIGH)
        high_confidence = [m for m in matches if m['confidence_level'] in ['VERY HIGH', 'HIGH']]
        if high_confidence:
            df_conf = pd.DataFrame(high_confidence)
            df_conf = df_conf.sort_values(['confidence_score', 'influence_score'], ascending=[False, False])
            sanitize_dataframe(df_conf)
            df_conf[main_columns].to_excel(writer, sheet_name='High Confidence Citations', index=False)
            print(f"  🎯 High Confidence Citations: {len(high_confidence)} rows")
        
        # Sheet 5: Needs Review (LOW confidence)
        needs_review = [m for m in matches if m['confidence_level'] == 'LOW']
        if needs_review:
            df_review = pd.DataFrame(needs_review)
            df_review = df_review.sort_values('confidence_score', ascending=True)
            sanitize_dataframe(df_review)
            df_review[main_columns].to_excel(writer, sheet_name='⚠️ Needs Review', index=False)
            print(f"  ⚠️ Needs Review (Low Confidence): {len(needs_review)} rows")
        
        # Sheet 6: Brief→Opinion Influence
        case_scholar = defaultdict(lambda: {'in_brief': False, 'in_opinion': False, 
                                            'opinion_score': 0, 'opinion_level': '', 'opinion_desc': '', 
                                            'year': '', 'confidence': '', 'brief_count': 0, 'opinion_count': 0,
                                            'brief_colors': set()})
        for m in matches:
            key = (m['case_name'], m['scholar_name'])
            case_scholar[key]['year'] = m['year']
            case_scholar[key]['confidence'] = m['confidence_level']
            if m['doc_type'] == 'Opinion':
                case_scholar[key]['in_opinion'] = True
                case_scholar[key]['opinion_count'] += 1
                if m['influence_score'] > case_scholar[key]['opinion_score']:
                    case_scholar[key]['opinion_score'] = m['influence_score']
                    case_scholar[key]['opinion_level'] = m['influence_level']
                    case_scholar[key]['opinion_desc'] = m['influence_description']
            else:
                case_scholar[key]['in_brief'] = True
                case_scholar[key]['brief_count'] += 1
                case_scholar[key]['brief_colors'].add(m['brief_color'])
        
        influence_rows = []
        for (case, scholar), data in case_scholar.items():
            if data['in_opinion']:
                if data['in_brief']:
                    analysis = "⭐ Scholar cited in BOTH brief AND opinion - potential influence!"
                    if data['brief_count'] > 1:
                        analysis += f" (in {data['brief_count']} briefs)"
                else:
                    analysis = "Court cited scholar directly (not from brief)"
                
                influence_rows.append({
                    'Case': case,
                    'Year': data['year'],
                    'Scholar': scholar,
                    'Appeared in Brief': 'Yes' if data['in_brief'] else 'No',
                    'Brief Count': data['brief_count'],
                    'Brief Colors': ', '.join(sorted(data['brief_colors'])) if data['brief_colors'] else 'N/A',
                    'Appeared in Opinion': 'Yes',
                    'Opinion Count': data['opinion_count'],
                    'Opinion Influence Score': data['opinion_score'],
                    'Opinion Influence Level': data['opinion_level'],
                    'Confidence': data['confidence'],
                    'How Opinion Used Scholar': data['opinion_desc'],
                    'Analysis': analysis
                })
        
        if influence_rows:
            df_inf = pd.DataFrame(influence_rows)
            df_inf = df_inf.sort_values('Opinion Influence Score', ascending=False)
            sanitize_dataframe(df_inf)
            df_inf.to_excel(writer, sheet_name='Brief→Opinion Influence', index=False)
            print(f"  🔗 Brief→Opinion Influence: {len(influence_rows)} rows")
        
        # Sheet 7: Scholar Impact Summary
        if matches:
            summary = defaultdict(lambda: {
                'citations': 0, 'cases': set(), 'years': set(), 'total_score': 0,
                'high_influence': 0, 'opinions': 0, 'briefs': 0, 'org': '', 'disc': '',
                'contact_id': '', 'high_confidence': 0, 'low_confidence': 0,
                'avg_confidence': 0, 'confidence_scores': [],
                'total_engaged': None, 'total_engagements_last_5_fy': None,
                'primary_affiliation_record_type': ''
            })
            
            for m in matches:
                name = m['scholar_name']
                summary[name]['citations'] += 1
                summary[name]['cases'].add(m['case_name'])
                if m['year'] != 'Unknown':
                    summary[name]['years'].add(str(m['year']))
                summary[name]['total_score'] += m['influence_score']
                summary[name]['org'] = m['organization']
                summary[name]['disc'] = m['discipline']
                summary[name]['contact_id'] = m['contact_id']
                summary[name]['confidence_scores'].append(m['confidence_score'])
                summary[name]['total_engaged'] = m['total_engaged']
                summary[name]['total_engagements_last_5_fy'] = m['total_engagements_last_5_fy']
                summary[name]['primary_affiliation_record_type'] = m['primary_affiliation_record_type']
                
                if m['influence_score'] >= 8:
                    summary[name]['high_influence'] += 1
                if m['doc_type'] == 'Opinion':
                    summary[name]['opinions'] += 1
                else:
                    summary[name]['briefs'] += 1
                if m['confidence_level'] in ['VERY HIGH', 'HIGH']:
                    summary[name]['high_confidence'] += 1
                elif m['confidence_level'] == 'LOW':
                    summary[name]['low_confidence'] += 1
            
            rows = []
            for name, data in summary.items():
                avg_conf = sum(data['confidence_scores']) / len(data['confidence_scores']) if data['confidence_scores'] else 0
                rows.append({
                    'Scholar Name': name,
                    'Contact ID': data['contact_id'],
                    'Total Citations': data['citations'],
                    'Unique Cases': len(data['cases']),
                    'Years Active': ', '.join(sorted(data['years'])),
                    'Average Influence Score': round(data['total_score'] / data['citations'], 1),
                    'Average Confidence Score': round(avg_conf, 1),
                    'High Confidence Citations': data['high_confidence'],
                    'Low Confidence Citations': data['low_confidence'],
                    'High Influence Citations': data['high_influence'],
                    'Cited in Opinions': data['opinions'],
                    'Cited in Briefs': data['briefs'],
                    'Organization': data['org'],
                    'Discipline': data['disc'],
                    'Total Engaged (All-Time)': data['total_engaged'],
                    'Engagements Last 5 FY': data['total_engagements_last_5_fy'],
                    'Primary Affiliation Record Type': data['primary_affiliation_record_type']
                })
            
            df_sum = pd.DataFrame(rows)
            df_sum = df_sum.sort_values('Average Influence Score', ascending=False)
            sanitize_dataframe(df_sum)
            df_sum.to_excel(writer, sheet_name='Scholar Impact Summary', index=False)
            print(f"  📊 Scholar Impact Summary: {len(rows)} scholars")
        
        # Sheet 8: By Year
        if matches:
            df_year = pd.DataFrame(matches)
            df_year = df_year.sort_values(['year', 'influence_score'], ascending=[False, False])
            sanitize_dataframe(df_year)
            df_year[main_columns].to_excel(writer, sheet_name='By Year', index=False)
        
        # Sheet 9: Confidence Summary
        if matches:
            conf_summary = {
                'VERY HIGH': sum(1 for m in matches if m['confidence_level'] == 'VERY HIGH'),
                'HIGH': sum(1 for m in matches if m['confidence_level'] == 'HIGH'),
                'MEDIUM': sum(1 for m in matches if m['confidence_level'] == 'MEDIUM'),
                'LOW': sum(1 for m in matches if m['confidence_level'] == 'LOW'),
            }
            
            conf_data = [
                {'Confidence Level': 'VERY HIGH', 'Count': conf_summary['VERY HIGH'], 
                 'Percentage': f"{100*conf_summary['VERY HIGH']/len(matches):.1f}%",
                 'Description': 'Near certain - strong legal citation patterns'},
                {'Confidence Level': 'HIGH', 'Count': conf_summary['HIGH'],
                 'Percentage': f"{100*conf_summary['HIGH']/len(matches):.1f}%",
                 'Description': 'Strong indicators - likely valid academic citation'},
                {'Confidence Level': 'MEDIUM', 'Count': conf_summary['MEDIUM'],
                 'Percentage': f"{100*conf_summary['MEDIUM']/len(matches):.1f}%",
                 'Description': 'Probable match - some ambiguity'},
                {'Confidence Level': 'LOW', 'Count': conf_summary['LOW'],
                 'Percentage': f"{100*conf_summary['LOW']/len(matches):.1f}%",
                 'Description': 'Possible match - needs manual review'},
            ]
            pd.DataFrame(conf_data).to_excel(writer, sheet_name='Confidence Summary', index=False)
            print(f"  📈 Confidence Summary")
        
        # Sheet 10: Executive Summary
        if matches:
            unique_scholars = len(set(m['scholar_name'] for m in matches))
            unique_cases = len(set(m['case_name'] for m in matches))
            opinions = sum(1 for m in matches if m['doc_type'] == 'Opinion')
            briefs = sum(1 for m in matches if m['doc_type'] != 'Opinion')
            high = sum(1 for m in matches if m['influence_score'] >= 8)
            high_conf = sum(1 for m in matches if m['confidence_level'] in ['VERY HIGH', 'HIGH'])
            both = sum(1 for r in influence_rows if r['Appeared in Brief'] == 'Yes') if influence_rows else 0
            
            # Brief color breakdown
            petitioner_briefs = sum(1 for m in matches if 'Petitioner' in str(m.get('brief_color', '')))
            respondent_briefs = sum(1 for m in matches if 'Respondent' in str(m.get('brief_color', '')))
            neither_briefs = sum(1 for m in matches if 'Neither' in str(m.get('brief_color', '')))
            
            exec_data = [
                {'Metric': 'Total Scholar Citations Found', 'Count': len(matches)},
                {'Metric': 'Unique Scholars Cited', 'Count': unique_scholars},
                {'Metric': 'Unique Cases with Citations', 'Count': unique_cases},
                {'Metric': 'Citations in Court Opinions', 'Count': opinions},
                {'Metric': 'Citations in Amicus Briefs', 'Count': briefs},
                {'Metric': '  - Supporting Petitioner (Blue)', 'Count': petitioner_briefs},
                {'Metric': '  - Supporting Respondent (Red)', 'Count': respondent_briefs},
                {'Metric': '  - Supporting Neither (Green)', 'Count': neither_briefs},
                {'Metric': 'High Influence Citations (8+)', 'Count': high},
                {'Metric': 'High Confidence Citations', 'Count': high_conf},
                {'Metric': 'Scholars in Both Brief & Opinion', 'Count': both},
            ]
            pd.DataFrame(exec_data).to_excel(writer, sheet_name='Executive Summary', index=False)
            print(f"  📈 Executive Summary")
        
        # Sheet 11: Full Details
        if matches:
            df_full = pd.DataFrame(matches)
            df_full = df_full.sort_values(['scholar_name', 'influence_score'], ascending=[True, False])
            full_columns = main_columns + ['extended_context', 'file']
            sanitize_dataframe(df_full)
            df_full[full_columns].to_excel(writer, sheet_name='Full Details', index=False)
    
    print(f"\n✓ Report saved: {output_file}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    args = parse_args()
    
    # Parse year filter
    if args.all_years:
        filter_years = None
        year_display = "ALL YEARS"
    else:
        filter_years = parse_years(args.years)
        year_display = f"{min(filter_years)}-{max(filter_years)}" if filter_years else "ALL"
    
    print("=" * 70)
    print("IHS SCHOLAR CITATION FINDER - STRICT VERSION (Enhanced v3)")
    print("Only finds actual academic citations (not signers/authors)")
    print("Now with CONFIDENCE ratings, engagement data, and brief colors")
    print("=" * 70)
    print(f"Contacts: {CONTACTS_FILE}")
    print(f"PDFs: {AMICUS_FOLDER}")
    print(f"Year filter: {year_display}")
    print(f"File limit: {args.limit if args.limit else 'None (all files)'}")
    print(f"Min citation score: {MIN_CITATION_SCORE}")
    print("=" * 70)
    
    scholars = load_scholars(CONTACTS_FILE)
    matches = process_all_files(AMICUS_FOLDER, scholars, filter_years, args.limit)
    
    if matches:
        create_report(matches)
    else:
        print("\nNo citations found!")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if matches:
        print(f"✓ Citations found: {len(matches)}")
        print(f"✓ Unique scholars: {len(set(m['scholar_name'] for m in matches))}")
        print(f"✓ Unique cases: {len(set(m['case_name'] for m in matches))}")
        
        # Confidence breakdown
        conf_counts = defaultdict(int)
        for m in matches:
            conf_counts[m['confidence_level']] += 1
        print(f"\n📊 Confidence Breakdown:")
        for level in ['VERY HIGH', 'HIGH', 'MEDIUM', 'LOW']:
            if conf_counts[level] > 0:
                pct = 100 * conf_counts[level] / len(matches)
                print(f"   {level}: {conf_counts[level]} ({pct:.1f}%)")
        
        # Brief color breakdown
        color_counts = defaultdict(int)
        for m in matches:
            if 'Petitioner' in str(m.get('brief_color', '')):
                color_counts['Petitioner'] += 1
            elif 'Respondent' in str(m.get('brief_color', '')):
                color_counts['Respondent'] += 1
            elif 'Neither' in str(m.get('brief_color', '')):
                color_counts['Neither'] += 1
            elif m.get('doc_type') == 'Opinion':
                color_counts['Opinion'] += 1
            else:
                color_counts['Unknown'] += 1
        
        print(f"\n📊 Brief Color Breakdown:")
        for color in ['Petitioner', 'Respondent', 'Neither', 'Opinion', 'Unknown']:
            if color_counts[color] > 0:
                pct = 100 * color_counts[color] / len(matches)
                emoji = {'Petitioner': '🔵', 'Respondent': '🔴', 'Neither': '🟢', 'Opinion': '⚖️', 'Unknown': '⚪'}
                print(f"   {emoji.get(color, '')} {color}: {color_counts[color]} ({pct:.1f}%)")
                
    print("=" * 70)


if __name__ == '__main__':
    main()