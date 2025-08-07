#!/usr/bin/env python3
"""
AI-Powered Deep Research Orchestrator with IHS Enhancements
Complete executable script
"""

import json
import asyncio
from datetime import datetime
import os
from typing import Dict, List, Optional
from openai import OpenAI
import pandas as pd
from collections import defaultdict
import re
import sys
from dotenv import load_dotenv

# Import your scraper class (adjust path as needed)
# from donor_research_scraper import DonorResearchScraper

class AIDeepResearchOrchestrator:
    """
    AI-powered orchestrator that intelligently sequences searches across multiple sources
    Enhanced for IHS donor research following Beth Miller standard
    """
    
    def __init__(self, scraper, openai_api_key):
        self.scraper = scraper
        self.client = OpenAI(api_key=openai_api_key)
        self.model = "gpt-4"
        
        # Available search methods organized by data source
        self.available_searches = {
            "search_engines": {
                "google": {
                    "method": self.scraper.search_google,
                    "params": ["first", "last", "city", "state"],
                    "requires_key": True,
                    "description": "General Google search"
                },
                "bing": {
                    "method": self.scraper.search_bing,
                    "params": ["first", "last", "city", "state"],
                    "requires_key": True,
                    "description": "Bing search results"
                }
            },
            "people_data": {
                "apollo": {
                    "method": self.scraper.search_apollo,
                    "params": ["first", "last", "city", "state"],
                    "requires_key": True,
                    "description": "Apollo business contacts and employment"
                },
                "pdl": {
                    "method": self.scraper.search_pdl,
                    "params": ["first", "last", "city", "state"],
                    "requires_key": True,
                    "description": "People Data Labs comprehensive profiles"
                },
                "linkedin": {
                    "method": self.scraper.search_linkedin,
                    "params": ["first", "last", "city", "state"],
                    "requires_key": True,
                    "description": "LinkedIn professional profile"
                }
            },
            "financial": {
                "fec": {
                    "method": self.scraper.search_fec,
                    "params": ["first", "last", "city", "state"],
                    "requires_key": False,
                    "description": "Federal Election Commission political contributions"
                },
                "opensecrets": {
                    "method": self.scraper.search_opensecrets,
                    "params": ["first", "last"],
                    "requires_key": True,
                    "description": "OpenSecrets political donation summaries"
                },
                "sec": {
                    "method": self.scraper.search_sec,
                    "params": ["first", "last"],
                    "requires_key": False,
                    "description": "SEC insider trading and holdings"
                }
            },
            "property": {
                "property_appraiser": {
                    "method": self.scraper.search_property_appraiser,
                    "params": ["first", "last", "city", "state"],
                    "requires_key": False,
                    "description": "Property ownership records"
                }
            },
            "nonprofits": {
                "irs_990": {
                    "method": self.scraper.search_irs_990,
                    "params": ["first", "last"],
                    "requires_key": False,
                    "description": "IRS 990 nonprofit filings"
                }
            },
            # IHS-specific search categories
            "personal_network": {
                "spouse_partner": {
                    "method": self.scraper.search_spouse_partner,
                    "params": ["first", "last", "city", "state"],
                    "requires_key": True,
                    "description": "Deep search for spouse/partner - marriage records, business connections, own wealth"
                },
                "college_connections": {
                    "method": self.scraper.search_college_connections,
                    "params": ["first", "last", "college", "grad_year"],
                    "requires_key": True,
                    "description": "Find roommates, Greek life, clubs, classmates now in power positions"
                },
                "social_clubs": {
                    "method": self.scraper.search_social_clubs,
                    "params": ["first", "last", "city"],
                    "requires_key": True,
                    "description": "Country clubs, YPO/WPO, civic organizations"
                }
            },
            "wealth_verification": {
                "business_exits": {
                    "method": self.scraper.search_business_exits,
                    "params": ["first", "last", "company"],
                    "requires_key": True,
                    "description": "Business sales, IPOs, liquidity events with amounts"
                },
                "wealth_indicators": {
                    "method": self.scraper.search_wealth_indicators,
                    "params": ["first", "last", "city", "state"],
                    "requires_key": True,
                    "description": "Real estate, planes (FAA), boats, luxury assets"
                },
                "family_foundation": {
                    "method": self.scraper.search_family_foundation,
                    "params": ["first", "last", "spouse_first", "spouse_last"],
                    "requires_key": True,
                    "description": "Family foundation 990-PFs, giving patterns"
                }
            },
            "ihs_alignment": {
                "board_positions": {
                    "method": self.scraper.search_all_board_positions,
                    "params": ["first", "last"],
                    "requires_key": True,
                    "description": "ALL boards - corporate, nonprofit, advisory"
                },
                "academic_connections": {
                    "method": self.scraper.search_academic_freedom_orgs,
                    "params": ["first", "last"],
                    "requires_key": True,
                    "description": "FIRE, Heterodox Academy, ODC-type groups"
                },
                "liberty_network": {
                    "method": self.scraper.search_liberty_organizations,
                    "params": ["first", "last", "city"],
                    "requires_key": True,
                    "description": "Think tanks, classical liberal orgs"
                }
            },
            "giving_history": {
                "named_gifts": {
                    "method": self.scraper.search_named_gifts,
                    "params": ["first", "last", "spouse_first", "spouse_last"],
                    "requires_key": True,
                    "description": "Named buildings, programs, scholarships"
                }
            }
        }
        
    async def research_person_deep(self, person: Dict, max_searches: int = 10) -> Dict:
        """
        Conduct deep research on a person using AI to guide the search sequence
        Enhanced for IHS donor research
        
        Args:
            person: Dict with 'first', 'last', 'city', 'state' keys
            max_searches: Maximum number of searches to conduct
            
        Returns:
            Dict with findings and analysis
        """
        print(f"\n🔬 Starting AI-guided deep research for {person['first']} {person['last']}")
        print(f"   Location: {person.get('city', '')}, {person.get('state', '')}")
        print(f"   Max searches: {max_searches}")
        
        # Initialize research context
        context = {
            'person': person,
            'findings': [],
            'search_history': [],
            'key_facts': {},
            'start_time': datetime.now(),
            'spouse_info': None
        }
        
        # Conduct searches guided by AI
        searches_conducted = 0
        
        # For IHS research, always start with critical searches
        if max_searches >= 15:  # If doing deep research
            print("\n📋 Phase 1: IHS Critical Searches")
            critical_searches = ["spouse_partner", "business_exits", "family_foundation", "google"]
            for search_name in critical_searches:
                if searches_conducted >= max_searches:
                    break
                await self._execute_search_by_name(search_name, context)
                searches_conducted += 1
            
            # Extract spouse info if found
            self._extract_spouse_from_findings(context)
        
        # AI-guided searches for remaining budget
        while searches_conducted < max_searches:
            # AI decides next searches
            next_searches = await self.ai_decide_next_searches(context)
            
            if not next_searches:
                print("\n✅ AI determined research is complete")
                break
                
            # Execute the suggested searches
            for search in next_searches:
                if searches_conducted >= max_searches:
                    break
                    
                print(f"\n📍 Search {searches_conducted + 1}/{max_searches}")
                print(f"   AI selected: {search['name']} ({search['category']})")
                print(f"   Reason: {search.get('reason', 'Strategic choice')}")
                
                # Execute the search
                success = await self._execute_search(search, context)
                searches_conducted += 1
                
                if not success:
                    print(f"   ⚠️  Search failed, AI will adapt strategy")
        
        # Final synthesis by AI
        print("\n🤖 AI synthesizing findings...")
        synthesis = await self.ai_synthesize_research(context)
        
        # Calculate IHS probability scores
        ihs_scores = self._calculate_ihs_probability(context['findings'])
        
        # Prepare final report
        duration = (datetime.now() - context['start_time']).total_seconds()
        
        final_report = {
            'person': person,
            'total_findings': len(context['findings']),
            'searches_conducted': len(context['search_history']),
            'duration_seconds': duration,
            'synthesis': synthesis,
            'ihs_scores': ihs_scores,
            'key_facts': context['key_facts'],
            'search_history': context['search_history'],
            'findings': context['findings']
        }
        
        print(f"\n✅ Research complete!")
        print(f"   Total findings: {final_report['total_findings']}")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   IHS Probability: {ihs_scores['probability']}")
        
        return final_report
    
    async def ai_decide_next_searches(self, context):
        """AI decides which searches to run next based on IHS priorities"""
        
        # Get unused searches
        used_searches = {s['name'] for s in context['search_history']}
        available_unused = []
        
        # IHS Priority order - these go first
        ihs_priority_searches = [
            "spouse_partner",      # CRITICAL - family giving decisions
            "business_exits",      # CRITICAL - liquidity events
            "family_foundation",   # CRITICAL - giving vehicle
            "college_connections", # HIGH - warm intros
            "board_positions",     # HIGH - interests/influence
            "academic_connections",# HIGH - IHS alignment
            "wealth_indicators",   # HIGH - capacity verification
            "named_gifts",        # HIGH - giving history
            "liberty_network",    # MEDIUM - ideological fit
            "fec",                # MEDIUM - political alignment
            "social_clubs",       # MEDIUM - peer networks
        ]
        
        # First add IHS priority searches if unused
        for priority_search in ihs_priority_searches:
            for category, sources in self.available_searches.items():
                if priority_search in sources and priority_search not in used_searches:
                    if sources[priority_search]['requires_key'] and not self._has_required_keys(sources[priority_search]):
                        continue
                    
                    priority_level = "CRITICAL" if priority_search in ["spouse_partner", "business_exits", "family_foundation"] else "HIGH"
                    available_unused.append({
                        "category": category,
                        "name": priority_search,
                        "description": sources[priority_search]['description'],
                        "params": sources[priority_search]['params'],
                        "priority": priority_level
                    })
        
        # Then add other available searches
        for category, sources in self.available_searches.items():
            for name, config in sources.items():
                if name not in used_searches and name not in ihs_priority_searches:
                    if config['requires_key'] and not self._has_required_keys(config):
                        continue
                    
                    available_unused.append({
                        "category": category,
                        "name": name,
                        "description": config['description'],
                        "params": config['params'],
                        "priority": "MEDIUM"
                    })
        
        if not available_unused:
            return []
        
        # Calculate IHS research gaps
        gaps = self._identify_ihs_gaps(context)
        
        prompt = f"""
        Based on IHS donor research priorities (Beth Miller standard), decide next searches.
        
        Person: {json.dumps(context['person'], indent=2)}
        Current findings: {len(context['findings'])} sources
        
        Research gaps:
        - Spouse researched: {gaps['spouse_found']}
        - Liquidity events found: {gaps['liquidity_found']}
        - Board positions found: {gaps['boards_count']}
        - Academic freedom connections: {gaps['academic_freedom']}
        - Warm intro paths: {gaps['warm_intros']}
        
        Available searches not yet used:
        {json.dumps(available_unused, indent=2)}
        
        Select up to 3 searches prioritizing:
        1. CRITICAL searches (spouse, business exits, foundation)
        2. Information gaps for IHS probability matrix
        3. Warm introduction opportunities
        
        Return JSON:
        {{
            "searches": [
                {{
                    "category": "category_name",
                    "name": "search_name",
                    "reason": "specific reason needed",
                    "params": {{"param_name": "value"}}
                }}
            ]
        }}
        
        Return ONLY valid JSON.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            # Clean up response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
                
            result = json.loads(response_text.strip())
            return result.get('searches', [])
            
        except Exception as e:
            print(f"    ✗ AI decision error: {str(e)}")
            # Return first CRITICAL search if available
            critical = next((s for s in available_unused if s.get('priority') == 'CRITICAL'), None)
            if critical:
                return [{
                    "category": critical['category'],
                    "name": critical['name'],
                    "reason": "Critical IHS priority search",
                    "params": {p: context['person'].get(p, '') for p in critical['params']}
                }]
            return []
    
    def _identify_ihs_gaps(self, context):
        """Identify research gaps for IHS assessment"""
        
        findings_text = ' '.join([str(f) for f in context.get('findings', [])]).lower()
        
        return {
            'spouse_found': 'Yes' if any(term in findings_text for term in ['spouse', 'wife', 'husband', 'married to']) else 'No',
            'liquidity_found': 'Yes' if any(term in findings_text for term in ['sold company', 'acquisition', 'ipo', 'exit']) else 'No',
            'boards_count': findings_text.count('board'),
            'academic_freedom': 'Yes' if any(term in findings_text for term in ['fire', 'heterodox', 'free speech', 'academic freedom']) else 'No',
            'warm_intros': len([f for f in context['findings'] if 'classmate' in str(f) or 'colleague' in str(f) or 'board' in str(f)])
        }
    
    def _extract_spouse_from_findings(self, context):
        """Extract spouse information from findings"""
        
        for finding in context['findings']:
            text = (finding.get('snippet', '') + ' ' + finding.get('summary', '')).lower()
            
            # Look for spouse patterns
            patterns = [
                r'married to ([A-Z][a-z]+ [A-Z][a-z]+)',
                r'wife ([A-Z][a-z]+ [A-Z][a-z]+)',
                r'husband ([A-Z][a-z]+ [A-Z][a-z]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, finding.get('snippet', ''))
                if match:
                    spouse_name = match.group(1)
                    context['spouse_info'] = {
                        'full_name': spouse_name,
                        'first': spouse_name.split()[0],
                        'last': spouse_name.split()[-1]
                    }
                    context['person']['spouse_first'] = context['spouse_info']['first']
                    context['person']['spouse_last'] = context['spouse_info']['last']
                    print(f"   ✓ Identified spouse: {spouse_name}")
                    return
    
    def _calculate_ihs_probability(self, findings):
        """Calculate IHS giving probability scores"""
        
        all_text = ' '.join([str(f) for f in findings]).lower()
        
        scores = {
            'financial_capacity': 0,
            'philanthropic_inclination': 0,
            'ideological_alignment': 0,
            'existing_relationship': 0
        }
        
        # Financial Capacity
        if any(term in all_text for term in ['sold company', 'acquisition', 'billion', 'ipo']):
            scores['financial_capacity'] = 90
        elif any(term in all_text for term in ['million', 'founder', 'ceo']):
            scores['financial_capacity'] = 70
        else:
            scores['financial_capacity'] = 40
        
        # Philanthropic Inclination
        phil_count = sum(1 for f in findings if 'philanthrop' in str(f).lower() or 'donat' in str(f).lower())
        board_count = sum(1 for f in findings if 'board' in str(f).lower())
        scores['philanthropic_inclination'] = min(100, (phil_count * 15) + (board_count * 10) + 20)
        
        # Ideological Alignment
        alignment_count = sum(1 for term in ['free speech', 'academic freedom', 'liberty', 'classical liberal'] 
                             if term in all_text)
        scores['ideological_alignment'] = min(100, alignment_count * 25 + 20)
        
        # Overall
        overall = (scores['financial_capacity'] * 0.3 + 
                  scores['philanthropic_inclination'] * 0.3 + 
                  scores['ideological_alignment'] * 0.3 + 
                  scores['existing_relationship'] * 0.1)
        
        return {
            'scores': scores,
            'overall': overall,
            'probability': 'High' if overall >= 70 else 'Medium' if overall >= 40 else 'Low'
        }
    
    async def _execute_search_by_name(self, search_name, context):
        """Execute a search by name"""
        
        for category, sources in self.available_searches.items():
            if search_name in sources:
                search_config = {
                    'name': search_name,
                    'category': category,
                    'params': {p: context['person'].get(p, '') for p in sources[search_name]['params']}
                }
                await self._execute_search(search_config, context)
                return
    
    async def _execute_search(self, search_config, context):
        """
        Execute a single search based on AI recommendation
        
        Args:
            search_config: Dict with 'category', 'name', 'params'
            context: Research context
            
        Returns:
            bool: Success status
        """
        category = search_config['category']
        name = search_config['name']
        params = search_config.get('params', {})
        
        # Get the search configuration
        if category not in self.available_searches or name not in self.available_searches[category]:
            print(f"   ✗ Unknown search: {category}/{name}")
            return False
            
        search_info = self.available_searches[category][name]
        method = search_info['method']
        
        # Check for required API keys
        if search_info['requires_key'] and not self._has_required_keys(search_info):
            print(f"   ✗ Missing API key for {name}")
            return False
        
        # Execute the search
        try:
            print(f"   🔍 Executing {name} search...")
            results = await method(**params)
            
            if results:
                print(f"   ✓ Found {len(results)} results")
                context['findings'].extend(results)
                
                # Extract key facts
                self._extract_key_facts(results, context)
            else:
                print(f"   - No results found")
                
            # Record in search history
            context['search_history'].append({
                'name': name,
                'category': category,
                'timestamp': datetime.now().isoformat(),
                'results_count': len(results) if results else 0,
                'params': params
            })
            
            return True
            
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            context['search_history'].append({
                'name': name,
                'category': category,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'params': params
            })
            return False
    
    def _has_required_keys(self, search_info):
        """Check if required API keys are available"""
        # This is a simplified check - implement based on your scraper's key management
        return True
    
    def _extract_key_facts(self, results, context):
        """Extract and store key facts from search results"""
        for result in results:
            # Extract emails
            if 'email' in result:
                if 'emails' not in context['key_facts']:
                    context['key_facts']['emails'] = set()
                context['key_facts']['emails'].add(result['email'])
                
            # Extract phone numbers
            if 'phone' in result:
                if 'phones' not in context['key_facts']:
                    context['key_facts']['phones'] = set()
                context['key_facts']['phones'].add(result['phone'])
                
            # Extract addresses
            if 'address' in result:
                if 'addresses' not in context['key_facts']:
                    context['key_facts']['addresses'] = []
                context['key_facts']['addresses'].append(result['address'])
                
            # Extract employment
            if 'company' in result or 'employer' in result:
                if 'employment' not in context['key_facts']:
                    context['key_facts']['employment'] = []
                context['key_facts']['employment'].append(
                    result.get('company', result.get('employer', ''))
                )
    
    async def ai_synthesize_research(self, context):
        """
        Use AI to synthesize all findings into a comprehensive analysis
        Enhanced for IHS donor report format
        """
        # Prepare findings summary
        findings_by_source = defaultdict(list)
        for finding in context['findings']:
            source = finding.get('source', 'unknown')
            findings_by_source[source].append(finding)
        
        # Create structured summary for AI
        findings_summary = []
        for source, items in findings_by_source.items():
            findings_summary.append(f"\n{source.upper()} ({len(items)} results):")
            for item in items[:5]:  # First 5 from each source
                if 'summary' in item:
                    findings_summary.append(f"- {item['summary']}")
                elif 'snippet' in item:
                    findings_summary.append(f"- {item['snippet'][:200]}...")
        
        findings_text = "\n".join(findings_summary)
        
        # Calculate IHS scores
        ihs_scores = self._calculate_ihs_probability(context['findings'])
        
        prompt = f"""
        Create a comprehensive IHS donor research report following the Beth Miller format.
        
        Person: {json.dumps(context['person'], indent=2)}
        Total findings: {len(context['findings'])}
        
        Key findings by source:
        {findings_text}
        
        Generate THREE documents:
        
        1. EXECUTIVE SUMMARY (2 pages):
        - Biographical Overview (education, career, family including spouse)
        - Career Trajectory & Financial Capacity (emphasize liquidity events)
        - Philanthropic Interests (specific organizations and amounts)
        - Ideological Alignment & Networks (liberty/academic freedom connections)
        - Probability Matrix Assessment (High/Medium/Low with evidence):
          * Financial Capacity: {self._score_to_level(ihs_scores['scores']['financial_capacity'])}
          * Philanthropic Inclination: {self._score_to_level(ihs_scores['scores']['philanthropic_inclination'])}
          * Ideological Alignment with IHS: {self._score_to_level(ihs_scores['scores']['ideological_alignment'])}
          * Existing IHS Relationship: Low
          * Overall Giving Probability: {ihs_scores['probability']}
        - Recommended IHS Engagement Steps (4 specific actions)
        
        2. FULL NARRATIVE REPORT with sections:
        - Background & Career (include spouse details)
        - Financial Capacity & Wealth Indicators
        - Philanthropic Track Record
        - Ideological Profile and Networks
        - Strategic Considerations for IHS
        
        3. SUMMARY TABLE with all key facts
        
        Format with **bold headers** and organized sections.
        Every fact must have [[source]] notation.
        Focus on IHS mission alignment: intellectual talent development, academic programs, ideas to market.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating synthesis: {str(e)}"
    
    def _score_to_level(self, score):
        """Convert numeric score to High/Medium/Low"""
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        else:
            return "Low"
    
    def generate_report(self, research_results: Dict) -> str:
        """
        Generate a formatted HTML report from research results
        """
        person = research_results['person']
        
        html = f"""
        <html>
        <head>
            <title>IHS Donor Research Report - {person['first']} {person['last']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .section {{ margin: 30px 0; }}
                .finding {{ margin: 10px 0; padding: 10px; background: #fff; border-left: 3px solid #007bff; }}
                .key-fact {{ display: inline-block; margin: 5px; padding: 5px 10px; background: #e9ecef; border-radius: 3px; }}
                .search-history {{ font-size: 0.9em; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>IHS Donor Research Report</h1>
            <h2>{person['first']} {person['last']} - {person.get('city', '')}, {person.get('state', '')}</h2>
            
            <div class="summary">
                <h3>IHS Probability Assessment</h3>
                <p><strong>Overall Probability:</strong> {research_results['ihs_scores']['probability']}</p>
                <p><strong>Financial Capacity:</strong> {self._score_to_level(research_results['ihs_scores']['scores']['financial_capacity'])}</p>
                <p><strong>Philanthropic Inclination:</strong> {self._score_to_level(research_results['ihs_scores']['scores']['philanthropic_inclination'])}</p>
                <p><strong>Ideological Alignment:</strong> {self._score_to_level(research_results['ihs_scores']['scores']['ideological_alignment'])}</p>
                <p><strong>Total Findings:</strong> {research_results['total_findings']}</p>
                <p><strong>Searches Conducted:</strong> {research_results['searches_conducted']}</p>
                <p><strong>Research Duration:</strong> {research_results['duration_seconds']:.1f} seconds</p>
            </div>
            
            <div class="section">
                <h3>AI Synthesis</h3>
                <div style="white-space: pre-wrap;">{research_results['synthesis']}</div>
            </div>
            
            <div class="section">
                <h3>Search History</h3>
                <div class="search-history">
                    {"<br>".join([f"• {s['name']} ({s.get('results_count', 0)} results)" for s in research_results['search_history']])}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html


# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Main execution function"""
    
    # Load environment variables
    load_dotenv()
    
    # Check for required API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in .env file or environment")
        return
    
    # Initialize scraper (import your actual scraper class)
    try:
        from donor_research_scraper import DonorResearchScraper
        scraper = DonorResearchScraper()
    except ImportError:
        print("❌ Error: Could not import DonorResearchScraper")
        print("Please ensure donor_research_scraper.py is in the same directory")
        return
    
    # Initialize orchestrator
    orchestrator = AIDeepResearchOrchestrator(scraper, openai_api_key)
    
    # Get person to research from command line or use default
    if len(sys.argv) > 1:
        first_name = sys.argv[1]
        last_name = sys.argv[2] if len(sys.argv) > 2 else ""
        city = sys.argv[3] if len(sys.argv) > 3 else ""
        state = sys.argv[4] if len(sys.argv) > 4 else ""
    else:
        # Default person for testing
        print("\nNo person specified. Usage: python spider_st.py <first> <last> <city> <state>")
        print("Using default test person...")
        first_name = "John"
        last_name = "Smith"
        city = "New York"
        state = "NY"
    
    # Create person dict
    person = {
        "first": first_name,
        "last": last_name,
        "city": city,
        "state": state
    }
    
    # Run research
    print(f"\n{'='*60}")
    print(f"IHS DONOR RESEARCH SYSTEM")
    print(f"{'='*60}")
    
    try:
        # Conduct deep research (20 searches for IHS mode)
        results = await orchestrator.research_person_deep(person, max_searches=20)
        
        # Generate HTML report
        report_html = orchestrator.generate_report(results)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"ihs_report_{person['last']}_{person['first']}_{timestamp}.html"
        
        with open(report_filename, 'w') as f:
            f.write(report_html)
        
        print(f"\n📄 Report saved to: {report_filename}")
        
        # Also save raw JSON results
        json_filename = f"ihs_data_{person['last']}_{person['first']}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Raw data saved to: {json_filename}")
        
    except Exception as e:
        print(f"\n❌ Error during research: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
