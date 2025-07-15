#!/usr/bin/env python3
"""
🚀 COMPLEX PROJECT GENERATION TEST

Tests Adaptrix's ability to generate complete, complex Python projects
and compares with Gemini's capabilities.

GOAL: Prove Adaptrix can handle real-world development tasks!
"""

import sys
import os
import requests
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

GEMINI_API_KEY = "AIzaSyAA-4qYJmlNtzO6gR-L5-pSEWPfuSl_xEA"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def query_gemini(prompt: str, max_tokens: int = 1000) -> str:
    """Query Gemini for comparison."""
    try:
        headers = {'Content-Type': 'application/json'}
        data = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'maxOutputTokens': max_tokens,
                'temperature': 0.7,
                'topP': 0.9,
            }
        }
        
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
        return "Gemini API Error"
    except Exception as e:
        return f"Gemini Error: {e}"


def evaluate_project_quality(response: str, project_type: str) -> dict:
    """Evaluate the quality of a generated project."""
    score = 0
    max_score = 20  # Higher standards for complex projects
    issues = []
    strengths = []
    
    # 1. Length and completeness (4 points)
    if len(response) >= 2000:
        score += 4
        strengths.append("Comprehensive project")
    elif len(response) >= 1000:
        score += 3
        strengths.append("Substantial project")
    elif len(response) >= 500:
        score += 2
        strengths.append("Adequate project")
    else:
        score += 1
        issues.append("Too brief for complex project")
    
    # 2. Code structure and organization (4 points)
    structure_indicators = ['class ', 'def ', 'import ', '__init__', 'if __name__']
    structure_count = sum(1 for indicator in structure_indicators if indicator in response)
    if structure_count >= 4:
        score += 4
        strengths.append("Well-structured code")
    elif structure_count >= 3:
        score += 3
        strengths.append("Good code structure")
    elif structure_count >= 2:
        score += 2
        strengths.append("Basic code structure")
    else:
        score += 1
        issues.append("Poor code structure")
    
    # 3. Multiple files/modules (3 points)
    file_indicators = ['# File:', '# filename:', 'main.py', '.py', 'requirements.txt', 'README']
    file_count = sum(1 for indicator in file_indicators if indicator.lower() in response.lower())
    if file_count >= 3:
        score += 3
        strengths.append("Multi-file project")
    elif file_count >= 2:
        score += 2
        strengths.append("Multiple files")
    elif file_count >= 1:
        score += 1
        strengths.append("File organization")
    else:
        issues.append("Single file only")
    
    # 4. Documentation and comments (3 points)
    doc_indicators = ['"""', "'''", '# ', 'Args:', 'Returns:', 'Example:']
    doc_count = sum(1 for indicator in doc_indicators if indicator in response)
    if doc_count >= 5:
        score += 3
        strengths.append("Excellent documentation")
    elif doc_count >= 3:
        score += 2
        strengths.append("Good documentation")
    elif doc_count >= 1:
        score += 1
        strengths.append("Basic documentation")
    else:
        issues.append("Poor documentation")
    
    # 5. Error handling and best practices (3 points)
    best_practices = ['try:', 'except:', 'raise', 'logging', 'if __name__', 'main()']
    practices_count = sum(1 for practice in best_practices if practice in response)
    if practices_count >= 4:
        score += 3
        strengths.append("Excellent best practices")
    elif practices_count >= 2:
        score += 2
        strengths.append("Good best practices")
    elif practices_count >= 1:
        score += 1
        strengths.append("Basic best practices")
    else:
        issues.append("Poor best practices")
    
    # 6. Project-specific functionality (3 points)
    if project_type == "web_scraper":
        specific_indicators = ['requests', 'BeautifulSoup', 'urllib', 'selenium', 'scrape', 'parse']
    elif project_type == "data_analysis":
        specific_indicators = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'csv', 'dataframe']
    elif project_type == "api_server":
        specific_indicators = ['flask', 'fastapi', 'django', 'route', 'endpoint', 'server']
    elif project_type == "cli_tool":
        specific_indicators = ['argparse', 'click', 'sys.argv', 'command', 'parser', 'main']
    else:
        specific_indicators = []
    
    if specific_indicators:
        specific_count = sum(1 for indicator in specific_indicators if indicator.lower() in response.lower())
        if specific_count >= 3:
            score += 3
            strengths.append("Domain-specific expertise")
        elif specific_count >= 2:
            score += 2
            strengths.append("Good domain knowledge")
        elif specific_count >= 1:
            score += 1
            strengths.append("Basic domain knowledge")
        else:
            issues.append("Lacks domain expertise")
    
    percentage = (score / max_score) * 100
    
    if percentage >= 90:
        quality = "EXCEPTIONAL"
    elif percentage >= 80:
        quality = "EXCELLENT"
    elif percentage >= 70:
        quality = "GOOD"
    elif percentage >= 60:
        quality = "FAIR"
    else:
        quality = "POOR"
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage,
        'quality': quality,
        'strengths': strengths,
        'issues': issues
    }


def complex_project_test():
    """Test complex project generation capabilities."""
    
    print("🚀" * 120)
    print("🚀 COMPLEX PROJECT GENERATION TEST - ADAPTRIX vs GEMINI 🚀")
    print("🚀" * 120)
    
    # Complex project scenarios
    project_scenarios = [
        {
            "name": "Web Scraper",
            "type": "web_scraper",
            "prompt": """Create a complete Python web scraping project that can scrape product information from e-commerce websites. The project should include:

1. A main scraper class with configurable settings
2. Support for multiple websites with different structures
3. Data cleaning and validation
4. Export to CSV and JSON formats
5. Error handling and retry logic
6. Rate limiting and respectful scraping
7. Configuration file support
8. Command-line interface
9. Logging system
10. Requirements file and documentation

Make it production-ready with proper code organization, error handling, and documentation."""
        },
        {
            "name": "Data Analysis Tool",
            "type": "data_analysis", 
            "prompt": """Create a comprehensive Python data analysis project for analyzing sales data. The project should include:

1. Data loading from multiple sources (CSV, Excel, JSON)
2. Data cleaning and preprocessing pipeline
3. Statistical analysis and insights generation
4. Interactive visualizations and charts
5. Automated report generation
6. Data validation and quality checks
7. Performance metrics calculation
8. Export functionality for results
9. Configuration management
10. Complete documentation and examples

Structure it as a professional data science project with proper organization and best practices."""
        },
        {
            "name": "REST API Server",
            "type": "api_server",
            "prompt": """Create a complete REST API server project for a task management system. The project should include:

1. RESTful API endpoints for CRUD operations
2. User authentication and authorization
3. Database integration with ORM
4. Input validation and error handling
5. API documentation (OpenAPI/Swagger)
6. Rate limiting and security features
7. Testing suite with unit and integration tests
8. Docker configuration
9. Environment configuration management
10. Deployment scripts and documentation

Make it production-ready with proper architecture, security, and scalability considerations."""
        }
    ]
    
    try:
        from src.core.engine import AdaptrixEngine
        
        print("\n🚀 INITIALIZING ADAPTRIX FOR COMPLEX PROJECTS...")
        engine = AdaptrixEngine("microsoft/phi-2", "cpu")
        
        if not engine.initialize():
            print("❌ Failed to initialize")
            return False
        
        print("✅ Adaptrix initialized for complex project generation!")
        
        # Load code specialist
        if not engine.load_adapter("code_specialist"):
            print("❌ Failed to load code specialist")
            return False
        
        print("✅ Code specialist loaded!")
        
        # Test each project scenario
        total_adaptrix_score = 0
        total_gemini_score = 0
        total_max_score = 0
        adaptrix_wins = 0
        gemini_wins = 0
        ties = 0
        
        for scenario in project_scenarios:
            print(f"\n{'='*100}")
            print(f"🏗️ TESTING: {scenario['name'].upper()}")
            print(f"{'='*100}")
            
            print(f"\n📋 PROJECT REQUIREMENTS:")
            print(f"   {scenario['prompt'][:200]}...")
            
            # Generate with Adaptrix
            print(f"\n🤖 ADAPTRIX GENERATING {scenario['name']}...")
            print("   This may take a while for complex projects...")
            
            start_time = time.time()
            adaptrix_response = engine.generate(
                scenario['prompt'],
                max_length=1000,  # Large project needs more tokens
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            adaptrix_time = time.time() - start_time
            
            print(f"✅ Adaptrix completed in {adaptrix_time:.1f}s")
            
            # Evaluate Adaptrix response
            adaptrix_quality = evaluate_project_quality(adaptrix_response, scenario['type'])
            
            print(f"\n📊 ADAPTRIX PROJECT QUALITY:")
            print(f"   Score: {adaptrix_quality['score']}/{adaptrix_quality['max_score']} ({adaptrix_quality['percentage']:.1f}%)")
            print(f"   Quality: {adaptrix_quality['quality']}")
            print(f"   Length: {len(adaptrix_response)} characters")
            if adaptrix_quality['strengths']:
                print(f"   ✅ Strengths: {', '.join(adaptrix_quality['strengths'])}")
            if adaptrix_quality['issues']:
                print(f"   ⚠️ Issues: {', '.join(adaptrix_quality['issues'])}")
            
            # Generate with Gemini
            print(f"\n🧠 GEMINI GENERATING {scenario['name']}...")
            
            start_time = time.time()
            gemini_response = query_gemini(scenario['prompt'], max_tokens=1000)
            gemini_time = time.time() - start_time
            
            print(f"✅ Gemini completed in {gemini_time:.1f}s")
            
            # Evaluate Gemini response
            gemini_quality = evaluate_project_quality(gemini_response, scenario['type'])
            
            print(f"\n📊 GEMINI PROJECT QUALITY:")
            print(f"   Score: {gemini_quality['score']}/{gemini_quality['max_score']} ({gemini_quality['percentage']:.1f}%)")
            print(f"   Quality: {gemini_quality['quality']}")
            print(f"   Length: {len(gemini_response)} characters")
            if gemini_quality['strengths']:
                print(f"   ✅ Strengths: {', '.join(gemini_quality['strengths'])}")
            if gemini_quality['issues']:
                print(f"   ⚠️ Issues: {', '.join(gemini_quality['issues'])}")
            
            # Compare results
            print(f"\n🏆 COMPARISON RESULTS:")
            if adaptrix_quality['percentage'] > gemini_quality['percentage']:
                winner = "ADAPTRIX"
                margin = adaptrix_quality['percentage'] - gemini_quality['percentage']
                print(f"   🏆 WINNER: ADAPTRIX (+{margin:.1f}%)")
                adaptrix_wins += 1
            elif gemini_quality['percentage'] > adaptrix_quality['percentage']:
                winner = "GEMINI"
                margin = gemini_quality['percentage'] - adaptrix_quality['percentage']
                print(f"   🏆 WINNER: GEMINI (+{margin:.1f}%)")
                gemini_wins += 1
            else:
                winner = "TIE"
                print(f"   🤝 TIE: Both performed equally")
                ties += 1
            
            print(f"   ⏱️ Speed: Adaptrix {adaptrix_time:.1f}s vs Gemini {gemini_time:.1f}s")
            
            # Show code samples
            print(f"\n📝 ADAPTRIX CODE SAMPLE:")
            print("   " + adaptrix_response[:300].replace('\n', '\n   ') + "...")
            
            print(f"\n📝 GEMINI CODE SAMPLE:")
            print("   " + gemini_response[:300].replace('\n', '\n   ') + "...")
            
            # Accumulate scores
            total_adaptrix_score += adaptrix_quality['score']
            total_gemini_score += gemini_quality['score']
            total_max_score += adaptrix_quality['max_score']
        
        # Final comprehensive report
        print(f"\n🎊 FINAL COMPLEX PROJECT GENERATION REPORT 🎊")
        print("=" * 100)
        
        if total_max_score > 0:
            adaptrix_avg = (total_adaptrix_score / total_max_score) * 100
            gemini_avg = (total_gemini_score / total_max_score) * 100
            
            print(f"\n📊 OVERALL PERFORMANCE:")
            print(f"   🤖 Adaptrix Average: {adaptrix_avg:.1f}%")
            print(f"   🧠 Gemini Average:   {gemini_avg:.1f}%")
            print(f"   📈 Performance Gap:  {abs(adaptrix_avg - gemini_avg):.1f}%")
            
            print(f"\n🏆 WIN STATISTICS:")
            print(f"   🤖 Adaptrix Wins: {adaptrix_wins}")
            print(f"   🧠 Gemini Wins:   {gemini_wins}")
            print(f"   🤝 Ties:          {ties}")
            
            print(f"\n🎯 COMPLEX PROJECT CAPABILITIES:")
            if adaptrix_avg >= 80:
                print(f"   ✅ ADAPTRIX: EXCELLENT - Ready for complex development tasks!")
            elif adaptrix_avg >= 70:
                print(f"   ⚠️ ADAPTRIX: GOOD - Can handle most complex projects")
            elif adaptrix_avg >= 60:
                print(f"   ⚠️ ADAPTRIX: FAIR - Basic complex project capabilities")
            else:
                print(f"   ❌ ADAPTRIX: POOR - Needs improvement for complex projects")
            
            if gemini_avg >= 80:
                print(f"   ✅ GEMINI: EXCELLENT - Strong complex development capabilities")
            elif gemini_avg >= 70:
                print(f"   ⚠️ GEMINI: GOOD - Solid complex project generation")
            else:
                print(f"   ⚠️ GEMINI: FAIR - Room for improvement")
            
            print(f"\n🚀 PRODUCTION READINESS FOR COMPLEX PROJECTS:")
            if adaptrix_avg >= 75 and adaptrix_wins >= gemini_wins:
                print(f"   🎊 ADAPTRIX IS READY FOR COMPLEX DEVELOPMENT TASKS!")
                print(f"   🔥 Can compete with state-of-the-art models")
                print(f"   🚀 Suitable for professional development workflows")
            elif adaptrix_avg >= 70:
                print(f"   ⚠️ ADAPTRIX shows strong potential for complex projects")
                print(f"   🔧 Minor optimizations needed for full production readiness")
            else:
                print(f"   ❌ ADAPTRIX needs significant improvement for complex projects")
            
            # Competitive analysis
            print(f"\n📈 COMPETITIVE ANALYSIS:")
            if adaptrix_avg >= gemini_avg - 5:  # Within 5% of Gemini
                print(f"   🎊 ADAPTRIX IS COMPETITIVE WITH GEMINI!")
                print(f"   🔥 Performance gap is minimal ({abs(adaptrix_avg - gemini_avg):.1f}%)")
            elif adaptrix_avg >= gemini_avg - 15:  # Within 15% of Gemini
                print(f"   ⚠️ ADAPTRIX is approaching Gemini-level performance")
                print(f"   🔧 {gemini_avg - adaptrix_avg:.1f}% improvement needed to match Gemini")
            else:
                print(f"   ❌ ADAPTRIX has significant gap compared to Gemini")
                print(f"   🔧 {gemini_avg - adaptrix_avg:.1f}% improvement needed")
        
        # Cleanup
        engine.unload_adapter("code_specialist")
        engine.cleanup()
        
        print(f"\n🎯 COMPLEX PROJECT TEST COMPLETE!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = complex_project_test()
    
    if success:
        print(f"\n🎯 COMPLEX PROJECT GENERATION TEST COMPLETED!")
    else:
        print(f"\n❌ Test failed")


if __name__ == "__main__":
    main()
