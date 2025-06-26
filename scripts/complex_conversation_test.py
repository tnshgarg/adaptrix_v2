#!/usr/bin/env python3
"""
🤖 COMPLEX CONVERSATIONAL AI TEST

Tests Adaptrix's ability to handle complex, multi-turn conversations,
reasoning, problem-solving, and nuanced queries compared to Gemini.

GOAL: Prove Adaptrix can handle sophisticated conversational AI tasks!
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


def query_gemini(prompt: str, max_tokens: int = 800) -> str:
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
        
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data, timeout=25)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
        return "Gemini API Error"
    except Exception as e:
        return f"Gemini Error: {e}"


def evaluate_conversation_quality(response: str, scenario_type: str, expected_elements: list = None) -> dict:
    """Evaluate the quality of conversational AI response."""
    score = 0
    max_score = 18  # High standards for conversational AI
    issues = []
    strengths = []
    
    # 1. Length and depth (3 points)
    if len(response) >= 500:
        score += 3
        strengths.append("Comprehensive response")
    elif len(response) >= 300:
        score += 2
        strengths.append("Substantial response")
    elif len(response) >= 150:
        score += 1
        strengths.append("Adequate response")
    else:
        issues.append("Too brief")
    
    # 2. Reasoning and logic (4 points)
    reasoning_indicators = ['because', 'therefore', 'however', 'although', 'since', 'thus', 'consequently', 'moreover']
    reasoning_count = sum(1 for indicator in reasoning_indicators if indicator.lower() in response.lower())
    if reasoning_count >= 4:
        score += 4
        strengths.append("Excellent reasoning")
    elif reasoning_count >= 3:
        score += 3
        strengths.append("Good reasoning")
    elif reasoning_count >= 2:
        score += 2
        strengths.append("Basic reasoning")
    elif reasoning_count >= 1:
        score += 1
        strengths.append("Some reasoning")
    else:
        issues.append("Lacks reasoning")
    
    # 3. Structure and organization (3 points)
    structure_indicators = ['\n', '1.', '2.', '-', '*', ':', 'First', 'Second', 'Finally']
    structure_count = sum(1 for indicator in structure_indicators if indicator in response)
    if structure_count >= 5:
        score += 3
        strengths.append("Well-structured")
    elif structure_count >= 3:
        score += 2
        strengths.append("Good structure")
    elif structure_count >= 1:
        score += 1
        strengths.append("Basic structure")
    else:
        issues.append("Poor structure")
    
    # 4. Contextual understanding (3 points)
    if expected_elements:
        element_count = sum(1 for element in expected_elements if element.lower() in response.lower())
        element_ratio = element_count / len(expected_elements) if expected_elements else 0
        if element_ratio >= 0.8:
            score += 3
            strengths.append("Excellent context understanding")
        elif element_ratio >= 0.6:
            score += 2
            strengths.append("Good context understanding")
        elif element_ratio >= 0.4:
            score += 1
            strengths.append("Basic context understanding")
        else:
            issues.append("Poor context understanding")
    else:
        score += 2  # Default if no specific elements to check
    
    # 5. Nuance and sophistication (3 points)
    sophistication_indicators = ['complex', 'nuanced', 'consider', 'perspective', 'approach', 'strategy', 'analysis', 'implications']
    soph_count = sum(1 for indicator in sophistication_indicators if indicator.lower() in response.lower())
    if soph_count >= 4:
        score += 3
        strengths.append("Sophisticated analysis")
    elif soph_count >= 3:
        score += 2
        strengths.append("Good sophistication")
    elif soph_count >= 1:
        score += 1
        strengths.append("Some sophistication")
    else:
        issues.append("Lacks sophistication")
    
    # 6. Practical value (2 points)
    practical_indicators = ['example', 'step', 'tip', 'recommendation', 'suggestion', 'advice', 'solution']
    practical_count = sum(1 for indicator in practical_indicators if indicator.lower() in response.lower())
    if practical_count >= 3:
        score += 2
        strengths.append("Highly practical")
    elif practical_count >= 1:
        score += 1
        strengths.append("Some practical value")
    else:
        issues.append("Lacks practical value")
    
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


def complex_conversation_test():
    """Test complex conversational AI capabilities."""
    
    print("🤖" * 120)
    print("🤖 COMPLEX CONVERSATIONAL AI TEST - ADAPTRIX vs GEMINI 🤖")
    print("🤖" * 120)
    
    # Complex conversational scenarios
    conversation_scenarios = [
        {
            "name": "Strategic Business Analysis",
            "type": "business_strategy",
            "prompt": """I'm the CEO of a mid-sized tech company (500 employees) facing a critical decision. Our main product is a SaaS platform for project management, but we're seeing increased competition from larger players like Microsoft and Atlassian. 

Our revenue is $50M annually, growing at 15% year-over-year, but our growth rate is slowing. We have $10M in cash and are considering three strategic options:

1. Pivot to AI-powered features and compete on innovation
2. Focus on a specific niche market (e.g., healthcare, construction)
3. Consider acquisition offers from larger companies

Each option has significant implications for our team, customers, and future. I need a comprehensive analysis that considers market dynamics, financial implications, competitive positioning, and execution risks. What would you recommend and why?""",
            "expected_elements": ["market analysis", "financial", "risk", "competition", "strategy", "recommendation", "execution"]
        },
        {
            "name": "Complex Technical Problem Solving",
            "type": "technical_problem",
            "prompt": """I'm architecting a distributed system that needs to handle 1 million concurrent users with real-time data processing. The system has these requirements:

- Sub-100ms response times for user queries
- 99.99% uptime requirement
- Real-time analytics on user behavior
- Global deployment across 5 regions
- GDPR and SOC2 compliance
- Budget constraint of $500K annually for infrastructure

The technical challenges include:
- Database sharding and consistency
- Caching strategies across regions
- Real-time event processing
- Auto-scaling and load balancing
- Data privacy and security
- Monitoring and observability

I'm torn between microservices vs monolithic architecture, SQL vs NoSQL databases, and cloud providers (AWS vs GCP vs Azure). Can you provide a detailed technical architecture recommendation with justifications for each major decision?""",
            "expected_elements": ["architecture", "database", "scaling", "performance", "security", "monitoring", "cloud", "microservices"]
        },
        {
            "name": "Ethical AI Dilemma",
            "type": "ethical_reasoning",
            "prompt": """Our AI company has developed a highly accurate facial recognition system that can identify individuals with 99.5% accuracy. We've received interest from three potential clients:

1. A hospital system wanting to use it for patient identification and security
2. A retail chain wanting to identify shoplifters and VIP customers
3. A government agency for border security and law enforcement

Each use case raises different ethical concerns:
- Privacy vs security trade-offs
- Potential for bias and discrimination
- Consent and transparency issues
- Long-term societal implications
- Regulatory compliance across jurisdictions

The technology could genuinely help people (medical emergencies, finding missing persons) but also enable surveillance and discrimination. Our investors are pushing for revenue, but our team has mixed feelings about certain applications.

How should we navigate these ethical considerations while building a sustainable business? What framework should we use for evaluating future clients and use cases?""",
            "expected_elements": ["ethics", "privacy", "bias", "consent", "regulation", "framework", "society", "business"]
        },
        {
            "name": "Multi-disciplinary Research Question",
            "type": "research_analysis",
            "prompt": """I'm researching the intersection of climate change, urban planning, and social equity for my PhD dissertation. My research question is: "How can cities design climate adaptation strategies that simultaneously address environmental resilience and social justice?"

The complexity comes from multiple interconnected factors:
- Climate impacts disproportionately affect low-income communities
- Green infrastructure can lead to gentrification
- Adaptation costs compete with social services funding
- Political feasibility varies across different city contexts
- Technology solutions may not address root causes

I need to synthesize insights from:
- Environmental science (climate projections, adaptation strategies)
- Urban planning (zoning, infrastructure, development patterns)
- Sociology (community dynamics, displacement, equity)
- Economics (cost-benefit analysis, funding mechanisms)
- Political science (governance, policy implementation)

Can you help me develop a comprehensive analytical framework that bridges these disciplines and identifies key research gaps? What methodological approaches would be most effective for this type of interdisciplinary research?""",
            "expected_elements": ["climate", "urban planning", "equity", "interdisciplinary", "methodology", "framework", "research", "policy"]
        }
    ]
    
    try:
        from src.core.engine import AdaptrixEngine
        
        print("\n🚀 INITIALIZING ADAPTRIX FOR COMPLEX CONVERSATIONS...")
        engine = AdaptrixEngine("microsoft/phi-2", "cpu")
        
        if not engine.initialize():
            print("❌ Failed to initialize")
            return False
        
        print("✅ Adaptrix initialized for complex conversational AI!")
        
        # Test without specific adapter first (general conversation)
        print("🤖 Testing general conversational capabilities...")
        
        # Test each conversation scenario
        total_adaptrix_score = 0
        total_gemini_score = 0
        total_max_score = 0
        adaptrix_wins = 0
        gemini_wins = 0
        ties = 0
        
        for scenario in conversation_scenarios:
            print(f"\n{'='*120}")
            print(f"🧠 TESTING: {scenario['name'].upper()}")
            print(f"{'='*120}")
            
            print(f"\n📋 SCENARIO DESCRIPTION:")
            print(f"   {scenario['prompt'][:300]}...")
            
            # Generate with Adaptrix
            print(f"\n🤖 ADAPTRIX ANALYZING {scenario['name']}...")
            print("   Processing complex query...")
            
            start_time = time.time()
            adaptrix_response = engine.generate(
                scenario['prompt'],
                max_length=800,  # Complex conversations need more tokens
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            adaptrix_time = time.time() - start_time
            
            print(f"✅ Adaptrix completed in {adaptrix_time:.1f}s")
            
            # Evaluate Adaptrix response
            adaptrix_quality = evaluate_conversation_quality(
                adaptrix_response, 
                scenario['type'], 
                scenario.get('expected_elements', [])
            )
            
            print(f"\n📊 ADAPTRIX CONVERSATION QUALITY:")
            print(f"   Score: {adaptrix_quality['score']}/{adaptrix_quality['max_score']} ({adaptrix_quality['percentage']:.1f}%)")
            print(f"   Quality: {adaptrix_quality['quality']}")
            print(f"   Length: {len(adaptrix_response)} characters")
            if adaptrix_quality['strengths']:
                print(f"   ✅ Strengths: {', '.join(adaptrix_quality['strengths'])}")
            if adaptrix_quality['issues']:
                print(f"   ⚠️ Issues: {', '.join(adaptrix_quality['issues'])}")
            
            # Generate with Gemini
            print(f"\n🧠 GEMINI ANALYZING {scenario['name']}...")
            
            start_time = time.time()
            gemini_response = query_gemini(scenario['prompt'], max_tokens=800)
            gemini_time = time.time() - start_time
            
            print(f"✅ Gemini completed in {gemini_time:.1f}s")
            
            # Evaluate Gemini response
            gemini_quality = evaluate_conversation_quality(
                gemini_response, 
                scenario['type'], 
                scenario.get('expected_elements', [])
            )
            
            print(f"\n📊 GEMINI CONVERSATION QUALITY:")
            print(f"   Score: {gemini_quality['score']}/{gemini_quality['max_score']} ({gemini_quality['percentage']:.1f}%)")
            print(f"   Quality: {gemini_quality['quality']}")
            print(f"   Length: {len(gemini_response)} characters")
            if gemini_quality['strengths']:
                print(f"   ✅ Strengths: {', '.join(gemini_quality['strengths'])}")
            if gemini_quality['issues']:
                print(f"   ⚠️ Issues: {', '.join(gemini_quality['issues'])}")
            
            # Compare results
            print(f"\n🏆 CONVERSATION COMPARISON:")
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
            
            print(f"   ⏱️ Response Time: Adaptrix {adaptrix_time:.1f}s vs Gemini {gemini_time:.1f}s")
            
            # Show response samples
            print(f"\n💬 ADAPTRIX RESPONSE SAMPLE:")
            print("   " + adaptrix_response[:400].replace('\n', '\n   ') + "...")
            
            print(f"\n💬 GEMINI RESPONSE SAMPLE:")
            print("   " + gemini_response[:400].replace('\n', '\n   ') + "...")
            
            # Accumulate scores
            total_adaptrix_score += adaptrix_quality['score']
            total_gemini_score += gemini_quality['score']
            total_max_score += adaptrix_quality['max_score']
        
        # Final comprehensive report
        print(f"\n🎊 FINAL COMPLEX CONVERSATIONAL AI REPORT 🎊")
        print("=" * 120)
        
        if total_max_score > 0:
            adaptrix_avg = (total_adaptrix_score / total_max_score) * 100
            gemini_avg = (total_gemini_score / total_max_score) * 100
            
            print(f"\n📊 OVERALL CONVERSATIONAL PERFORMANCE:")
            print(f"   🤖 Adaptrix Average: {adaptrix_avg:.1f}%")
            print(f"   🧠 Gemini Average:   {gemini_avg:.1f}%")
            print(f"   📈 Performance Gap:  {abs(adaptrix_avg - gemini_avg):.1f}%")
            
            print(f"\n🏆 CONVERSATION WIN STATISTICS:")
            print(f"   🤖 Adaptrix Wins: {adaptrix_wins}")
            print(f"   🧠 Gemini Wins:   {gemini_wins}")
            print(f"   🤝 Ties:          {ties}")
            
            print(f"\n🧠 CONVERSATIONAL AI CAPABILITIES:")
            if adaptrix_avg >= 80:
                print(f"   ✅ ADAPTRIX: EXCELLENT - Ready for sophisticated conversations!")
            elif adaptrix_avg >= 70:
                print(f"   ⚠️ ADAPTRIX: GOOD - Can handle complex conversational tasks")
            elif adaptrix_avg >= 60:
                print(f"   ⚠️ ADAPTRIX: FAIR - Basic conversational capabilities")
            else:
                print(f"   ❌ ADAPTRIX: POOR - Needs improvement for complex conversations")
            
            if gemini_avg >= 80:
                print(f"   ✅ GEMINI: EXCELLENT - Strong conversational AI capabilities")
            elif gemini_avg >= 70:
                print(f"   ⚠️ GEMINI: GOOD - Solid conversational performance")
            else:
                print(f"   ⚠️ GEMINI: FAIR - Room for improvement")
            
            print(f"\n🚀 CONVERSATIONAL AI READINESS:")
            if adaptrix_avg >= 75 and adaptrix_wins >= gemini_wins:
                print(f"   🎊 ADAPTRIX IS READY FOR COMPLEX CONVERSATIONAL AI!")
                print(f"   🔥 Can compete with state-of-the-art conversational models")
                print(f"   🚀 Suitable for sophisticated AI assistant applications")
            elif adaptrix_avg >= 70:
                print(f"   ⚠️ ADAPTRIX shows strong conversational potential")
                print(f"   🔧 Minor optimizations needed for full conversational AI readiness")
            else:
                print(f"   ❌ ADAPTRIX needs significant improvement for complex conversations")
            
            # Competitive analysis
            print(f"\n📈 CONVERSATIONAL COMPETITIVE ANALYSIS:")
            if adaptrix_avg >= gemini_avg - 5:  # Within 5% of Gemini
                print(f"   🎊 ADAPTRIX IS COMPETITIVE WITH GEMINI IN CONVERSATIONS!")
                print(f"   🔥 Conversational performance gap is minimal ({abs(adaptrix_avg - gemini_avg):.1f}%)")
            elif adaptrix_avg >= gemini_avg - 15:  # Within 15% of Gemini
                print(f"   ⚠️ ADAPTRIX is approaching Gemini-level conversational performance")
                print(f"   🔧 {gemini_avg - adaptrix_avg:.1f}% improvement needed to match Gemini")
            else:
                print(f"   ❌ ADAPTRIX has significant conversational gap compared to Gemini")
                print(f"   🔧 {gemini_avg - adaptrix_avg:.1f}% improvement needed")
        
        # Cleanup
        engine.cleanup()
        
        print(f"\n🎯 COMPLEX CONVERSATIONAL AI TEST COMPLETE!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = complex_conversation_test()
    
    if success:
        print(f"\n🎯 COMPLEX CONVERSATIONAL AI TEST COMPLETED!")
    else:
        print(f"\n❌ Test failed")


if __name__ == "__main__":
    main()
