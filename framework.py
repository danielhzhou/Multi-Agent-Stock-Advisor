"""
Financial AI Agent Framework

A framework for creating and managing financial AI agents using LangChain, LangGraph, and Tavily.
This framework enables two AI agents to collaborate on financial tasks:
1. Research Agent: Finds relevant financial information
2. Advisory Agent: Analyzes information and provides recommendations
"""

import os
from typing import Dict, List, Any, Tuple, Annotated, Literal, Optional
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from pydantic import BaseModel, Field

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()  # take environment variables from .env
except ImportError:
    print("python-dotenv not installed. Skipping .env file loading.")

# LangChain Imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.tavily_search import TavilySearchResults

# LangGraph Imports - Updated to use proper imports
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# OpenAI
from langchain_openai import ChatOpenAI

# Required Environment Variables
# OPENAI_API_KEY
# TAVILY_API_KEY

class AgentState(TypedDict):
    """State for agent interaction in the graph."""
    messages: Annotated[list, add_messages]  # Messages will be appended, not overwritten
    current_agent: Literal["research", "advisory", "user"]
    research_output: Optional[str]
    research_reasoning: Optional[str]
    advisory_reasoning: Optional[str]
    original_input: Optional[str]
    result: Optional[str]
    agent_turns: int

# Initialize LLM
def get_llm(model="gpt-4-turbo", temperature=0, max_tokens=None):
    """Initialize and return an OpenAI language model."""
    if not max_tokens:
        # Set reasonable defaults based on model
        if "gpt-4" in model:
            max_tokens = 4000  # Leave room for context
        else:
            max_tokens = 2000  # For smaller models
            
    return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

# ======= TOOLS DEFINITION =======

@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price for a given ticker symbol using Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        
        price = info.get('regularMarketPrice', 'N/A')
        change = info.get('regularMarketChange', 'N/A')
        change_percent = info.get('regularMarketChangePercent', 'N/A')
        
        if price != 'N/A':
            change_percent_formatted = f"{change_percent:.2f}%" if isinstance(change_percent, (int, float)) else change_percent
            return f"Current stock price for {ticker}: ${price} | Change: {change} ({change_percent_formatted})"
        
        return f"Could not retrieve data for ticker {ticker}. Please check if the ticker symbol is correct."
    except Exception as e:
        return f"Error retrieving stock price for {ticker}: {str(e)}"

@tool
def get_company_financials(ticker: str) -> str:
    """
    Get key financial metrics for a company using Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Key financial metrics
        metrics = {
            "Market Cap": info.get('marketCap', 'N/A'),
            "P/E Ratio": info.get('trailingPE', 'N/A'),
            "EPS (TTM)": info.get('trailingEps', 'N/A'),
            "Revenue (TTM)": info.get('totalRevenue', 'N/A'),
            "Profit Margin": info.get('profitMargins', 'N/A'),
            "52-Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52-Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
            "Analyst Rating": info.get('recommendationKey', 'N/A'),
            "Dividend Yield": info.get('dividendYield', 'N/A'),
            "Beta": info.get('beta', 'N/A')
        }
        
        results = [f"{key}: {value}" for key, value in metrics.items()]
        return f"Financial metrics for {ticker}:\n" + "\n".join(results)
    except Exception as e:
        return f"Error retrieving financial data for {ticker}: {str(e)}"

@tool
def get_economic_indicator(indicator: str, country: str = "US") -> str:
    """
    Get economic indicators with clear sourcing information.
    
    Args:
        indicator: Type of indicator (e.g., "inflation", "interest_rate", "unemployment", "gdp_growth")
        country: Country code (default: "US")
    """
    # Mapping of user-friendly terms to specific data sources 
    indicators_map = {
        "inflation": {
            "US": "https://fred.stlouisfed.org/series/CPIAUCSL",
            "url": "https://www.usinflationcalculator.com/inflation/current-inflation-rates/",
            "extraction": lambda soup: soup.select("table.tablepress tr")[1].select("td")[1].text
        },
        "interest_rate": {
            "US": "https://fred.stlouisfed.org/series/FEDFUNDS",
            "url": "https://www.federalreserve.gov/releases/h15/",
            "extraction": lambda soup: soup.select("table.statistics tr")[1].select("td")[1].text
        },
        "unemployment": {
            "US": "https://fred.stlouisfed.org/series/UNRATE",
            "url": "https://www.bls.gov/news.release/empsit.nr0.htm",
            "extraction": lambda soup: re.search(r"unemployment rate at (\d+\.\d+) percent", soup.text).group(1)
        }
    }
    
    # Try to fetch real-time data first if possible
    try:
        # Attempt to get live data for US indicators only
        if country == "US" and indicator.lower() in indicators_map:
            try:
                # For US, we'll make a best effort to scrape updated data
                import requests
                from bs4 import BeautifulSoup
                from datetime import datetime
                
                info = indicators_map[indicator.lower()]
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                
                if "url" in info:
                    response = requests.get(info["url"], headers=headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        try:
                            extracted = info["extraction"](soup)
                            # If we got something that looks like a percentage or number
                            if re.search(r'\d+\.?\d*%?', extracted):
                                return f"{indicator.title()} for {country}: {extracted} (Live data from {info['url']})"
                        except Exception:
                            # Continue to cached data
                            pass
            except Exception:
                # Silently fall through to cached data
                pass
        
        # Use cached data as fallback
        data = {
            "inflation": {
                "US": "8.3% (as of last reported period)",
                "UK": "6.2% (as of last reported period)",
                "EU": "5.5% (as of last reported period)"
            },
            "interest_rate": {
                "US": "5.25-5.50% (Federal Funds Rate)",
                "UK": "5.25% (Bank Rate)",
                "EU": "4.50% (ECB Main Refinancing Rate)"
            },
            "unemployment": {
                "US": "3.8% (as of last reported period)",
                "UK": "4.2% (as of last reported period)",
                "EU": "6.5% (as of last reported period)"
            },
            "gdp_growth": {
                "US": "2.1% annual rate (last quarter)",
                "UK": "0.6% annual rate (last quarter)",
                "EU": "0.8% annual rate (last quarter)"
            }
        }
        
        if indicator.lower() not in data:
            return f"Indicator '{indicator}' not found. Available indicators: {', '.join(data.keys())}"
        
        if country not in data[indicator.lower()]:
            return f"Data for {country} not available. Available countries: {', '.join(data[indicator.lower()].keys())}"
            
        # Clearly indicate this is cached data
        cached_result = f"{indicator.title()} for {country}: {data[indicator.lower()][country]}"
        cached_result += f"\n\nNote: This is cached data for demonstration purposes. For real-time data, please visit:"
        
        if country == "US" and indicator.lower() in indicators_map:
            cached_result += f"\n- {indicators_map[indicator.lower()]['US']}"
        elif indicator.lower() == "inflation":
            cached_result += "\n- https://www.usinflationcalculator.com/ (US)"
            cached_result += "\n- https://www.ons.gov.uk/ (UK)"
            cached_result += "\n- https://ec.europa.eu/eurostat (EU)"
        elif indicator.lower() == "interest_rate":
            cached_result += "\n- https://www.federalreserve.gov/ (US)"
            cached_result += "\n- https://www.bankofengland.co.uk/ (UK)"
            cached_result += "\n- https://www.ecb.europa.eu/ (EU)"
        
        return cached_result
        
    except Exception as e:
        return f"Error retrieving economic indicator data: {str(e)}"

@tool
def scrape_financial_news(search_term: str = "stock market", max_articles: int = 5) -> str:
    """
    Scrape recent financial news articles related to a specific search term.
    
    Args:
        search_term: Term to search for in financial news
        max_articles: Maximum number of articles to return
    """
    results = []
    error_messages = []
    
    # Try each method until we get some results
    try:
        # 1. First attempt: Use Tavily search
        try:
            from langchain.tools.tavily_search import TavilySearchResults
            tavily = TavilySearchResults(max_results=max_articles)
            
            search_query = f"latest financial news about {search_term}"
            tavily_results = tavily.invoke(search_query)
            
            if tavily_results and len(tavily_results) > 0:
                formatted_results = "Recent Financial News from Tavily Search:\n\n"
                for i, result in enumerate(tavily_results, 1):
                    source = result.get('source', 'Unknown Source')
                    title = result.get('title', 'No Title')
                    content = result.get('content', 'No content available')[:200] + "..."  # Limit content length
                    url = result.get('url', '#')
                    formatted_results += f"{i}. {title} ({source})\n"
                    formatted_results += f"   Summary: {content}\n"
                    formatted_results += f"   URL: {url}\n\n"
                return formatted_results
        except Exception as e:
            error_messages.append(f"Tavily search failed: {str(e)}")
            # Continue to next method
        
        # 2. Second attempt: Try MarketWatch
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            url = f"https://www.marketwatch.com/search?q={search_term}&ts=0&tab=All%20News"
            response = requests.get(url, headers=headers, timeout=10)  # Add timeout
            soup = BeautifulSoup(response.text, 'html.parser')
            
            articles = soup.select('.article__content')
            
            if articles and len(articles) > 0:
                formatted_results = "Recent Financial News from MarketWatch:\n\n"
                for i, article in enumerate(articles[:max_articles], 1):
                    headline_elem = article.select_one('.article__headline')
                    time_elem = article.select_one('.article__timestamp')
                    
                    headline = headline_elem.text.strip() if headline_elem else "No headline"
                    timestamp = time_elem.text.strip() if time_elem else "No timestamp"
                    link = headline_elem.get('href', '#') if headline_elem else "#"
                    
                    formatted_results += f"{i}. {headline}\n"
                    formatted_results += f"   Published: {timestamp}\n"
                    formatted_results += f"   URL: {link}\n\n"
                return formatted_results
        except Exception as e:
            error_messages.append(f"MarketWatch scraping failed: {str(e)}")
        
        # 3. Last resort: Try a more general approach with Yahoo Finance
        try:
            import yfinance as yf
            
            # Get some general market news by looking at major indices
            indices = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones, NASDAQ
            general_news = "Recent Market Overview:\n\n"
            
            for index in indices:
                ticker = yf.Ticker(index)
                info = ticker.info
                name = info.get('shortName', index)
                price = info.get('regularMarketPrice', 'N/A')
                change = info.get('regularMarketChange', 'N/A')
                change_percent = info.get('regularMarketChangePercent', 'N/A')
                
                if price != 'N/A':
                    change_str = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
                    change_percent_str = f"+{change_percent:.2f}%" if change_percent > 0 else f"{change_percent:.2f}%"
                    general_news += f"{name}: ${price} ({change_str}, {change_percent_str})\n"
            
            general_news += "\nUnable to fetch specific news about '{search_term}', but here's some general market information.\n"
            general_news += "Consider checking financial news websites directly for the most current information."
            
            return general_news
        except Exception as e:
            error_messages.append(f"Yahoo Finance fallback failed: {str(e)}")
        
        # If we made it here, all methods failed
        fallback_message = f"Unable to retrieve financial news about '{search_term}' at this time.\n\n"
        fallback_message += "Error details:\n" + "\n".join(error_messages)
        fallback_message += "\n\nPlease try again later or specify a different search term."
        return fallback_message
        
    except Exception as e:
        # This is the global catch-all
        return f"Error retrieving financial news: {str(e)}. Please try a different search term or try again later."

class StockAnalysis(BaseModel):
    """Schema for stock analysis results."""
    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Company name")
    current_price: float = Field(description="Current stock price")
    recommendation: str = Field(description="Buy, Hold, or Sell recommendation")
    reasons: List[str] = Field(description="Reasons for the recommendation")
    potential_upside: str = Field(description="Estimated potential upside percentage")
    risk_level: str = Field(description="Low, Medium, or High risk assessment")

@tool
def analyze_news_for_stock_picks(search_term: str = "undervalued stocks", num_picks: int = 5) -> str:
    """
    Analyze recent financial news to identify promising stock picks.
    
    Args:
        search_term: Term to search for (e.g., "undervalued stocks", "growth stocks")
        num_picks: Number of stock picks to recommend
    """
    try:
        # Get news about the search term
        news = scrape_financial_news(search_term, max_articles=10)
        
        # Trim news if it's too long to avoid token limit issues
        if len(news) > 8000:  # Reasonable limit for context
            news_lines = news.split("\n")
            # Keep the header and first few articles
            trimmed_news = "\n".join(news_lines[:50])
            trimmed_news += "\n\n[Note: News content was trimmed due to length. Showing first 3-5 articles only.]"
            news = trimmed_news
        
        # Use the LLM to analyze the news and extract stock picks
        llm = get_llm(temperature=0, max_tokens=2000)  # Limit response size
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a financial expert specialized in stock analysis. 
            Based on the financial news provided, identify the top {num_picks} most promising stocks to buy.
            
            For each stock:
            1. Provide the ticker symbol
            2. Name of the company
            3. Why it's a good investment right now
            4. Estimated potential upside (percentage range)
            5. Risk level (Low, Medium, High)
            
            Focus on evidence-based analysis from the news. If the news doesn't mention enough specific stocks,
            use your knowledge of current market trends to supplement, but make it clear when you're doing so.
            
            Keep your response concise and focused on the {num_picks} recommendations only.
            """),
            HumanMessage(content=f"Here is recent financial news about {search_term}:\n\n{news}\n\nBased on this news, what are the top {num_picks} stocks to buy right now?")
        ])
        
        response = llm.invoke(analysis_prompt)
        content = response.content
        
        # Further trim if the response is still too long
        if len(content) > 4000:
            content = content[:3997] + "..."
        
        # Return the analysis result
        return f"Analysis of Top {num_picks} Stocks to Buy Based on Recent News:\n\n{content}"
        
    except Exception as e:
        return f"Error analyzing news for stock picks: {str(e)}"

# ======= AGENTS DEFINITION =======

def get_tool_descriptions(tools):
    """
    Format tool descriptions according to the ReAct pattern.
    
    Args:
        tools: List of tool objects
        
    Returns:
        str: Formatted tool descriptions
    """
    tool_descriptions = []
    for tool in tools:
        tool_name = tool.name
        tool_desc = tool.description
        tool_descriptions.append(f"{tool_name}: {tool_desc}")
    
    return "\n".join(tool_descriptions)

def create_research_agent(tools):
    """Create a research agent for financial data gathering with ability to handoff to advisory."""
    # Add handoff tool to the tools list
    all_tools = tools + [transfer_to_advisory_agent]
    
    # Get formatted tool descriptions for the agent prompt
    tool_descriptions = get_tool_descriptions(all_tools)
    
    # Construct system prompt with enhanced ReAct instructions and added handoff capability
    system_prompt = f"""You are an expert Financial Research Agent that searches for and provides accurate, relevant financial information.
    
You have access to the following tools to help gather information:

{tool_descriptions}

Always search for the most current and accurate information available. 
For financial metrics, always include the date or time period the data represents.

To solve the user's query, follow the ReAct pattern (Reasoning and Acting) with this structure:

1. Thought: Think step-by-step about what information is needed and how to find it.
   - What specific data points are needed?
   - Which tools would provide the most reliable information?
   - How to break down a complex question into searchable components?

2. Action: Choose ONE appropriate tool from your available tools.
   - Only use ONE tool at a time
   - Make your selection based on which tool will provide the most relevant information

3. Action Input: Provide the specific input for the chosen tool.
   - Be precise with stock tickers, search terms, and time frames
   - Format the input exactly as the tool expects it

4. Observation: Review the information returned by the tool.
   - Based on what you learn, plan your next steps
   - Determine if you need additional information

COLLABORATION INSTRUCTIONS:
- Once you have gathered sufficient factual information, use transfer_to_advisory_agent to hand off to the Advisory Agent.
- The Advisory Agent specializes in analyzing data and providing investment recommendations.
- You should transfer to the Advisory Agent when:
  * You have gathered all the factual information required
  * The query requires analysis or recommendations beyond just facts
  * The question asks for investment advice or predictions

If you're handling a purely factual question that doesn't need analysis or recommendations, you can provide a final answer yourself:

Final Answer: Provide a comprehensive, well-organized summary of the information gathered.
- Present financial data in a clear, readable format (tables where appropriate)
- Include relevant metrics with their values and dates
- Cite sources where applicable
- Keep your answer factual and data-driven

If you cannot find specific information, state this clearly rather than making assumptions.
If the information found is incomplete, acknowledge the limitations of the data.

IMPORTANT: Break down complex answers into organized sections with headings for readability.
If your answer is becoming very long, prioritize the most crucial information first.
"""

    # Create a ReAct agent with enhanced parsing configuration
    # Using a lower temperature for research to prioritize accuracy
    llm = get_llm(temperature=0, max_tokens=4000)
    
    research_agent = create_react_agent(
        llm=llm,
        tools=all_tools,
        prompt=ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    )
    
    # Configure executor with enhanced error handling
    research_executor = AgentExecutor(
        agent=research_agent, 
        tools=all_tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10  # Prevent excessive iterations
    )
    
    return research_executor


def create_advisory_agent(tools):
    """Create an advisory agent for financial analysis and recommendations with ability to handoff to research."""
    # Add handoff tool to the tools list
    all_tools = tools + [transfer_to_research_agent]
    
    # Get formatted tool descriptions for the agent prompt
    tool_descriptions = get_tool_descriptions(all_tools)
    
    # Construct system prompt with enhanced ReAct instructions and added handoff capability
    system_prompt = f"""You are an expert Financial Advisory Agent that analyzes financial data and provides personalized investment recommendations.
    
You have access to the following tools to help with your analysis:

{tool_descriptions}

Your role is to interpret financial data, identify trends, assess risks, and provide actionable recommendations.

To provide high-quality financial advice, follow the ReAct pattern (Reasoning and Acting) with this structure:

1. Thought: Think step-by-step about the analysis needed and how to approach it.
   - What analysis methods are appropriate for this data?
   - What economic factors should be considered?
   - What are the key risks and opportunities?
   - How does this fit with the user's goals or constraints?

2. Action: Choose ONE appropriate tool from your available tools.
   - Only use ONE tool at a time
   - Select tools that will enhance your analysis

3. Action Input: Provide the specific input for the chosen tool.
   - Be precise with parameters
   - Format the input exactly as the tool expects it

4. Observation: Review the information returned by the tool.
   - Incorporate this into your overall analysis
   - Determine if you need additional information

COLLABORATION INSTRUCTIONS:
- If you need additional factual information, use transfer_to_research_agent to hand off to the Research Agent.
- The Research Agent specializes in gathering factual financial information.
- You should transfer to the Research Agent when:
  * You need more data points to complete your analysis
  * You need updated prices or metrics
  * You need background information on a company or economic trend

When you have all the information you need, provide your final answer:

Final Answer: Deliver comprehensive, well-reasoned financial advice that includes:
- Summary of key findings from your analysis
- Clear, actionable recommendations with justification
- Relevant risks and considerations
- Time horizon and context for your recommendations
- Alternative options where appropriate

Your recommendations should be:
- Data-driven and supported by specific findings
- Risk-aware, noting potential downsides
- Personalized to the user's situation when possible
- Clear about assumptions made
- Honest about limitations in the analysis

IMPORTANT: Break down complex analyses into organized sections with headings for readability.
If your analysis is becoming very long, prioritize the most crucial insights and recommendations first.
"""

    # Create a ReAct agent with enhanced parsing configuration
    # Using a slightly higher temperature for advisory to allow for creative problem-solving
    llm = get_llm(temperature=0.2, max_tokens=4000)
    
    advisory_agent = create_react_agent(
        llm=llm,
        tools=all_tools,
        prompt=ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    )
    
    # Configure executor with enhanced error handling
    advisory_executor = AgentExecutor(
        agent=advisory_agent, 
        tools=all_tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10  # Prevent excessive iterations
    )
    
    return advisory_executor

# ======= WORKFLOW GRAPH NODES =======

# Nodes are now implemented as methods in the FinancialAgentFramework class

# ======= HELPER FUNCTIONS =======

def extract_final_answer(text):
    """Extract the final answer from agent output with flexible pattern matching."""
    patterns = [
        r"(?i)final answer:\s*(.*?)(?=$)",
        r"(?i)final answer\s*(.*?)(?=$)",
        r"(?i)answer:\s*(.*?)(?=$)",
        r"(?i)recommendation:\s*(.*?)(?=$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, return the last 75% of the text as fallback
    # (Heuristic: final answers usually come after thinking/actions)
    lines = text.split("\n")
    start_idx = max(0, int(len(lines) * 0.25))
    return "\n".join(lines[start_idx:])


def extract_reasoning(text):
    """Extract the reasoning from agent output with flexible pattern matching."""
    patterns = [
        r"(?i)thought:\s*(.*?)(?=\n\s*(?:action|action input|observation|final answer|$))",
        r"(?i)reasoning:\s*(.*?)(?=\n\s*(?:action|action input|observation|final answer|$))",
        r"(?i)analysis:\s*(.*?)(?=\n\s*(?:action|action input|observation|final answer|$))"
    ]
    
    all_reasoning = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        all_reasoning.extend([m.strip() for m in matches if m.strip()])
    
    # Join all reasoning segments, but limit total length
    if all_reasoning:
        combined = " ".join(all_reasoning)
        if len(combined) > 1000:
            return combined[:997] + "..."
        return combined
    
    return ""

# Global framework instance for access by node functions
GLOBAL_FRAMEWORK = None

# ======= MAIN APPLICATION =======

class FinancialAgentFramework:
    """
    A multi-agent framework for financial analysis and recommendations.
    Implements LangGraph's multi-agent network approach where agents can hand off to each other.
    """
    
    def __init__(self):
        """Initialize the financial agent framework."""
        # Define all available tools for research (excluding handoff tools which are added separately)
        self.research_tools = [
            TavilySearchResults(max_results=3),
            get_stock_price,
            get_company_financials,
            get_economic_indicator,
            scrape_financial_news
        ]
        
        # Define all available tools for advisory (excluding handoff tools which are added separately)
        self.advisory_tools = [
            get_stock_price,
            get_company_financials,
            get_economic_indicator,
            analyze_news_for_stock_picks
        ]
        
        # Create LLM
        self.llm = get_llm()
        
        # Create agents with specific tool sets (handoff tools are added inside these functions)
        self.research_executor = create_research_agent(self.research_tools)
        self.advisory_executor = create_advisory_agent(self.advisory_tools)
        self.human_executor = create_human_agent()
        
        # Create the workflow graph
        self.workflow = self._create_graph()
        
        # Register this instance for global access by the node functions
        global GLOBAL_FRAMEWORK
        GLOBAL_FRAMEWORK = self
    
    def _create_graph(self):
        """Create the state graph for the multi-agent workflow using LangGraph's multi-agent network approach."""
        # Initialize the state graph with our state type
        graph_builder = StateGraph(AgentState)
        
        # Add all agent nodes to the graph
        # Each node is a function that processes messages using the respective executor
        graph_builder.add_node("human", self._run_human_node)
        graph_builder.add_node("research", self._run_research_node)
        graph_builder.add_node("advisory", self._run_advisory_node)
        
        # Add an entry point that passes to the human agent first
        graph_builder.add_edge(START, "human")
        
        # Add a conditional default edge for each agent
        # This edge is taken when the agent doesn't return a Command
        graph_builder.add_conditional_edges(
            "human",
            self._get_next_node,
            {
                "research": "research",
                "advisory": "advisory",
                "human": "human",  # Self-loop for clarification
                "end": END
            }
        )
        
        graph_builder.add_conditional_edges(
            "research",
            self._get_next_node,
            {
                "research": "research",  # Self-loop for more research
                "advisory": "advisory",
                "human": "human",
                "end": END
            }
        )
        
        graph_builder.add_conditional_edges(
            "advisory",
            self._get_next_node,
            {
                "research": "research",
                "advisory": "advisory",  # Self-loop for more analysis
                "human": "human",
                "end": END
            }
        )
        
        # Compile the graph
        return graph_builder.compile()
    
    def _run_human_node(self, state: AgentState) -> dict:
        """Process the current state with the human agent."""
        # Ensure the state has proper messages format
        messages = state.get("messages", [])
        
        try:
            # Run the human agent
            print("\n👤 Human Agent is processing your request...")
            result = self.human_executor.invoke({"input": messages[-1]["content"] if messages else "", "agent_scratchpad": []})
            
            # Check if a Command was returned (agent used a handoff tool)
            if isinstance(result, Command):
                # Return the Command directly to control routing
                return result
                
            # Otherwise, it's a regular result
            response = result.get("output", "I'm sorry, I couldn't determine how to help. Let me transfer you to the Research Agent.")
            
            # Update state with agent's message
            messages.append({"role": "assistant", "content": response, "name": "human_agent"})
            
            # Default to transferring to research if no explicit handoff
            return {
                "messages": messages,
                "current_agent": "research"
            }
        except Exception as e:
            print(f"\n❌ Human agent error: {str(e)}")
            messages.append({
                "role": "assistant", 
                "content": "I'm having trouble understanding. Let me connect you with our Research Agent for help.",
                "name": "human_agent"
            })
            return {
                "messages": messages,
                "current_agent": "research"
            }
    
    def _run_research_node(self, state: AgentState) -> dict:
        """Process the current state with the research agent."""
        # Extract the last message's content for the agent
        messages = state.get("messages", [])
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") in ["user", "assistant", "system"]:
                user_content = msg.get("content", "")
                break
        
        if not user_content:
            # Default message if none found
            messages.append({
                "role": "assistant",
                "content": "I don't have enough information. Please provide more details.",
                "name": "research_agent"
            })
            return {
                "messages": messages,
                "current_agent": "human"
            }
        
        try:
            # Run the research agent
            print("\n🔍 Research Agent is gathering information...")
            result = self.research_executor.invoke({"input": user_content, "agent_scratchpad": []})
            
            # Check if a Command was returned (agent used a handoff tool)
            if isinstance(result, Command):
                # Return the Command directly to control routing
                return result
            
            # Otherwise, it's a regular result
            response = result.get("output", "I couldn't find relevant information.")
            
            # Extract final answer and reasoning if needed
            final_answer = extract_final_answer(response)
            reasoning = extract_reasoning(response)
            
            # Store extracted information in state
            messages.append({"role": "assistant", "content": final_answer, "name": "research_agent"})
            
            # Return updated state
            return {
                "messages": messages,
                "research_output": final_answer,
                "research_reasoning": reasoning if reasoning else None,
                "current_agent": "human"  # Default return to human
            }
        except Exception as e:
            print(f"\n❌ Research agent error: {str(e)}")
            messages.append({
                "role": "assistant", 
                "content": "I encountered an issue while researching. Let's try a different approach.",
                "name": "research_agent"
            })
            return {
                "messages": messages,
                "current_agent": "human"
            }
    
    def _run_advisory_node(self, state: AgentState) -> dict:
        """Process the current state with the advisory agent."""
        # Extract message context for the agent
        messages = state.get("messages", [])
        
        # Get the most recent content for context
        recent_content = ""
        for msg in reversed(messages):
            if msg.get("role") in ["user", "assistant"]:
                recent_content = msg.get("content", "")
                break
        
        # Also get the original query if available
        original_query = state.get("original_input", "")
        
        # Combine research output with recent content
        research_output = state.get("research_output", "")
        
        # Prepare the prompt with available context
        prompt = f"""Question: {original_query if original_query else recent_content}

"""
        if research_output:
            prompt += f"Research Findings:\n{research_output}\n\n"
        
        prompt += "Based on this information, provide your best financial advice and recommendations."
        
        try:
            # Run the advisory agent
            print("\n💼 Advisory Agent is analyzing and preparing recommendations...")
            result = self.advisory_executor.invoke({"input": prompt, "agent_scratchpad": []})
            
            # Check if a Command was returned (agent used a handoff tool)
            if isinstance(result, Command):
                # Return the Command directly to control routing
                return result
            
            # Otherwise, it's a regular result
            response = result.get("output", "I couldn't develop a recommendation based on the available information.")
            
            # Extract final answer and reasoning if needed
            final_answer = extract_final_answer(response)
            reasoning = extract_reasoning(response)
            
            # Store extracted information in state
            messages.append({"role": "assistant", "content": final_answer, "name": "advisory_agent"})
            
            # Return updated state with the recommendation result
            return {
                "messages": messages,
                "advisory_reasoning": reasoning if reasoning else None,
                "result": final_answer,
                "current_agent": "human"  # Default return to human
            }
        except Exception as e:
            print(f"\n❌ Advisory agent error: {str(e)}")
            fallback = """
I encountered an issue while analyzing the information. Here are some general guidelines:

1. Diversify your investments across different asset classes
2. Consider your time horizon and risk tolerance
3. Stay informed about market conditions
4. Consult with a licensed financial advisor for personalized advice
"""
            messages.append({
                "role": "assistant", 
                "content": fallback,
                "name": "advisory_agent"
            })
            return {
                "messages": messages,
                "result": fallback,
                "current_agent": "human"
            }
    
    def _get_next_node(self, state: dict) -> str:
        """Determine which node to route to next."""
        # Check if this is the end of a conversation (has a result)
        if "result" in state and state["result"]:
            # End the conversation if we have a result
            return "end"
        
        # Get current agent from state
        current = state.get("current_agent", "human")
        
        # Return the current agent's preference if specified
        return current
    
    def run(self, query):
        """Run the financial agent framework with the given query."""
        # Initialize state for the graph
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "current_agent": "human",
            "research_output": None,
            "research_reasoning": None,
            "advisory_reasoning": None,
            "original_input": query,
            "result": None,
            "agent_turns": 0
        }
        
        # Invoke the graph with initial state
        final_state = self.workflow.invoke(initial_state)
        return final_state
    
    def run_interactive_session(self):
        """Run an interactive session with the financial agent framework."""
        print("\n======= Financial AI Agent Network Framework =======")
        print("Type 'exit' to quit at any time.\n")
        
        # Initial welcome message
        print("👋 Welcome to the Financial AI Agent Network! I can help you with:")
        print("  • Stock prices and company financials")
        print("  • Economic indicators and market trends")
        print("  • Investment advice and recommendations")
        print("  • Financial news analysis")
        print("What would you like to know about today?\n")
        
        conversation_state = None
        
        while True:
            try:
                # Get user input
                query = input("\nYour question: ")
                if query.lower() in ['exit', 'quit']:
                    print("\nThank you for using the Financial AI Agent Framework. Goodbye!")
                    break
                
                # Create initial state or continue existing conversation
                if conversation_state is None:
                    # New conversation
                    initial_state = {
                        "messages": [{"role": "user", "content": query}],
                        "current_agent": "human",
                        "research_output": None,
                        "research_reasoning": None,
                        "advisory_reasoning": None,
                        "original_input": query,
                        "result": None,
                        "agent_turns": 0
                    }
                else:
                    # Continue conversation
                    conversation_state["messages"].append({"role": "user", "content": query})
                    initial_state = conversation_state
                
                # Process the query
                conversation_state = self.workflow.invoke(initial_state)
                
                # Display the latest agent message
                if conversation_state.get("messages", []):
                    latest_msg = conversation_state["messages"][-1]
                    if latest_msg.get("role") == "assistant":
                        print(f"\n{latest_msg.get('name', 'Assistant')}: {latest_msg['content']}")
                
                # Check if conversation is complete
                if "result" in conversation_state and conversation_state["result"]:
                    print("\n✅ Conversation complete. You can ask a new question.")
                    conversation_state = None  # Reset for new conversation
                
            except KeyboardInterrupt:
                print("\nSession terminated by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                print("Let's start a new conversation.")
                conversation_state = None

# ======= EXAMPLE USAGE =======

def example_usage():
    """Example demonstrating how to use the financial agent framework with multi-agent networking."""
    print("\n======= Financial AI Agent Network Framework Example =======")
    print("This framework demonstrates LangGraph's multi-agent network approach where agents can hand off to each other.\n")
    
    framework = FinancialAgentFramework()
    
    # Example 1: Ask about a specific stock
    query1 = "What's the current price of Apple stock and do you recommend buying it?"
    print(f"\nExample Query 1: {query1}")
    result1 = framework.run(query1)
    if "result" in result1:
        print(f"\nFinal Recommendation: {result1['result']}")
        
        # Show the conversation flow if debugging
        print("\nConversation History:")
        for i, msg in enumerate(result1.get("messages", [])):
            role = msg.get("role", "unknown")
            name = msg.get("name", "system" if role == "system" else role)
            print(f"  {i+1}. {name}: {msg.get('content', '')[:100]}..." if len(msg.get('content', '')) > 100 else f"  {i+1}. {name}: {msg.get('content', '')}")
    
    print("\n" + "-"*50)
    
    # Example 2: Economic question that involves research only
    query2 = "What are the current inflation rates in the US?"
    print(f"\nExample Query 2: {query2}")
    result2 = framework.run(query2)
    if "research_output" in result2:
        print(f"\nResearch Findings: {result2['research_output']}")
        
    print("\n" + "-"*50)
    
    # Example 3: Complex question requiring both research and advisory input
    query3 = "What stocks should I consider for a long-term investment in renewable energy?"
    print(f"\nExample Query 3: {query3}")
    result3 = framework.run(query3)
    if "result" in result3:
        print(f"\nFinal Recommendation: {result3['result']}")
    
    print("\n======= End of Examples =======")
    print("To start an interactive session, run with --interactive flag.")

if __name__ == "__main__":
    # More flexible environment variable checking
    required_vars = []
    optional_vars = {"OPENAI_API_KEY": "Required for LLM functionality", 
                     "TAVILY_API_KEY": "Optional: Used for web search (needed for comprehensive research)"}
    
    missing_required = []
    missing_optional = []
    
    # Check for required environment variables
    for key in required_vars:
        if not os.environ.get(key):
            missing_required.append(key)
    
    # Check for optional environment variables
    for key, description in optional_vars.items():
        if not os.environ.get(key):
            missing_optional.append(f"{key} - {description}")
    
    # Handle missing environment variables
    if missing_required:
        print(f"Error: Missing required environment variables: {', '.join(missing_required)}")
        print("Please set these environment variables before running the script.")
        exit(1)
    
    if missing_optional:
        print("Warning: Some optional environment variables are not set:")
        for var in missing_optional:
            print(f"  - {var}")
        print("\nYou can continue, but some features may be limited or unavailable.")
        
        # Give the user a chance to continue or exit
        if "OPENAI_API_KEY" in [var.split()[0] for var in missing_optional]:
            print("\nOPENAI_API_KEY is required for the primary functionality of this application.")
            response = input("Continue with limited functionality? (y/n): ")
            if response.lower() != 'y':
                print("Exiting.")
                exit(1)
    
    # Run the example or interactive session
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        framework = FinancialAgentFramework()
        framework.run_interactive_session()
    else:
        example_usage()

# ======= HANDOFF TOOLS FOR MULTI-AGENT NETWORK =======

from typing import Annotated
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


def make_handoff_tool(*, agent_name: str):
    """Create a tool that allows an agent to hand off to another agent via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        
        # Return a Command to transfer control to another agent
        return Command(
            # navigate to the target agent node
            goto=agent_name,
            # This is the state update that the target agent will see
            update={"messages": state.get("messages", []) + [tool_message]}
        )

    return handoff_to_agent


# Create specific handoff tools for our agents
transfer_to_research_agent = make_handoff_tool(agent_name="research")
transfer_to_advisory_agent = make_handoff_tool(agent_name="advisory")

# Create human agent handoff tool
transfer_to_human = make_handoff_tool(agent_name="human")

def create_human_agent():
    """Creates a simple human agent for handling user interaction."""
    system_prompt = """You are a friendly assistant that helps users interact with our financial AI agents.
    
Your role is to:
1. Welcome users and explain the capabilities of our financial AI system
2. Clarify user questions if needed
3. Hand off questions to the appropriate specialized agent:
   - Research Agent: For gathering factual financial information
   - Advisory Agent: For analyzing data and providing investment recommendations

If the user has a question about financial information, stocks, economic indicators, or company data, 
use transfer_to_research_agent to route their question to our Research Agent.

If the user has a question about investment advice, portfolio recommendations, or market analysis,
use transfer_to_advisory_agent to route their question to our Advisory Agent.
"""

    # Create tools for human agent
    human_tools = [transfer_to_research_agent, transfer_to_advisory_agent]
    
    # Get the LLM
    llm = get_llm(temperature=0.3, max_tokens=2000)
    
    # Create the human agent
    human_agent = create_react_agent(
        llm=llm,
        tools=human_tools,
        prompt=ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    )
    
    # Configure executor with error handling
    human_executor = AgentExecutor(
        agent=human_agent,
        tools=human_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3  # Limit iterations as human agent should quickly hand off
    )
    
    return human_executor
