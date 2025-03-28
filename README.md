# Stock Analysis Multi-Agent System

A powerful LangGraph-based multi-agent system that analyzes financial markets to identify and research promising stocks. The system leverages a collaborative team of specialized AI agents to deliver comprehensive stock analysis with visualizations.

## Overview

This project implements a distributed AI workflow for stock analysis and investment recommendations:

1. **Research Agent** - Identifies promising stocks through market research and financial data analysis
2. **Decision Maker** - Evaluates research to select the top 2 most promising stocks
3. **Chart Generator** - Creates insightful visualizations of stock performance and metrics
4. **Report Agent** - Compiles findings into a comprehensive investment report

## Features

- Research-driven stock selection using real-time market data
- Collaborative agent workflow with specialized roles
- Financial data analysis using Yahoo Finance API
- Data visualization with matplotlib
- Web search capabilities using Tavily Search
- Configurable agent parameters (token limits, delay between calls)

## Requirements

- Python 3.7+
- OpenAI API key
- Tavily API key
- Required Python libraries (see requirements.txt for complete list):
  - langchain (and related packages)
  - langgraph
  - yfinance
  - pandas
  - matplotlib
  - openai

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-analysis-agents.git
cd stock-analysis-agents
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies using the requirements.txt file:
```bash
pip install -r requirements.txt
```

4. Set up API keys (create a .env file or export directly):
```bash
# Option 1: Create a .env file
echo "OPENAI_API_KEY=your_openai_api_key" > .env
echo "TAVILY_API_KEY=your_tavily_api_key" >> .env

# Option 2: Export directly in terminal
export OPENAI_API_KEY='your_openai_api_key'
export TAVILY_API_KEY='your_tavily_api_key'
```

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook FinalLabProject.ipynb
```

Follow the notebook cells to:
1. Configure agent parameters (delay between calls, token limits)
2. Run the agent workflow to analyze stocks
3. Review the final stock recommendations and report

## How It Works

The system implements a collaborative agent workflow:

1. **Research Phase**
   - Research agent develops a research strategy
   - Collects data on promising stocks using Tavily search and Yahoo Finance

2. **Decision Phase**
   - Decision maker analyzes research to select top 2 stocks
   - Evaluates growth potential, financial health, competitive advantage

3. **Visualization Phase**
   - Chart generator creates visual representations of stock data
   - Produces price trends, comparative analysis, and key metrics

4. **Reporting Phase**
   - Report agent creates a comprehensive investment thesis for selected stocks
   - Combines research, decisions, and charts into a final report

## Troubleshooting

- If you encounter API rate limit issues, try increasing the delay between agent calls
- For memory issues, reduce the max token limit for LLM agents
- Ensure your API keys are correctly set up and have sufficient quota

## License

MIT

## Acknowledgments

- This project utilizes LangGraph and LangChain frameworks
- Financial data provided by Yahoo Finance API
- Web search capabilities powered by Tavily 