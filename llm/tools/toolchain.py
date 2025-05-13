from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import DuckDuckGoSearchRun


@tool
def add(a: int, b: int) -> int:
 """ Adds two numbers together. """ # docstring gets used as the description
 return a + b

@tool
def multiply(a: int, b: int) -> int:
    """ Multiply two numbers. """
    return a * b

@tool
def divide(a:int, b: int) -> int:
    """ Divide two numbers. """
    return a / b
    
@tool
def square(a) -> int:
    """ Calculate the square of a number. """
    a = int(a)
    return a * a

# Requires API key
web_search = TavilySearchResults(max_results=2)

# Privacy-focused search engine
#web_search = DuckDuckGoSearchRun()
finance_search = YahooFinanceNewsTool()
toolkit = [web_search, finance_search, add, multiply, divide, square]