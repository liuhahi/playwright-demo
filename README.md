# MCP Playwright Demo

A demonstration of using Model Context Protocol (MCP) with Playwright for automated web interactions using LangChain and OpenAI's GPT-4.

## Overview

This demo shows how to connect to a Playwright MCP server to perform automated web browser tasks. The example demonstrates logging into a web application and extracting clickable elements from the resulting page.

## Prerequisites

- Python 3.7+
- OpenAI API key
- Playwright MCP server running on localhost:8931

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required dependencies:
```bash
pip install langchain-mcp-adapters langchain langchain-openai python-dotenv
```

3. Create a `.env` file in the project root with your credentials:
```env
OPENAI_API_KEY=your_openai_api_key_here
USER_NAME=your_username
PASSWORD=your_password
VERIFICATION_CODE=your_verification_code
```

## Usage

1. Start the MCP Playwright server:
```bash
npx @playwright/mcp@latest --port 8931 --shared-browser-context
```

2. Ensure the Playwright MCP server is running on `http://localhost:8931/mcp`

3. Run the demo:
```bash
python mcp-demo.py
```

## What It Does

The demo performs the following automated tasks:
1. Connects to a Playwright MCP server
2. Navigates to a login page
3. Fills in username, password, and verification code fields
4. Clicks the Sign In button
5. Waits for the page to load
6. Returns all clickable items on the resulting page

## Configuration

- **MCP Server URL**: Configure the Playwright server URL in the `MultiServerMCPClient` configuration
- **Model**: Currently uses GPT-4o, can be changed in the `ChatOpenAI` initialization
- **Credentials**: All sensitive data is loaded from environment variables

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys and credentials secure
- This demo is for educational purposes only