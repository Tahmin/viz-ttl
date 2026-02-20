# viz-ttl

## Overview

TASS Report Explorer — An interactive genomic surveillance dashboard for Influenza A Virus positive control QC analysis. Visualize and analyze genetic diversity, sequencing quality, and geographic distribution across multiple samples.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd viz-ttl
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Dashboard

1. Set up your OpenAI API key in the `.env` file:
```
OPENAI_API_KEY="your-openai-api-key-here"
```

2. Start the dashboard with:
```bash
python visualize.py
```

The dashboard will be available at **http://127.0.0.1:8050** in your browser.

## Features

- **Interactive Visualizations**: Explore TASS scores, breadth coverage, and segment analysis
- **AI Chat Panel**: Ask questions about your dataset using the right-side chat powered by OpenAI
- **Multi-File Support**: Upload one or multiple CSV files through the UI
- **Filtering**: Filter by sample, state, and segment
- **Geographic View**: See the distribution of positive controls across the US
- **Quality Metrics**: Access detailed QC metrics for each sample

## Data Format

Upload CSV files containing:
- Sample identifiers
- TASS Score
- Breadth Coverage percentage
- Geographic data (state)
- Collection dates

## Chat Feature

The AI chat panel on the right side of the dashboard allows you to ask questions about your data:

- **Ask analytical questions**: "What's the average TASS score?", "Which samples have high breadth coverage?"
- **Get insights**: The AI analyzes your dataset and provides data-driven responses
- **Conversation history**: All messages are preserved during your session and automatically cleared when you upload new data

**Requirements:**
- Valid OpenAI API key stored in `.env` file
- Data must be uploaded first before using the chat

