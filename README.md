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

Start the dashboard with:
```bash
python visualize.py
```

The dashboard will be available at **http://127.0.0.1:8050** in your browser.

## Features

- **Interactive Visualizations**: Explore TASS scores, breadth coverage, and segment analysis
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

