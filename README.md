
# SBECMS — Satellite-Based Environmental Change Monitoring System

SBECMS is an interactive Streamlit dashboard for analyzing environmental change using satellite-derived indicators. It helps users explore land condition, water presence, surface heat, and overall change patterns through a clean visual interface.

## Overview

This project is designed to monitor and compare environmental conditions across different area types and event categories. It supports both uploaded datasets and demo data, making it useful for project demonstrations, academic presentations, and portfolio work.

The dashboard focuses on:
- Green cover analysis
- Water presence analysis
- Surface heat analysis
- Change monitoring
- Place-based comparison
- Time-based trend analysis
- Map-based visualization

## What the project does

SBECMS allows users to:

- Upload CSV or Excel datasets
- Use built-in demo data for testing
- Filter data by:
  - Area Type
  - Situation
  - Green Cover Score
  - Water Presence Score
  - Surface Heat
  - Change Score
  - Place
- View summary cards for key metrics
- Explore:
  - Main comparison charts
  - Level breakdown charts
  - Place comparison
  - Time view
  - Map view
  - Filtered data table
- Download filtered data and summary outputs

## Main Features

### 1. Interactive Dashboard
A dark-themed Streamlit interface with a clean dashboard layout for real-time exploration of the dataset.

### 2. Data Upload Support
Users can upload:
- `.csv`
- `.xlsx`

### 3. Demo Dataset
The app includes generated demo data so the dashboard can run even without external files.

### 4. Data Cleaning and Validation
The system:
- standardizes column names
- converts numeric fields
- parses dates
- handles missing values
- validates required columns

### 5. Comparative Visual Analysis
The dashboard includes:
- Green Cover vs Surface Heat
- Green Cover vs Water Presence
- Change Score by Situation

### 6. Place Comparison
Shows which places have the highest average change score.

### 7. Time View
Displays monthly trends for:
- Green Cover
- Water Presence
- Change Score
- Surface Heat

### 8. Map View
If latitude and longitude are available, the dashboard displays a geographic map of points with event-based coloring.

### 9. Key Findings
Automatically generates short observations from the filtered data.

### 10. Data Export
Users can download:
- filtered dataset
- place summary

## Technologies Used

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **PyDeck**
- **OpenPyXL**

## Project Structure

```bash
SBECMS/
│
├── app.py
├── requirements.txt
├── README.md