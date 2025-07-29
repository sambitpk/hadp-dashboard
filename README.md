
# HADP Scheme Kiosk POC

This is a proof-of-concept (PoC) kiosk interface built using **Streamlit**, designed to help users browse and search various HADP government scheme data in kiosk mode.

## Features

- Local kiosk-ready UI
- Browse Excel sheets of scheme work data
- Search by block, department, or scheme
- Download filtered data

## Running Locally

1. Install dependencies:
```bash
pip install streamlit pandas openpyxl
```

2. Run the app:
```bash
streamlit run dashboard.py
```

3. Use in full screen (kiosk) mode via browser.

## Folder Structure

```
HADP_Kiosk_POC/
│
├── dashboard.py               # Main Streamlit UI
├── README.md                  # Instructions
└── HADP WORK LIST...xlsx      # Excel data source
```
