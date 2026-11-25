# **User Interfaces of OpenBrowser**

<!-- toc:start -->
## Contents

- [1. Setup](#1-setup)
- [2. Available interfaces](#2-available-interfaces)

<!-- toc:end -->

## 1. Setup

Install the optional UI dependencies first:

```bash
pip install gradio streamlit
```

## 2. Available interfaces

| **File Name**          | **User Interface** | **Description**                           | **Example Usage**                         |
|------------------------|-------------------|-------------------------------------------|-------------------------------------------|
| `command_line.py`      | **Terminal**      | Parses arguments for command-line execution. | `python command_line.py`                  |
| `gradio_demo.py`       | **Gradio**        | Provides a Gradio-based interactive UI.  | `python gradio_demo.py`                   |
| `streamlit_demo.py`    | **Streamlit**     | Runs a Streamlit-based web interface.    | `python -m streamlit run streamlit_demo.py` |
