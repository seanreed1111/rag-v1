#! /bin/bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt install python3 -y && \
sudo apt install python3-pip -y && \
export HOME="/home/vmuser" && \
export PATH="$HOME/.local/bin:$PATH" && \
pip install -r requirements.txt && \

# DEFAULT PORT for streamlit is 8501
streamlit run app.py
