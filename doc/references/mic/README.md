<!-- ABOUT THE PROJECT -->
## About The Project

This is a reference for using webcam mic.


<!-- GETTING STARTED -->
## Getting Started

Instructions on setting up your project locally.

### Prerequisites

  ```sh
  sudo apt update
  sudo apt install python3-pyaudio
  ```

### Python virtual env

  ```sh
  python3 -m venv venv_mic
  source venv_mic/bin/activate
  pip install -U pip
  ```

### Installation

  ```sh
  pip install pyaudio
  ```

<!-- USAGE EXAMPLES -->
## Usage

  1. Run the python code.
  ```sh
  python3 test.py
  ```
  2. Record you voice for the predefined time. (default: 5 sec)
  3. `output.wav` will be created as a result of your record.
