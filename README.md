# Primer AI Kit Demo

## Project Description

This repository provides a guide for setting up a local Large Language Model (LLM) serving environment on a Rubik Pi 3 (RPi3), utilizing Ubuntu Server, Python 3.12, and the Ollama framework.

## 1. Initial Hardware Setup: Ubuntu on Rubik Pi 3

The first step is to follow [this guide](https://www.thundercomm.com/rubik-pi-3/en/docs/rubik-pi-3-user-manual/1.0.0-u/Device%20Setup/set-up-your-device) to install Ubuntu Server on the Rubik Pi 3. Ubuntu host is reccomended.

## 2. Setting Up the Development Environment

After logging in, execute the following commands to install necessary dependencies and clone the project.

**System Update:**
```bash
sudo apt update
sudo apt upgrade -y
```

**Install Git:**
```bash
sudo apt install git -y
```

**Clone the Project:**
```bash
git clone https://github.com/SaintSampo/Primer-Software
cd Primer-Software
```

### 2.2. Installing Python 3.12

**Install Python 3.12 and Venv Module:**
```bash
sudo apt install python3.12 python3.12-venv -y
```

**Verify Installation:**
```bash
python3.12 --version
# Output should be: Python 3.12.x
```

### 2.3. Installing Ollama

Ollama is the core LLM serving application. Its installer script automatically handles architecture detection for the ARM platform.

**Execute the Ollama Installer:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Verify Service Status:**
Check that the Ollama service is running in the background.
```bash
systemctl status ollama
```

**Run a Test Model:**
Download a lightweight model to confirm functionality. (Due to RPi3's 1GB RAM, always choose small, highly quantized models like `tinyllama`).
```bash
ollama run tinyllama
```
Type `/bye` to exit the interactive session.

## 3. Python Development Environment

### 3.0 install system requirement library

```bash
sudo usermod -a -G gpio ubuntu
```

### 3.1. Creating a Virtual Environment (venv)

Isolating dependencies in a virtual environment is essential for project stability.

**Create Venv using Python 3.12:**
```bash
python3.12 -m venv venv
```

**Activate the Virtual Environment:**
```bash
source venv/bin/activate
```
> **Note**: Your command prompt will change (e.g., `(venv)user@host:~...`) indicating the environment is active. You must run this command every time you open a new terminal session.

### 3.2. Installing Dependencies

Install all necessary libraries defined in the `requirements.txt` file.

**Install Packages:**
```bash
pip install -r requirements.txt
```

The Rubik Pi 3 is now fully configured.

### 3.3. Run the Demo

Now that the virtual enviornment is active with all the needed dependancies, you can run the example file
```bash
sudo ./src/Primer.py
```
