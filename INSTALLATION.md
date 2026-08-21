# Installation Guide

Complete step-by-step instructions for installing the Brain Hemorrhage Detection System.

## Table of Contents

1. [Windows Installation](#windows-installation)
2. [macOS Installation](#macos-installation)
3. [Linux Installation](#linux-installation)
4. [Troubleshooting](#troubleshooting)

---

## Windows Installation

### Step 1: Install Python

1. **Download Python**
   - Visit https://www.python.org/downloads/
   - Download Python 3.10 or 3.11 (recommended)
   - Click on the installer

2. **Run the Installer**
   - Double-click the downloaded `.exe` file
   - Check "Add Python to PATH" ⚠️ **IMPORTANT**
   - Click "Install Now"

3. **Verify Installation**
   - Open Command Prompt (Win + R, type `cmd`, press Enter)
   - Type: `python --version`
   - Should show: `Python 3.x.x`

### Step 2: Download the Application

1. Download or clone the `brain_hemorrhage_detection` folder to your desired location
2. Open Command Prompt
3. Navigate to the folder: `cd C:\path\to\brain_hemorrhage_detection`

### Step 3: Run Setup Script

**Option A: Automatic Setup (Recommended)**
- Double-click `setup.bat` in the project folder
- Follow the prompts
- The script will install Python dependencies and create the model

**Option B: Manual Setup**

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
python -m pip install -r requirements.txt

# Create the model
python model/train_model.py
```

### Step 4: Run the Application

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Step 5: Open in Browser

Open your web browser and go to: **http://localhost:5000**

---

## macOS Installation

### Step 1: Install Python

**Option A: Using Homebrew (Recommended)**
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Add to PATH (add to ~/.zprofile or ~/.bash_profile)
echo 'export PATH="/usr/local/opt/python@3.11/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile
```

**Option B: Direct Download**
1. Visit https://www.python.org/downloads/
2. Download macOS installer
3. Run the installer

### Step 2: Download the Application

```bash
cd ~/Downloads  # or your preferred location
# Download or clone the project folder
cd brain_hemorrhage_detection
```

### Step 3: Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

The script will:
- Create a virtual environment
- Install dependencies
- Create the model

### Step 4: Run the Application

```bash
source venv/bin/activate
python app.py
```

### Step 5: Open in Browser

Open: **http://localhost:5000**

---

## Linux Installation

### Step 1: Install Python

**Ubuntu/Debian**
```bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv python3.11-dev
sudo apt-get install python3-pip
```

**Fedora/RHEL**
```bash
sudo dnf install python3.11 python3.11-devel
```

**Arch**
```bash
sudo pacman -S python
```

### Step 2: Download the Application

```bash
cd ~/projects  # or your preferred location
# Download or clone the project folder
cd brain_hemorrhage_detection
```

### Step 3: Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

### Step 4: Run the Application

```bash
source venv/bin/activate
python app.py
```

### Step 5: Open in Browser

Open: **http://localhost:5000**

---

## Manual Installation (All Platforms)

If the setup scripts don't work, follow these steps:

### Step 1: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Upgrade pip

```bash
# Windows
python -m pip install --upgrade pip

# macOS/Linux
python3 -m pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask 2.3.3
- TensorFlow 2.13.0
- OpenCV 4.8.0.76
- NumPy 1.24.3
- And more...

### Step 4: Create the Model

```bash
# Windows
python model/train_model.py

# macOS/Linux
python3 model/train_model.py
```

### Step 5: Run the Application

```bash
# Windows
python app.py

# macOS/Linux
python3 app.py
```

---

## Troubleshooting

### Issue: "Python not found"

**Solution:**
- Restart your terminal after Python installation
- On Windows, ensure "Add Python to PATH" was checked during installation
- Try using `python3` instead of `python`

### Issue: "pip install fails"

**Possible causes and solutions:**

1. **Internet connection issue**
   - Check your internet connection
   - Try: `pip install --retries 5 -r requirements.txt`

2. **Insufficient disk space**
   - TensorFlow requires ~2GB
   - Free up disk space and retry

3. **Outdated pip**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

4. **Permission denied on macOS/Linux**
   - Use: `pip install --user -r requirements.txt`
   - Or use virtual environment (recommended)

### Issue: "Model file not found"

**Solution:**
```bash
python model/train_model.py
```

This will create `model/hemorrhage_model.h5`

### Issue: "Port 5000 already in use"

**Solution:**
Edit `app.py` and change the port:
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Changed from 5000 to 5001
```

### Issue: "Out of memory" error

**Solution:**
1. Close other applications
2. Increase virtual memory:
   - Windows: Settings → System → About → Advanced system settings
   - macOS/Linux: Check available RAM with `free -h`
3. Restart the system

### Issue: "TensorFlow/CUDA errors"

**Solution:**
For GPU support (optional):
```bash
pip install tensorflow[and-cuda]
```

For CPU-only (recommended for first-time users):
```bash
pip install tensorflow
```

### Issue: "Module not found" errors

**Solution:**
Make sure virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Then reinstall:
```bash
pip install -r requirements.txt
```

---

## Verification

After installation, verify everything works:

```bash
# Check Python
python --version

# Check pip packages
pip list

# Test model creation
python model/train_model.py

# Test application startup
python app.py
```

Expected output:
- Python version should be 3.8+
- pip list should show Flask, TensorFlow, etc.
- Model creation should complete without errors
- App should start at http://127.0.0.1:5000

---

## System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Python | 3.8+ | 3.10 or 3.11 recommended |
| RAM | 4GB minimum | 8GB recommended for TensorFlow |
| Disk Space | 2GB | For dependencies and model |
| OS | Windows, macOS, Linux | Any modern OS supported |
| Internet | Required for installation | Only during setup |

---

## Next Steps

After successful installation:

1. Read the [README.md](README.md) for usage instructions
2. Try uploading a test CT scan image
3. Explore the API endpoints
4. Consider training with your own data

---

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review the README.md
3. Check Python version: `python --version`
4. Verify pip: `pip --version`
5. Try a fresh virtual environment

---

## Uninstallation

To remove the application:

**Windows:**
1. Delete the `brain_hemorrhage_detection` folder
2. Optionally uninstall Python from Control Panel

**macOS/Linux:**
```bash
rm -rf ~/path/to/brain_hemorrhage_detection
# Virtual environment will be deleted with the folder
```

---

Happy analyzing! 🧠
