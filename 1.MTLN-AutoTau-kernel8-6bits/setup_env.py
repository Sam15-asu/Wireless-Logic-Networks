import os
import subprocess
import sys

def setup_environment():
    # 1. Create .venv if it doesn't exist
    if not os.path.exists(".venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    else:
        print("Virtual environment already exists.")

    # 2. Determine path to python/pip in .venv
    if os.name == "nt":  # Windows
        python_executable = os.path.join(".venv", "Scripts", "python.exe")
        pip_executable = os.path.join(".venv", "Scripts", "pip.exe")
    else:  # Linux/macOS
        python_executable = os.path.join(".venv", "bin", "python")
        pip_executable = os.path.join(".venv", "bin", "pip")

    # 3. Upgrade pip
    print("Updating pip...")
    subprocess.run([python_executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # 4. Install requirements
    if os.path.exists("requirements.txt"):
        print("Installing requirements.txt...")
        subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
    else:
        print("requirements.txt not found.")

if __name__ == "__main__":
    setup_environment()
    print("Setup complete.")

source .venv/bin/activate
