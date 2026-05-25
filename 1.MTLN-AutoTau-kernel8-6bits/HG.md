# Guide: Downloading Wireless Dataset from Hugging Face to a Remote Server

This document outlines the step-by-step action items required to verify your remote server's environment, set up authentication, and securely download the private [Sam10Man/Wireless](https://huggingface.co/datasets/Sam10Man/Wireless/tree/main) dataset.

---

## Phase 1: Remote Server Health & Space Checks
Before running any download scripts, you must ensure the remote server has enough storage and the correct network permissions.

- [ ] **1. Check Available Disk Space**
  * **Why:** Your dataset is approximately 25 GB. You need to ensure the server directory has enough room to store it safely without crashing.
  * **Action:** Run the following command in your server's terminal to check available space in gigabytes:
    ```bash
    df -h .
    ```
  * **Verification:** Ensure the `Avail` column shows at least **30 GB to 40 GB** of free space to accommodate the download and any unzipping/processing.

- [ ] **2. Verify Internet & Hugging Face Connectivity**
  * **Why:** Some remote enterprise or university servers restrict external web traffic via firewalls.
  * **Action:** Test if your server can communicate with Hugging Face by running:
    ```bash
    curl -I [https://huggingface.co](https://huggingface.co)
    ```
  * **Verification:** If you see a response starting with `HTTP/2 200` or `HTTP/1.1 200 OK`, your server has external web access.

---

## Phase 2: Environment Setup
Prepare the remote server with the necessary software packages.

- [ ] **3. Verify Python Installation**
  * **Action:** Check if Python 3 is installed:
    ```bash
    python3 --version
    ```
  * **Verification:** Ensure it returns Python `3.8` or higher.

- [ ] **4. Install the Hugging Face Hub Library**
  * **Action:** Install the official download utility package:
    ```bash
    pip install huggingface_hub
    ```

---

## Phase 3: Authentication (Required for Private Repos)
Because your repository is marked as **private**, the remote server needs explicit permission to access it using your specific token.

- [ ] **5. Authenticate the Remote Server**
  * **Action:** Run the login command in your server's terminal:
    ```bash
    huggingface-cli login
    ```
  * **Verification:** When prompted for your token, copy and paste your actual token string:
    ```text
    REMOVED_TOKEN
    ```
    *(Note: The terminal will not show characters while pasting for security. Just paste and hit Enter.)* Once it says `Login successful`, your server is globally authenticated.

---

## Phase 4: Execution (The Python Script)
Create and execute the download script inside your model project directory on the server.

- [ ] **6. Create the Download Script**
  * **Action:** Create a new file named `download_data.py` on your server and paste the following optimized script. It uses your token as a backup environment fallback:

    ```python
    import os
    from huggingface_hub import snapshot_download

    # Configuration
    REPO_ID = "Sam10Man/Wireless"
    LOCAL_TARGET_DIR = "./local_wireless_data"
    HF_TOKEN = "REMOVED_TOKEN"

    print(f"Starting download from {REPO_ID}...")

    try:
        # snapshot_download pulls files directly into your specified directory
        downloaded_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_TARGET_DIR,
            token=HF_TOKEN,
            ignore_patterns=[".gitattributes"] # Skips metadata files you don't need for training
        )
        print("\n" + "="*40)
        print("SUCCESS!")
        print(f"Dataset successfully downloaded and verified.")
        print(f"Location on server: {os.path.abspath(downloaded_path)}")
        print("="*40)

    except Exception as e:
        print("\n" + "!"*40)
        print("DOWNLOAD FAILED!")
        print(f"Error details: {e}")
        print("!"*40)
    ```

- [ ] **7. Run the Script**
  * **Action:** Execute the script in your terminal background or inside a screen/tmux session (since 25 GB might take a while depending on server speeds):
    ```bash
    python3 download_data.py
    ```