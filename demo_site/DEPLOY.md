# Deployment Guide: OptiRAG-lite Interactive Demo

This directory (`demo_site/`) contains a static, zero-dependency web application (vanilla HTML5, CSS3, and JavaScript).
It requires **no build step, no npm packages, and no external CDNs**. It operates completely offline.

---

## Option 1: Deploy to Vercel (Static Hosting - 100% Free)

> [!CAUTION]
> **Strict Free-Tier Rule**: Vercel offers free static hosting on their Hobby plan. If Vercel ever prompts you for a credit card, payment details, or an account upgrade, **STOP immediately and do not proceed**.

### Step-by-Step Instructions:

1. **Prerequisites**: Ensure Node.js is installed on your computer.
   - You can verify by opening PowerShell / terminal and running:
     ```bash
     node -v
     ```

2. **Navigate to the demo site directory**:
   ```bash
   cd demo_site
   ```

3. **Log in to Vercel**:
   ```bash
   npx vercel login
   ```
   *(Select your preferred login method, e.g., GitHub or Email, and authenticate in your browser).*

4. **Deploy Preview**:
   ```bash
   npx vercel
   ```
   When prompted by the Vercel CLI, answer with the following exact settings:
   - **Set up and deploy?**: `Y`
   - **Which scope do you want to deploy to?**: Select your personal account
   - **Link to existing project?**: `N`
   - **What’s your project’s name?**: `optirag-lite-demo` (or press Enter)
   - **In which directory is your code located?**: `./` (press Enter)
   - **Want to modify these settings?**: `N`

5. **Deploy to Production**:
   ```bash
   npx vercel --prod
   ```
   Vercel will output a live production URL (e.g., `https://optirag-lite-demo.vercel.app`).

---

## Option 2: Local Offline Execution (Backup / Classroom Demo)

If you have no internet connection or prefer not to deploy online, you can run the full interactive demonstration locally using Python's built-in web server:

1. **Open PowerShell / Terminal in `demo_site/`**:
   ```bash
   cd demo_site
   python -m http.server 8000
   ```

2. **Open in Your Web Browser**:
   - Main Demo: [http://localhost:8000](http://localhost:8000)
   - Parity Test Suite: [http://localhost:8000/test.html](http://localhost:8000/test.html)

3. **To Stop the Local Server**:
   - Press `Ctrl + C` in the console window.
