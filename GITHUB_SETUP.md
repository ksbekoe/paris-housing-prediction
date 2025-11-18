# 🚀 GitHub Setup Guide

Follow these steps to connect your local repository to GitHub and push your code.

## Step 1: Configure Git (if not already done)

First, set your git identity (replace with your actual name and email):

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Note:** Use the same email associated with your GitHub account for best results.

## Step 2: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `linear-regression` (or any name you prefer)
   - **Description**: "Paris Housing Price Prediction - End-to-End ML Project"
   - **Visibility**: Choose Public or Private
   - **⚠️ IMPORTANT**: Do NOT initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## Step 3: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these commands in your terminal:

```bash
# Make sure you're in the project directory
cd /Users/kyi/Desktop/ds-projects/linear-regression

# Add all files (already done, but just to be sure)
git add .

# Create your first commit
git commit -m "Initial commit: Paris Housing Price Prediction ML Project"

# Rename branch to main (if needed)
git branch -M main

# Add GitHub as remote (replace YOUR_USERNAME and YOUR_REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

## Step 4: Authentication

When you run `git push`, you'll be prompted for authentication. You have two options:

### Option A: Personal Access Token (Recommended)
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name (e.g., "Linear Regression Project")
4. Select scopes: check `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. When prompted for password during `git push`, paste the token instead

### Option B: SSH Key (More Secure for Long-term)
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your.email@example.com"`
2. Add to SSH agent: `eval "$(ssh-agent -s)"` then `ssh-add ~/.ssh/id_ed25519`
3. Copy public key: `cat ~/.ssh/id_ed25519.pub` (copy the output)
4. Go to GitHub → Settings → SSH and GPG keys → New SSH key
5. Paste your public key and save
6. Use SSH URL instead: `git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git`

## Troubleshooting

### If you get "remote origin already exists":
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### If you get authentication errors:
- Make sure you're using a Personal Access Token (not your GitHub password)
- Check that the token has `repo` permissions
- Try using SSH instead (Option B above)

### If you need to update the remote URL:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

## Success! 🎉

Once pushed, you can view your repository at:
`https://github.com/YOUR_USERNAME/YOUR_REPO_NAME`


