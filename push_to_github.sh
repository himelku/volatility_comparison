#!/bin/bash

# ------------------------------------------
# CONFIGURATION — CHANGE THESE VALUES
# ------------------------------------------
GITHUB_USERNAME="himelku"                           # Your GitHub username
GITHUB_REPO_NAME="volatility_comparison"    # Your GitHub repository name
COMMIT_MESSAGE="Initial commit with all files" # Customize your commit message
BRANCH_NAME="main"                             # Usually 'main' or 'master'

# ------------------------------------------
# SCRIPT START
# ------------------------------------------

# Initialize git repository (if not already initialized)
git init

# Add remote origin (HTTPS with token recommended)
# NOTE: Replace 'https://github.com' with your repo URL
# e.g., 'https://github.com/username/repo.git' or SSH 'git@github.com:username/repo.git'
git remote add origin https://github.com/$GITHUB_USERNAME/$GITHUB_REPO_NAME.git

# Stage all files and folders
git add .

# Commit with a message
git commit -m "$COMMIT_MESSAGE"

# Create main branch (if not already)
git branch -M $BRANCH_NAME

# Push to GitHub (you'll be prompted for username/password or token)
git push -u origin $BRANCH_NAME

# ------------------------------------------
# SCRIPT END
# ------------------------------------------
