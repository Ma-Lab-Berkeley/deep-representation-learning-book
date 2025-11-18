#!/bin/bash

# Book v2 Sync Functions
# Add these to your ~/.bashrc or ~/.zshrc by running:
#   cat bookv2_sync_aliases.sh >> ~/.bashrc  # or ~/.zshrc
# Then reload your shell: source ~/.bashrc

# REQUIRED SETUP!
# OVERLEAF:
# https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/git-integration-authentication-tokens
# Follow the above guide to allow git access to Overleaf projects.
#
# GITHUB:
# Add a SSH authentication token to your Github
# And be added as a collaborator on the book repo to have push/pull access
# For this, you can be in the Ma-Lab-Berkeley org, or manually be a collaborator.
# Ask Druv or Sam.

# 1. Initial setup: Clone from Overleaf and add GitHub remote
bookv2_setup() {
    echo "Cloning from Overleaf..."
    git clone https://git@git.overleaf.com/691bd155731668031f326b2c book-v2 || {
        echo "✗ ERROR: Failed to clone repository"
        return 1
    }

    cd book-v2 || return 1

    echo "Adding GitHub remote..."
    git remote add github git@github.com:Ma-Lab-Berkeley/deep-representation-learning-book.git || {
        echo "✗ ERROR: Failed to add remote"
        return 1
    }

    git fetch github || {
        echo "✗ ERROR: Failed to fetch from GitHub"
        return 1
    }

    echo "✓ Setup complete! Already in book-v2 directory."
    echo "Note: Make sure you're on the correct branch (git checkout v2-preview)"
}

# 2. Sync GitHub → Overleaf (pull from GitHub, push to Overleaf master)
bookv2_gh_to_ol() {
    echo "Fetching from GitHub..."
    git fetch github || {
        echo "✗ ERROR: Failed to fetch from GitHub"
        return 1
    }

    echo "Merging github/v2-preview..."
    git merge github/v2-preview || {
        echo "✗✗✗ PANIC: MERGE CONFLICT! ✗✗✗"
        echo "Fix conflicts manually, then run:"
        echo "  git merge --continue  # after fixing"
        echo "  git merge --abort     # to cancel"
        return 1
    }

    echo "Pushing to Overleaf master..."
    # Overleaf only accepts pushes to master branch
    git push origin HEAD:master || {
        echo "✗ ERROR: Failed to push to Overleaf"
        return 1
    }

    echo "✓ Successfully synced GitHub → Overleaf"
}

# 3. Sync Overleaf → GitHub (pull from Overleaf master, push to GitHub v2-preview)
bookv2_ol_to_gh() {
    echo "Fetching from Overleaf..."
    git fetch origin || {
        echo "✗ ERROR: Failed to fetch from Overleaf"
        return 1
    }

    echo "Merging from Overleaf master..."
    git merge origin/master || {
        echo "✗✗✗ PANIC: MERGE CONFLICT! ✗✗✗"
        echo "Fix conflicts manually, then run:"
        echo "  git merge --continue  # after fixing"
        echo "  git merge --abort     # to cancel"
        return 1
    }

    echo "Pushing to GitHub v2-preview..."
    git push github HEAD:v2-preview || {
        echo "✗ ERROR: Failed to push to GitHub"
        return 1
    }

    echo "✓ Successfully synced Overleaf → GitHub"
}

# 4. Full bidirectional sync (Overleaf → GitHub, then GitHub → Overleaf)
bookv2_sync() {
    echo "═══════════════════════════════════════"
    echo "Starting bidirectional sync..."
    echo "═══════════════════════════════════════"

    echo ""
    echo "Step 1/2: Overleaf → GitHub"
    echo "───────────────────────────────────────"
    bookv2_ol_to_gh || {
        echo "✗✗✗ Sync failed at step 1 ✗✗✗"
        return 1
    }

    echo ""
    echo "Step 2/2: GitHub → Overleaf"
    echo "───────────────────────────────────────"
    bookv2_gh_to_ol || {
        echo "✗✗✗ Sync failed at step 2 ✗✗✗"
        return 1
    }

    echo ""
    echo "═══════════════════════════════════════"
    echo "✓✓✓ Full sync complete! ✓✓✓"
    echo "═══════════════════════════════════════"
}

# Quick reference:
# bookv2_setup      - Initial setup (run once)
# bookv2_gh_to_ol   - Sync GitHub → Overleaf
# bookv2_ol_to_gh   - Sync Overleaf → GitHub
# bookv2_sync       - Full bidirectional sync
