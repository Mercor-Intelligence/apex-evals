#!/bin/bash
# xurls Setup Script
# Installs xurls for URL extraction (requires Go 1.22+)

if command -v go &> /dev/null; then
    echo "Installing xurls..."
    go install mvdan.cc/xurls/v2/cmd/xurls@latest
    
    if [ -f "$HOME/go/bin/xurls" ]; then
        echo "xurls installed successfully to $HOME/go/bin/xurls"
    else
        echo "Error: xurls installation failed"
        exit 1
    fi
else
    echo "Error: Go not found. Install Go first."
    echo "macOS: brew install go"
    echo "Linux: https://go.dev/doc/install"
    exit 1
fi

echo ""
if command -v xurls &> /dev/null; then
    echo "Setup complete. xurls is ready to use."
else
    echo "Next step: Add xurls to your PATH"
    echo ""
    echo "Run this command:"
    echo "  echo 'export PATH=\"\$HOME/go/bin:\$PATH\"' >> ~/.zshrc && source ~/.zshrc"
    echo ""
    echo "Then verify with: xurls --version"
fi

