#!/bin/sh

which python3
if [ $? -ne 0 ]; then
  echo "Python3 is not installed. Please install Python3."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

if [ ! -d "models" ]; then
  echo "Creating models directory..."
  mkdir models
fi

which direnv
if [ $? -ne 0 ]; then
  echo "Do you want to install direnv to automatically load your venv each time you navigate to this folder? (y/N)"
  read -r answer
  if [ "$answer" = "Y" ] || [ "$answer" = "y" ]; then
    echo "Installing direnv..."
    if [ "$(uname)" = "Darwin" ]; then
      brew install direnv
    elif [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
      sudo apt-get install direnv
    fi
    direnv allow
  else
    echo "Okay, don't forget to run 'source .envrc'"
    exit 1
  fi
fi
