#!/bin/bash

default_port=4000
port=$default_port

echo "Starting script..."

for ((i=1; i<=$#; i++)); do
  echo "Checking argument $i: ${!i}"
  # If the argument is -p or --port
  if [ "${!i}" = "-p" ] || [ "${!i}" = "--port" ]; then
    echo "Port argument found"
    # Increment i to get the next argument, which should be the port number
    ((i++))
    port=${!i:-$default_port}
    echo "Using port: $port"
    break
  fi
done

if [ -d "_site" ]; then
  cd _site
  echo "Changed directory to _site. Starting HTTP server on port $port..."
  ../node_modules/.bin/http-server -p $port
else
  echo "Directory _site does not exist."
fi

echo "Script execution completed."
