#!/bin/bash
for arg in "$@"; do
  if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
    echo "Usage: ./build.sh [-s|--serve [port]]"
    echo ""
    echo "Options:"
    echo "-s, --serve: After building the site, serve it using a Node.js HTTP server. If a port number is provided, the server will listen on that port. If no port number is provided, the server will listen on port 4000."
    exit 0
  fi
done

bundle exec jekyll build
node assets/server/search_index.js

default_port=4000

for ((i=1; i<=$#; i++)); do
  # If the argument is -s or --serve
  if [ "${!i}" = "-s" ] || [ "${!i}" = "--serve" ]; then
    # Increment i to get the next argument, which should be the port number
    ((i++))
    port=${!i:-$default_port}
    cd _site
    ../node_modules/.bin/http-server -p $port
    break
  fi
done