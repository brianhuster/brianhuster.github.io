#!/bin/bash
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