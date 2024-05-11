default_port=4000

for ((i=1; i<=$#; i++)); do
  # If the argument is -s or --serve
  if [ "${!i}" = "-p" ] || [ "${!i}" = "--port" ]; then
    # Increment i to get the next argument, which should be the port number
    ((i++))
    port=${!i:-$default_port}
    cd _site
    ../node_modules/.bin/http-server -p $port
    break
  fi
done