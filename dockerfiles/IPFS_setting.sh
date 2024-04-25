#!/bin/sh
/usr/local/bin/ipfs init
/usr/local/bin/ipfs bootstrap rm --all
/usr/local/bin/ipfs config Addresses.API /ip4/0.0.0.0/tcp/5001
/usr/local/bin/ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8080
/usr/local/bin/ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["PUT", "GET", "POST", "OPTIONS"]'
/usr/local/bin/ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
/usr/local/bin/ipfs config --json API.HTTPHeaders.Access-Control-Allow-Credentials '["true"]'
/usr/local/bin/ipfs daemon
