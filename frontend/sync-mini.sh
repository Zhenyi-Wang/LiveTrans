#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Sync frontend build to mini server"
    echo
    echo "Options:"
    echo "  -b, --build    Build project before syncing"
    echo "  -h, --help     Display this help message"
    exit 0
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--build) 
            echo "Building project..."
            yarn build
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *) 
            echo "Unknown parameter: $1"
            echo "Use -h or --help to see available options"
            exit 1
            ;;
    esac
done

rsync -avz --delete /home/zhenyi/ownprojects/livetrans/frontend/docker-compose.yml mini:/opt/livetrans/docker-compose.yml
rsync -avz --delete /home/zhenyi/ownprojects/livetrans/frontend/.output mini:/opt/livetrans/
ssh mini "docker restart livetrans"