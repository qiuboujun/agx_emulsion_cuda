#!/bin/bash
# Script to copy filtered CSV files to OFX plugin directory
sudo mkdir -p /usr/OFX/Plugins/data/
sudo cp filtered_*_data.csv /usr/OFX/Plugins/data/
echo 'Filtered CSV files copied to /usr/OFX/Plugins/data/'
