#!/bin/bash

read -p "Are you sure [y/n]? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "sudo rm -rf /usr/local/include/mumps"
    sudo rm -rf /usr/local/include/mumps

    echo "sudo rm -rf /usr/local/lib/mumps"
    sudo rm -rf /usr/local/lib/mumps

    echo "sudo rm -f /etc/ld.so.conf.d/mumps.conf"
    sudo rm -f /etc/ld.so.conf.d/mumps.conf

    echo "sudo ldconfig"
    sudo ldconfig

    echo "DONE"
else
    echo "..cancelled.."
fi
