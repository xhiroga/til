#!/usr/bin/env sh

sudo sh -c 'echo "192.168.1.100 raspberry.pi" >> /etc/hosts'
# /etc/hostsにリダイレクトするためにはroot権限が必要だが、sudoでリダイレクトはできない。
# したがって、shにコマンド文字列を与えて解決する。
