#!/usr/bin/env poetry run python
from scapy.all import *

pkt = IP(dst="8.8.8.8")
pkt.show()
sr1(pkt)
