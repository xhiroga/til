import sys

dotted_decimal_addr = sys.argv[1]
binary_addr = ""
for dec in str.split(dotted_decimal_addr, "."):
    binary_addr = binary_addr + str("{0:b}".format(int(dec)))
print("Binary IP Address:   ", binary_addr)

subnet_mask = sys.argv[2]
subnet_mask = int(subnet_mask)
print("Network Part:        ", binary_addr[:subnet_mask])
print("Host Part:           ", binary_addr[subnet_mask:])