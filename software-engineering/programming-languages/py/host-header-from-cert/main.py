import ssl
from os import environ

import OpenSSL

host_ip = environ.get("HOST_IP")

cert = ssl.get_server_certificate((host_ip, 443))

x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, cert)
components = x509.get_subject().get_components()

hostname_byte = components[0][1]
print(hostname_byte.decode('UTF-8'))
