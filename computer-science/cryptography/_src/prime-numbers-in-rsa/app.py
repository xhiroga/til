from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# 公開鍵を読み込む
with open("id_rsa.pub", "rb") as key_file:
    public_key = serialization.load_ssh_public_key(
        key_file.read(),
        backend=default_backend()
    )

# 秘密鍵を読み込む
with open("id_rsa", "rb") as key_file:
    private_key = serialization.load_pem_private_key(
        key_file.read(),
        password=None,
        backend=default_backend()
    )

# 公開鍵からn, eを取得
public_numbers = public_key.public_numbers()
n = public_numbers.n
e = public_numbers.e

# 秘密鍵からp, q, dを取得
private_numbers = private_key.private_numbers()
p = private_numbers.p
q = private_numbers.q
d = private_numbers.d

# nがpとqの積であることを検証
n_check = p * q == n

# ed ≡ 1 (mod (p-1)(q-1))を検証
ed_check = (e * d) % ((p - 1) * (q - 1)) == 1

print("n is the product of p and q:", n_check)
print("ed ≡ 1 (mod (p-1)(q-1)):", ed_check)
