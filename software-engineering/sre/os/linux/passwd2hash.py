import crypt
import sys

passwd = sys.argv[1]
salt = sys.argv[2]
# /etc/shadow のハッシュと同じ内容を生成
print(crypt.crypt(passwd, salt))